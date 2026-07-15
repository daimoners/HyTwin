"""
HyTwin 2.0 — Real-Time Web Dashboard
======================================
FastAPI + WebSocket server that runs the advanced simulation in a
background thread and broadcasts live state to connected browsers.

Architecture
------------
  Main thread   : FastAPI (Uvicorn)  →  serves static HTML + REST API
  Sim thread    : SimulationEngine loop  →  puts GridState on a queue
  WS broadcaster: pulls from queue  →  pushes JSON to all WS clients

REST API  (all relative to /)
  GET  /              → serve dashboard HTML
  POST /control       → change controller type { "type": "classical"|"rl"|"none" }
  POST /sim/start     → start / resume simulation
  POST /sim/stop      → pause simulation
  POST /sim/reset     → reset + restart
  GET  /status        → current engine status JSON

WebSocket
  ws://host:port/ws   → stream of GridStateMessage JSON at every sim step
"""

import asyncio
import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# State message schema
# ============================================================================

def _gs_to_dict(gs, weather: Dict[str, Any] = None, step: int = 0) -> Dict[str, Any]:
    """Serialise GridState to a JSON-compatible dict."""
    return {
        "ts": gs.timestamp.isoformat(),
        "step": step,
        # Power flows
        "wind_kw":   round(gs.wind_power_kw, 1),
        "pv_kw":     round(gs.pv_power_kw, 1),
        "el_kw":     round(gs.electrolyzer_power_kw, 1),
        "fc_kw":     round(gs.fuel_cell_power_kw, 1),
        "load_kw":   round(gs.load_kw, 1),
        "grid_exchange_kw": round(gs.grid_exchange_kw, 1),
        "grid_connection_kw": round(gs.grid_connection_kw, 1),
        # H₂
        "h2_prod_kg_h": round(gs.h2_production_kg_h, 3),
        "h2_cons_kg_h": round(gs.h2_consumption_kg_h, 3),
        "h2_storage_kg": round(gs.h2_storage_kg, 2),
        "h2_soc":       round(gs.h2_soc, 4),
        "h2_pressure_bar": round(gs.h2_pressure_bar, 1),
        # Economics
        "price_eur_kwh": round(gs.energy_price_eur_kwh, 4),
        "step_cost_eur": round(gs.step_cost_eur, 4),
        "cum_cost_eur":  round(gs.cumulative_cost_eur, 2),
        "co2_kg_step":   round(gs.step_co2_kg, 4),
        # KPIs
        "renewable_fraction": round(gs.renewable_fraction, 4),
        "self_sufficiency": round(gs.grid_self_sufficiency, 4),
        "efficiency": round(gs.system_efficiency, 4),
        "health": round(gs.overall_health, 4),
        "grid_available": gs.grid_available,
        "anomaly_nodes": gs.anomaly_nodes,
        # Weather context
        "wind_ms":    round(weather.get("wind_speed_ms", weather.get("wind_ms", 0)), 2) if weather else 0,
        "ghi_wm2":    round(weather.get("ghi_wm2", 0), 1) if weather else 0,
        "temp_c":     round(weather.get("temperature_c", weather.get("temp_c", 15)), 1) if weather else 15,
    }


# ============================================================================
# Simulation worker thread
# ============================================================================

class SimulationWorker:
    """Runs the SimulationEngine in a background thread."""

    def __init__(
        self,
        config_path: str,
        dt_seconds: float = 600.0,
        speed_factor: float = 0.0,
        seed: int = 42,
        queue_maxsize: int = 200,
    ) -> None:
        self._config_path = config_path
        self._dt = dt_seconds
        self._speed_factor = speed_factor
        self._seed = seed
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._running = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._controller_type = "classical"
        self._controller_lock = threading.Lock()
        self._speed_lock = threading.Lock()
        self._step = 0
        self._last_weather: Dict[str, Any] = {}
        # Per-component history and sensor snapshots
        self._node_history: Dict[str, list] = {}
        self._last_node_snapshot: Dict[str, Any] = {}
        self._last_sensor_readings: Dict[str, Any] = {}
        self._node_specs: Dict[str, Any] = {}
        self._history_lock = threading.Lock()

    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._running.set()
            return
        self._stop.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("SimulationWorker started")

    def stop(self) -> None:
        self._running.clear()

    def reset(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._step = 0
        with self._history_lock:
            self._node_history.clear()
            self._last_node_snapshot.clear()
            self._last_sensor_readings.clear()
        self.start()

    def set_controller(self, ctrl_type: str) -> None:
        with self._controller_lock:
            self._controller_type = ctrl_type

    def set_speed_factor(self, speed_factor: float) -> float:
        sf = max(0.0, min(300.0, float(speed_factor)))
        with self._speed_lock:
            self._speed_factor = sf
        return sf

    def get_speed_factor(self) -> float:
        with self._speed_lock:
            return self._speed_factor

    def get_state_nowait(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def get_queue(self) -> queue.Queue:
        return self._q

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        import sys
        sys.path.insert(0, str(ROOT))
        from hytwin.simulation.scenario import Scenario
        from hytwin.simulation.engine import SimulationEngine
        from hytwin.models.energy_cost import EnergyCostModel
        from hytwin.control.classical_controller import ClassicalController

        np.random.seed(self._seed)
        scenario = Scenario.from_yaml(self._config_path)
        scenario.dt_seconds = self._dt
        cost_model = EnergyCostModel(scenario.grid_config.get("energy_cost", {}))
        classical_ctrl = ClassicalController(
            grid_config=scenario.grid_config,
            cost_model=cost_model,
        )

        def _schedule(step, ts, gs):
            with self._controller_lock:
                ctrl_type = self._controller_type
            if ctrl_type == "classical" and gs is not None:
                return classical_ctrl.compute_actions(gs, timestamp=ts)
            return {}

        scenario.set_schedule(_schedule)

        engine = SimulationEngine(
            scenario=scenario,
            dt_seconds=self._dt,
            speed_factor=0.0,
        )

        engine._running = True
        while not self._stop.is_set():
            if not self._running.is_set():
                time.sleep(0.05)
                continue
            try:
                step_t0 = time.perf_counter()
                gs, ctx = engine.step_once()
                self._last_weather = ctx.get("weather", {})
                payload = _gs_to_dict(gs, self._last_weather, self._step)
                self._step += 1

                # ── Per-node state snapshot ─────────────────────────────
                node_snap: Dict[str, Any] = {}
                for nid, node in engine._twin._nodes.items():
                    if node.state is not None:
                        st = node.state
                        node_snap[nid] = {
                            "values": {
                                k: round(float(v), 4)
                                for k, v in st.fused_values.items()
                            },
                            "health": round(float(st.health_score), 3),
                            "anomaly": round(float(st.anomaly_score), 3),
                        }

                # ── Sensor readings snapshot ────────────────────────────
                sensor_snap: Dict[str, Any] = {}
                if engine._sensor_mgr is not None:
                    for sid, (sens, mkey) in engine._sensor_mgr._sensors.items():
                        lr = sens.last_reading
                        if lr is not None:
                            sensor_snap[sid] = {
                                "value": round(float(lr.value), 4),
                                "quality": round(float(lr.quality), 3),
                                "true_value": (
                                    round(float(lr.true_value), 4)
                                    if lr.true_value is not None else None
                                ),
                                "status": lr.status.name,
                            }

                # ── Populate static specs once ──────────────────────────
                if not self._node_specs:
                    for nid, node in engine._twin._nodes.items():
                        self._node_specs[nid] = {
                            "type": type(node.model).__name__,
                            "params": {
                                k: (float(v) if isinstance(v, (int, float)) else v)
                                for k, v in node.model.params.items()
                                if isinstance(v, (int, float, str, bool))
                            },
                        }

                # ── Update rolling histories (thread-safe) ──────────────
                ts_str = gs.timestamp.isoformat()
                with self._history_lock:
                    self._last_node_snapshot = node_snap
                    self._last_sensor_readings = sensor_snap
                    for nid, state in node_snap.items():
                        hist = self._node_history.setdefault(nid, [])
                        hist.append({
                            "ts": ts_str,
                            "values": state["values"],
                            "health": state["health"],
                            "anomaly": state["anomaly"],
                        })
                        if len(hist) > 288:
                            hist.pop(0)

                # ── Attach to WS payload ────────────────────────────────
                payload["nodes"] = node_snap
                payload["sensors"] = sensor_snap
                try:
                    self._q.put_nowait(payload)
                except queue.Full:
                    # Drop oldest if queue full (dashboard can't keep up)
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    self._q.put_nowait(payload)

                speed = self.get_speed_factor()
                if speed > 0:
                    target_step_wall_s = self._dt / speed
                    elapsed = time.perf_counter() - step_t0
                    if elapsed < target_step_wall_s:
                        time.sleep(target_step_wall_s - elapsed)
            except Exception as exc:
                logger.exception("SimulationWorker error: %s", exc)
                time.sleep(1)


# ============================================================================
# FastAPI application factory
# ============================================================================

def create_app(
    config_path: str,
    dt_seconds: float = 600.0,
    speed_factor: float = 0.0,
    seed: int = 42,
):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        import uvicorn
    except ImportError as e:
        raise ImportError(
            f"Dashboard requires fastapi + uvicorn: pip install fastapi uvicorn\n{e}"
        )

    app = FastAPI(title="HyTwin 2.0 Dashboard")
    worker = SimulationWorker(config_path, dt_seconds, speed_factor, seed)
    active_ws: Set[WebSocket] = set()

    # ── Serve static files ────────────────────────────────────────────
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── Dashboard HTML ────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = Path(__file__).parent / "static" / "index.html"
        return html_path.read_text(encoding="utf-8")

    # ── REST control ──────────────────────────────────────────────────
    @app.post("/sim/start")
    async def sim_start():
        worker.start()
        return {"status": "started"}

    @app.post("/sim/stop")
    async def sim_stop():
        worker.stop()
        return {"status": "stopped"}

    @app.post("/sim/reset")
    async def sim_reset():
        # worker.reset() calls thread.join() — must run in a thread pool
        # to avoid blocking the asyncio event loop
        import asyncio
        await asyncio.get_event_loop().run_in_executor(None, worker.reset)
        return {"status": "reset"}

    @app.post("/control")
    async def set_control(body: dict):
        ctrl = body.get("type", "classical")
        worker.set_controller(ctrl)
        return {"status": "ok", "controller": ctrl}

    @app.get("/status")
    async def status():
        return {
            "running": worker._running.is_set(),
            "step": worker._step,
            "controller": worker._controller_type,
            "speed_factor": worker.get_speed_factor(),
        }

    @app.post("/sim/speed")
    async def sim_speed(body: dict):
        speed_factor = body.get("speed_factor", 0.0)
        current = worker.set_speed_factor(speed_factor)
        return {"status": "ok", "speed_factor": current}

    @app.get("/component/{node_id}")
    async def component_detail(node_id: str):
        from fastapi import HTTPException
        specs = worker._node_specs.get(node_id)
        if specs is None:
            raise HTTPException(status_code=404, detail=f"Component '{node_id}' not found")
        with worker._history_lock:
            history = list(worker._node_history.get(node_id, []))
            current = dict(worker._last_node_snapshot.get(node_id, {}))
            sensors = {
                k: v for k, v in worker._last_sensor_readings.items()
                if k.startswith(node_id + ".")
            }
        return {
            "id": node_id,
            "type": specs["type"],
            "params": specs["params"],
            "current": current,
            "history": history[-144:],
            "sensors": sensors,
        }

    # ── WebSocket broadcaster ─────────────────────────────────────────
    # Rate-limit: send at most 1 frame every WS_INTERVAL seconds (~5 fps)
    WS_INTERVAL = 0.20

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        active_ws.add(websocket)
        logger.info("WS client connected (%d total)", len(active_ws))
        last_send = 0.0
        try:
            while True:
                payload = worker.get_state_nowait()
                now = asyncio.get_event_loop().time()
                if payload is not None and (now - last_send) >= WS_INTERVAL:
                    msg = json.dumps(payload)
                    # Broadcast to all clients
                    dead = set()
                    for ws in list(active_ws):
                        try:
                            await ws.send_text(msg)
                        except Exception:
                            dead.add(ws)
                    active_ws.difference_update(dead)
                    last_send = now
                else:
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            active_ws.discard(websocket)
            logger.info("WS client disconnected (%d total)", len(active_ws))

    @app.on_event("startup")
    async def startup():
        worker.start()

    @app.on_event("shutdown")
    async def shutdown():
        worker.stop()

    return app, uvicorn


def run(
    config_path: str = None,
    dt_seconds: float = 600.0,
    speed_factor: float = 0.0,
    port: int = 8050,
    seed: int = 42,
) -> None:
    if config_path is None:
        config_path = str(ROOT / "config" / "advanced_grid.yaml")

    print(f"\n{'='*60}")
    print(f"  HyTwin 2.0 — Real-Time Dashboard")
    print(f"{'='*60}")
    print(f"  URL:    http://localhost:{port}")
    print(f"  Config: {config_path}")
    print(f"  Speed:  {'max' if speed_factor == 0 else f'{speed_factor}×'}")
    print(f"  Press Ctrl-C to stop\n")

    app, uvicorn = create_app(config_path, dt_seconds, speed_factor, seed)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    run()
