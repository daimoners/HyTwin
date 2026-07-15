"""
Simulation Engine
==================
Central orchestrator for the digital twin simulation loop.

The engine wires together:
  · SimulationClock    — time advance
  · WeatherModel       — environmental inputs
  · GridTwin           — physics + digital twin
  · SensorManager      — virtual sensors
  · StateManager       — state store
  · EventBus           — inter-component messaging
  · DataRecorder       — time-series logging

Operating modes (set in config):
  'simulation' — run as fast as possible (offline analysis)
  'realtime'   — paced to wall clock (for HMI / integration test)

The engine exposes a simple ``run()`` method for batch runs and a
``step_once()`` method for interactive / REPL use.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core.event_bus import EventBus
from ..core.state_manager import StateManager
from ..core.time_engine import SimulationClock
from ..digital_twin.grid_twin import GridTwin, GridState
from ..models.energy_cost import EnergyCostModel
from ..sensors.sensor_manager import SensorManager
from ..weather.weather_model import WeatherModel
from .scenario import Scenario
from ..data.time_series import TimeSeriesRecorder

logger = logging.getLogger(__name__)


StepCallback = Callable[[int, GridState, Dict[str, Any]], None]


class SimulationEngine:
    """
    Main simulation engine.

    Parameters
    ----------
    scenario : Scenario
        Fully configured scenario (grid config, weather params, sensors).
    dt_seconds : float
        Duration of each simulation step [s], default 60.
    speed_factor : float
        Wall-clock speed. 0 = as-fast-as-possible.
    recorder : TimeSeriesRecorder, optional
        If provided, each step is logged.
    callbacks : list of StepCallback, optional
        Called after each step: fn(step_index, grid_state, context).
    """

    def __init__(
        self,
        scenario: "Scenario",
        dt_seconds: float = 60.0,
        speed_factor: float = 0.0,
        recorder: Optional[TimeSeriesRecorder] = None,
        callbacks: Optional[List[StepCallback]] = None,
        cost_model: Optional["EnergyCostModel"] = None,
    ) -> None:
        self._scenario = scenario
        self._dt = dt_seconds
        self._recorder = recorder
        self._callbacks: List[StepCallback] = callbacks or []

        # Core services
        self._bus = EventBus()
        self._sm = StateManager()
        self._clock = SimulationClock(
            start_time=scenario.start_time,
            dt_seconds=dt_seconds,
            speed_factor=speed_factor,
        )

        # Weather
        self._weather = WeatherModel(**scenario.weather_params)

        # Grid twin — pass shared cost_model if provided for RNG synchronization
        self._twin = GridTwin(scenario.grid_config, self._bus, self._sm, cost_model=cost_model)
        self._twin.build()

        # Sensors
        self._sensor_mgr: Optional[SensorManager] = None
        if scenario.sensor_config:
            self._sensor_mgr = _build_sensor_manager(
                scenario.sensor_config, self._bus
            )

        self._step_count = 0
        self._running = False

        logger.info(
            "SimulationEngine ready — dt=%.0fs, scenario=%s",
            dt_seconds, scenario.name,
        )

    # ------------------------------------------------------------------
    # Run methods
    # ------------------------------------------------------------------

    def run(self, steps: int = 1440) -> List[GridState]:
        """Run *steps* simulation steps and return list of GridState."""
        self._running = True
        results: List[GridState] = []

        logger.info("Starting simulation — %d steps (%.1f h)",
                    steps, steps * self._dt / 3600)

        for i in range(steps):
            if not self._running:
                break
            gs, ctx = self.step_once()
            results.append(gs)

            if (i + 1) % max(1, steps // 10) == 0:
                logger.info(
                    "Step %d/%d | %s | RF=%.0f%% SOC=%.2f H2=%.1fkg",
                    i + 1, steps,
                    gs.timestamp.strftime("%Y-%m-%d %H:%M"),
                    gs.renewable_fraction * 100,
                    gs.h2_soc,
                    gs.h2_storage_kg,
                )

        self._running = False
        logger.info("Simulation complete — %d steps", self._step_count)
        return results

    def step_once(self) -> tuple[GridState, Dict[str, Any]]:
        """Advance simulation by exactly one step."""
        ts = self._clock.tick()

        # Weather for this step
        weather = self._weather.step(ts)

        # Build control actions from scenario schedule (or RL agent if set)
        control_actions = self._scenario.get_control_actions(
            self._step_count, ts, self._twin.grid_state
        )

        # Collect sensor readings
        sensor_readings = None
        if self._sensor_mgr is not None:
            snap = self._sm.snapshot()
            sensor_readings = self._sensor_mgr.update(snap, timestamp=ts)

        # Advance twin
        gs = self._twin.step(
            self._dt, weather, control_actions,
            sensor_readings=sensor_readings,
            timestamp=ts,
        )

        # H2 tank charging: what electrolyzers produced this step
        # feed into tank; fuel cell draws from tank
        self._update_h2_storage(gs)

        # Record
        if self._recorder is not None:
            self._recorder.record(self._step_count, gs, weather)

        # Callbacks
        ctx = {"weather": weather, "control_actions": control_actions}
        for cb in self._callbacks:
            try:
                cb(self._step_count, gs, ctx)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error in step callback: %s", exc)

        self._step_count += 1
        return gs, ctx

    # ------------------------------------------------------------------
    # H2 flow management
    # ------------------------------------------------------------------

    def _update_h2_storage(self, gs: GridState) -> None:
        """Charge H2 tanks with production; discharge with FC consumption."""
        from ..models import HydrogenTankModel
        h2_produced_kg = gs.h2_production_kg_h * self._dt / 3600.0
        h2_consumed_kg = gs.h2_consumption_kg_h * self._dt / 3600.0

        # Collect all tanks and distribute H2 equally.
        # Use charge()/discharge() directly (not step()) to avoid applying
        # boiloff twice — boiloff is already applied in GridTwin.step().
        tank_nodes = [
            node for node in self._twin._nodes.values()
            if isinstance(node.model, HydrogenTankModel)
        ]
        if not tank_nodes:
            return
        n = len(tank_nodes)
        for node in tank_nodes:
            node.model.charge(h2_produced_kg / n, self._dt)
            node.model.discharge(h2_consumed_kg / n, self._dt)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._clock.reset(self._scenario.start_time)
        self._twin.reset()
        self._weather.reset()
        if self._sensor_mgr is not None:
            self._sensor_mgr.reset_all()
        self._sm.clear()
        self._step_count = 0
        self._running = False

    def set_rl_agent(self, agent) -> None:
        """Inject an RL agent; its predict() will override scenario actions."""
        self._scenario.set_agent(agent)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def clock(self) -> SimulationClock:
        return self._clock

    @property
    def twin(self) -> GridTwin:
        return self._twin

    @property
    def step_count(self) -> int:
        return self._step_count


# ------------------------------------------------------------------
# Sensor manager factory
# ------------------------------------------------------------------

def _build_sensor_manager(
    sensor_config: List[Dict[str, Any]],
    event_bus: EventBus,
    rng: Optional[np.random.Generator] = None,
) -> SensorManager:
    """
    Build a SensorManager from a list of sensor config dicts.

    If *rng* is given, each sensor gets an independent generator spawned from
    it (isolates this manager's noise/fault streams from any other manager
    running concurrently, e.g. a live simulation vs. a background RL
    training job). ``rng=None`` (default) keeps the legacy global-RNG
    behaviour.
    """
    from ..sensors import (
        SensorManager, PowerSensor, PressureSensor,
        TemperatureSensor, FlowSensor, HydrogenLevelSensor,
        WindSpeedSensor, IrradianceSensor,
    )
    from ..core.rng import spawn_generators

    mgr = SensorManager(event_bus)
    child_rngs = spawn_generators(rng, len(sensor_config))
    type_map = {
        "power": PowerSensor,
        "pressure": PressureSensor,
        "temperature": TemperatureSensor,
        "flow": FlowSensor,
        "h2_level": HydrogenLevelSensor,
        "wind_speed": WindSpeedSensor,
        "irradiance": IrradianceSensor,
    }

    # Mapping of generic YAML key → sensor-specific constructor kwarg
    _generic_alias: Dict[str, Dict[str, str]] = {
        "power":       {"noise_std": "noise_std_kw",  "drift_rate": "drift_rate_kw"},
        "pressure":    {"noise_std": "noise_std_bar", "drift_rate": "drift_rate_bar"},
        "temperature": {"noise_std": "noise_std_c",   "drift_rate": "drift_rate_c"},
        "flow":        {"noise_std": "noise_std_kg_h","drift_rate": "drift_rate_kg_h"},
        "h2_level":    {"noise_std": "noise_std_kg",  "drift_rate": "drift_rate_kg"},
        "wind_speed":  {"noise_std": "noise_std_ms",  "drift_rate": "drift_rate_ms"},
        "irradiance":  {"noise_std": "noise_std_wm2", "drift_rate": "drift_rate_wm2"},
    }

    for scfg, child_rng in zip(sensor_config, child_rngs):
        s_type = scfg["type"]
        cls = type_map.get(s_type)
        if cls is None:
            logger.warning("Unknown sensor type '%s'", s_type)
            continue
        alias = _generic_alias.get(s_type, {})
        kwargs = {}
        for k, v in scfg.items():
            if k in ("type", "model_key", "id"):
                continue
            kwargs[alias.get(k, k)] = v
        sensor = cls(sensor_id=scfg["id"], event_bus=event_bus, rng=child_rng, **kwargs)
        mgr.register(sensor, scfg["model_key"])

    return mgr
