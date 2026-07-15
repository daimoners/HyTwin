"""
demo_simulation.py
==================
Runs a complete 24-hour simulation of the H2 pilot grid using the
default_grid.yaml configuration and prints summary KPIs.

Usage
-----
    python demos/demo_simulation.py [--steps 144] [--dt 600] [--plot]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── make package importable when run directly ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.simulation.engine import SimulationEngine
from hytwin.data.time_series import TimeSeriesRecorder
from hytwin.visualization.plotter import plot_simulation_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HyTwin 2.0 — 24-h simulation demo")
    p.add_argument("--config", default=str(ROOT / "config" / "default_grid.yaml"),
                   help="Path to scenario YAML (default: config/default_grid.yaml)")
    p.add_argument("--steps", type=int, default=144, help="Number of time steps (default: 144 = 24 h @ dt=600 s)")
    p.add_argument("--dt", type=int, default=600, help="Time-step size in seconds (default: 600)")
    p.add_argument("--plot", action="store_true", help="Show matplotlib dashboard at end")
    p.add_argument("--save", default="", help="Save plot to this path (e.g. output/sim.png)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rule-based fallback schedule
# ---------------------------------------------------------------------------
def _rule_based_control(step: int, ts, grid_state) -> dict:
    """Simple rule: excess renewable → electrolyser; deficit → fuel cell."""
    if grid_state is None:
        return {"el_setpoint": 0.3, "fc_setpoint": 0.0, "demand_response": 0.0}

    soc = grid_state.h2_soc
    net_kw = grid_state.wind_power_kw + grid_state.pv_power_kw - grid_state.load_kw

    el_sp = 0.0
    fc_sp = 0.0
    dr = 0.0

    if net_kw > 20 and soc < 0.90:          # surplus → electrolysis
        el_sp = min(1.0, net_kw / 300.0)
    elif net_kw < -10 and soc > 0.10:       # deficit → fuel cell
        fc_sp = min(1.0, -net_kw / 150.0)
    elif net_kw < -50:                       # large deficit → demand response
        dr = min(0.5, (-net_kw - 50) / 200.0)

    return {"el_setpoint": el_sp, "fc_setpoint": fc_sp, "demand_response": dr}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    print("=" * 64)
    print("  HyTwin 2.0 — Simulation Demo")
    print("=" * 64)

    # ── Load scenario ────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    scenario = Scenario.from_yaml(str(config_path))
    scenario.set_schedule(_rule_based_control)
    print(f"  Scenario : {scenario.name}")
    print(f"  Steps    : {args.steps} × {args.dt} s = {args.steps * args.dt / 3600:.1f} h")
    print(f"  Location : {scenario.weather_params.get('latitude_deg', '?')}°N, "
          f"{scenario.weather_params.get('longitude_deg', '?')}°E")
    print()

    # ── Recorder ─────────────────────────────────────────────────────────────
    out_csv = ROOT / "output" / "demo_simulation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    recorder = TimeSeriesRecorder(csv_path=str(out_csv))

    # ── Engine ───────────────────────────────────────────────────────────────
    engine = SimulationEngine(
        scenario=scenario,
        dt_seconds=args.dt,
        speed_factor=0.0,    # as-fast-as-possible
        recorder=recorder,
    )

    # ── Run ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("  Running simulation …")
    records = engine.run(steps=args.steps)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f} s  ({len(records)} states recorded)\n")

    # ── KPI summary ──────────────────────────────────────────────────────────
    summary = recorder.summary()
    print("  ── KPI Summary ─────────────────────────────────────────────")
    fmt = "  {:40s} {:>10}"
    for k, v in summary.items():
        if isinstance(v, float):
            print(fmt.format(k, f"{v:.3f}"))
        else:
            print(fmt.format(k, str(v)))
    print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    if args.plot or args.save:
        df = recorder.to_dataframe()
        if df.empty:
            print("  [WARN] No data to plot.")
        else:
            save_path = args.save if args.save else None
            plot_simulation_results(
                records=records,
                title=f"HyTwin 2.0 — {scenario.name} (24 h)",
                save_path=save_path,
                show=args.plot,
            )

    print("  Output CSV :", out_csv)
    print("=" * 64)


if __name__ == "__main__":
    main()
