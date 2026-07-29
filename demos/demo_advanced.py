"""
demo_advanced.py
================
Comprehensive advanced demo of HyTwin.

NEW FEATURES demonstrated
--------------------------
  • Non-renewable grid connection with outages and capacity limits
  • Time-varying Italian PUN electricity price model
  • Multi-source dispatch: direct renewables / H₂ storage / grid
  • ClassicalController — cost-aware rule-based dispatch
  • PPO RL agent training on AdvancedH2GridEnv (optional)
  • Side-by-side comparison: Classical vs RL controller
  • Variable simulation speed (--speed-factor)

Modes
-----
  simulate   — run one simulation with classical control and print KPIs
  train_rl   — train a PPO agent then evaluate it (saves model)
  compare    — run Classical vs RL side by side, print comparison table
  dashboard  — start the Network Control Room dashboard (multi-site)

Usage examples
--------------
  python demos/demo_advanced.py --mode simulate --steps 144
  python demos/demo_advanced.py --mode train_rl --timesteps 100000
  python demos/demo_advanced.py --mode compare --steps 144
  python demos/demo_advanced.py --mode dashboard --port 8060
  python demos/demo_advanced.py --mode simulate --speed-factor 60
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── make package importable when run directly ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.simulation.scenario import Scenario
from hytwin.simulation.engine import SimulationEngine
from hytwin.digital_twin.grid_twin import GridState
from hytwin.models.energy_cost import EnergyCostModel
from hytwin.control.classical_controller import ClassicalController
from hytwin.data.time_series import TimeSeriesRecorder

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("demo_advanced")

DEFAULT_CONFIG = str(ROOT / "config" / "advanced_stress.yaml")
MODEL_DIR = ROOT / "output" / "rl_models"

# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HyTwin — Advanced multi-source H2 grid demo"
    )
    p.add_argument(
        "--mode", choices=["simulate", "train_rl", "compare", "dashboard"],
        default="simulate",
        help="Demo mode (default: simulate)",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Path to advanced scenario YAML (default: advanced_stress.yaml)")
    p.add_argument("--steps", type=int, default=144,
                   help="Simulation steps (default: 144 = 24 h @ dt=600 s)")
    p.add_argument("--dt", type=int, default=600,
                   help="Time-step in seconds (default: 600)")
    p.add_argument("--speed-factor", type=float, default=0.0,
                   help="Wall-clock speed factor: 0=as-fast-as-possible, "
                        "1=real-time, 60=1 sim-min per real second")
    p.add_argument("--timesteps", type=int, default=100_000,
                   help="RL training timesteps (mode=train_rl)")
    p.add_argument("--model-path", default=str(MODEL_DIR / "advanced_ppo"),
                   help="Path to save/load RL model")
    p.add_argument("--port", type=int, default=8050,
                   help="Dashboard port (mode=dashboard)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="Show matplotlib plots")
    return p.parse_args()


# ============================================================================
# Simulation runner helpers
# ============================================================================

def _load_scenario(config_path: str, dt: int, speed_factor: float) -> "Scenario":
    scenario = Scenario.from_yaml(config_path)
    scenario.dt_seconds = dt
    return scenario


def _run_simulation(
    scenario: "Scenario",
    steps: int,
    dt: int,
    speed_factor: float,
    record: bool = True,
    label: str = "sim",
    cost_model: Optional[EnergyCostModel] = None,
    controller: Optional[Any] = None,
) -> Tuple[List[GridState], Dict[str, float]]:
    """
    Run *steps* of simulation.

    Parameters
    ----------
    controller : ClassicalController | RLController | None
        If given, its ``compute_actions`` output is used as scenario schedule.
    """
    recorder = TimeSeriesRecorder() if record else None

    if controller is not None:
        if hasattr(controller, "reset"):
            controller.reset()
        def _schedule(step, ts, gs):
            if gs is None:
                return {}
            return controller.compute_actions(gs, timestamp=ts)
        scenario.set_schedule(_schedule)

    engine = SimulationEngine(
        scenario=scenario,
        dt_seconds=dt,
        speed_factor=speed_factor,
        recorder=recorder,
    )

    t0 = time.perf_counter()
    results = engine.run(steps=steps)
    elapsed = time.perf_counter() - t0

    kpis = _compute_kpis(results, elapsed, dt, label)
    return results, kpis


def _compute_kpis(
    results: List[GridState],
    elapsed_wall_s: float,
    dt_seconds: int,
    label: str,
) -> Dict[str, float]:
    """Aggregate KPIs from list of GridState."""
    if not results:
        return {}

    rf_mean      = np.mean([gs.renewable_fraction for gs in results])
    ss_mean      = np.mean([gs.grid_self_sufficiency for gs in results])
    h2_soc_mean  = np.mean([gs.h2_soc for gs in results])
    h2_soc_min   = np.min([gs.h2_soc for gs in results])
    h2_soc_max   = np.max([gs.h2_soc for gs in results])
    total_cost   = results[-1].cumulative_cost_eur if results else 0.0
    total_co2    = sum(gs.step_co2_kg for gs in results)
    price_mean   = np.mean([gs.energy_price_eur_kwh for gs in results])
    price_peak   = np.max([gs.energy_price_eur_kwh for gs in results])
    grid_import  = np.sum([max(0.0, gs.grid_connection_kw) for gs in results]) * dt_seconds / 3600.0
    grid_avail   = np.mean([1.0 if gs.grid_available else 0.0 for gs in results])
    health_mean  = np.mean([gs.overall_health for gs in results])
    wind_total   = sum(gs.wind_power_kw for gs in results) * dt_seconds / 3600.0
    pv_total     = sum(gs.pv_power_kw for gs in results) * dt_seconds / 3600.0
    el_total     = sum(gs.electrolyzer_power_kw for gs in results) * dt_seconds / 3600.0
    fc_total     = sum(gs.fuel_cell_power_kw for gs in results) * dt_seconds / 3600.0
    load_total   = sum(gs.load_kw for gs in results) * dt_seconds / 3600.0
    n_outages    = sum(1 for gs in results if not gs.grid_available)

    sim_hours = len(results) * dt_seconds / 3600.0
    rtf = sim_hours / max(1e-9, elapsed_wall_s / 3600.0)   # sim-hours per wall-hour

    return {
        "label": label,
        "steps": len(results),
        "sim_hours": sim_hours,
        "wall_s": elapsed_wall_s,
        "rtf": rtf,
        "renewable_fraction_pct": rf_mean * 100,
        "self_sufficiency_pct": ss_mean * 100,
        "h2_soc_mean": h2_soc_mean,
        "h2_soc_min": h2_soc_min,
        "h2_soc_max": h2_soc_max,
        "total_cost_eur": total_cost,
        "total_co2_kg": total_co2,
        "price_mean_eur_kwh": price_mean,
        "price_peak_eur_kwh": price_peak,
        "grid_import_kwh": grid_import,
        "grid_availability_pct": grid_avail * 100,
        "n_grid_outage_steps": n_outages,
        "wind_kwh": wind_total,
        "pv_kwh": pv_total,
        "el_kwh": el_total,
        "fc_kwh": fc_total,
        "load_kwh": load_total,
        "health_mean": health_mean,
    }


# ============================================================================
# Mode: simulate
# ============================================================================

def mode_simulate(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  HyTwin — Advanced Simulation  (Classical Controller)")
    print("=" * 70)

    scenario = _load_scenario(args.config, args.dt, args.speed_factor)
    cost_model = EnergyCostModel(scenario.grid_config.get("energy_cost", {}))
    controller = ClassicalController(
        grid_config=scenario.grid_config,
        cost_model=cost_model,
    )

    print(f"\n  Config    : {args.config}")
    print(f"  Steps     : {args.steps} × {args.dt}s  "
          f"= {args.steps * args.dt / 3600:.1f} h simulated")
    print(f"  Speed     : {'as-fast-as-possible' if args.speed_factor == 0 else f'{args.speed_factor}×'}")
    print(f"  Start     : {scenario.start_time}")
    print(f"  Controller: {controller.name}\n")

    results, kpis = _run_simulation(
        scenario, args.steps, args.dt, args.speed_factor,
        label="classical", cost_model=cost_model, controller=controller,
    )

    _print_step_trace(results, args.dt)
    _print_kpis(kpis)

    if args.plot:
        _plot_results({"Classical": results}, args.dt)


def _print_step_trace(
    results: List[GridState], dt: int, max_rows: int = 24
) -> None:
    """Print a table of hourly simulation snapshots."""
    stride = max(1, len(results) // max_rows)
    print("\n  ── Hourly Trace ─────────────────────────────────────────────────────")
    hdr = (
        f"  {'Time':>5}  {'Wind':>6}  {'PV':>6}  {'EL':>6}  {'FC':>5}  "
        f"{'Load':>6}  {'Grid':>7}  {'Price':>6}  {'SOC':>5}  "
        f"{'RF':>5}  {'Cost':>7}  {'Avail'}"
    )
    print(hdr)
    print("  " + "─" * 85)
    for i, gs in enumerate(results):
        if i % stride != 0:
            continue
        avail = "✓" if gs.grid_available else "✗ OUTAGE"
        print(
            f"  {gs.timestamp.strftime('%H:%M'):>5}  "
            f"{gs.wind_power_kw:6.0f}  "
            f"{gs.pv_power_kw:6.0f}  "
            f"{gs.electrolyzer_power_kw:6.0f}  "
            f"{gs.fuel_cell_power_kw:5.0f}  "
            f"{gs.load_kw:6.0f}  "
            f"{gs.grid_connection_kw:+7.0f}  "
            f"{gs.energy_price_eur_kwh:6.3f}  "
            f"{gs.h2_soc:5.3f}  "
            f"{gs.renewable_fraction:5.1%}  "
            f"{gs.cumulative_cost_eur:7.2f}  "
            f"  {avail}"
        )
    print()


def _print_kpis(kpis: Dict[str, float], label: str = "") -> None:
    lbl = f"  [{kpis.get('label', label)}]" if kpis.get("label") else ""
    print(f"\n  ── KPI Summary{lbl} ──────────────────────────────────────────────")
    rows = [
        ("Simulation period",   f"{kpis['sim_hours']:.1f} h  ({kpis['steps']} steps)"),
        ("Wall-clock time",     f"{kpis['wall_s']:.2f} s  (RTF {kpis['rtf']:.0f}×)"),
        ("Renewable fraction",  f"{kpis['renewable_fraction_pct']:.1f} %"),
        ("Self-sufficiency",    f"{kpis['self_sufficiency_pct']:.1f} %"),
        ("H₂ SOC  mean/min/max",
         f"{kpis['h2_soc_mean']:.3f} / {kpis['h2_soc_min']:.3f} / {kpis['h2_soc_max']:.3f}"),
        ("Total energy cost",   f"€ {kpis['total_cost_eur']:.2f}"),
        ("Grid CO₂ emissions",  f"{kpis['total_co2_kg']:.2f} kg"),
        ("Price  mean/peak",
         f"{kpis['price_mean_eur_kwh']:.3f} / {kpis['price_peak_eur_kwh']:.3f} €/kWh"),
        ("Grid import energy",  f"{kpis['grid_import_kwh']:.1f} kWh"),
        ("Grid availability",   f"{kpis['grid_availability_pct']:.1f} %"
                                f"  ({kpis['n_grid_outage_steps']} outage steps)"),
        ("Wind / PV energy",    f"{kpis['wind_kwh']:.0f} / {kpis['pv_kwh']:.0f} kWh"),
        ("EL / FC energy",      f"{kpis['el_kwh']:.0f} / {kpis['fc_kwh']:.0f} kWh"),
        ("Load energy served",  f"{kpis['load_kwh']:.0f} kWh"),
        ("System health",       f"{kpis['health_mean']:.3f}"),
    ]
    for k, v in rows:
        print(f"  {k:<28}  {v}")
    print()


# ============================================================================
# Mode: train_rl
# ============================================================================

def mode_train_rl(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  HyTwin — RL Training  (PPO on AdvancedH2GridEnv)")
    print("=" * 70)

    try:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("[ERROR] stable-baselines3 and sb3-contrib not installed. Run:")
        print("        pip install stable-baselines3 sb3-contrib")
        return

    from hytwin.rl.advanced_environment import AdvancedH2GridEnv

    scenario = Scenario.from_yaml(args.config)
    grid_config = scenario.grid_config
    weather_params = scenario.weather_params
    cost_params = grid_config.get("energy_cost", {})

    env_kwargs = dict(
        grid_config=grid_config,
        weather_params=weather_params,
        cost_params=cost_params,
        dt_seconds=float(args.dt),
        episode_length=args.steps,
        history_window=8,
        forecast_horizon=3,
    )

    print(f"\n  Config   : {args.config}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Episode  : {args.steps} steps × {args.dt}s = {args.steps*args.dt/3600:.1f} h")
    print(f"  Save dir : {MODEL_DIR}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Training environment (vectorised for speed)
    def make_env():
        e = AdvancedH2GridEnv(**env_kwargs)
        return Monitor(e)

    train_env = make_vec_env(make_env, n_envs=4, seed=args.seed)

    # Evaluation environment
    eval_env = Monitor(AdvancedH2GridEnv(**env_kwargs))

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR),
            eval_freq=max(args.timesteps // 20, 5000),
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=max(args.timesteps // 10, 10_000),
            save_path=str(MODEL_DIR / "checkpoints"),
            name_prefix="advanced_ppo",
            verbose=0,
        ),
    ]

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(ROOT / "output" / "tb_logs"),
    )

    print("  Training started …  (Ctrl-C to interrupt)\n")
    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n  [interrupted by user]")

    elapsed = time.perf_counter() - t0
    model_path = str(MODEL_DIR / "advanced_ppo_final")
    model.save(model_path)
    print(f"\n  Training complete in {elapsed:.1f} s")
    print(f"  Model saved → {model_path}.zip\n")

    # Quick evaluation
    print("  Evaluating trained agent…")
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    print(f"  Mean episode reward: {mean_r:.2f} ± {std_r:.2f}\n")


# ============================================================================
# Mode: compare
# ============================================================================

def mode_compare(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  HyTwin — Controller Comparison  (Classical vs RL)")
    print("=" * 70)

    scenario_cls = _load_scenario(args.config, args.dt, 0.0)
    cost_model = EnergyCostModel(scenario_cls.grid_config.get("energy_cost", {}))

    classical_ctrl = ClassicalController(
        grid_config=scenario_cls.grid_config,
        cost_model=cost_model,
    )

    # Seed match: both simulations use same weather sequence
    np.random.seed(args.seed)
    results_cls, kpis_cls = _run_simulation(
        scenario_cls, args.steps, args.dt, 0.0,
        label="classical", cost_model=cost_model, controller=classical_ctrl,
    )

    # RL controller (only if model exists)
    results_rl: Optional[List[GridState]] = None
    kpis_rl: Optional[Dict[str, float]] = None
    rl_available = False

    model_path = Path(args.model_path + ".zip")
    fallback_path = MODEL_DIR / "best_model.zip"
    actual_path = model_path if model_path.exists() else (
        fallback_path if fallback_path.exists() else None
    )

    if actual_path is not None:
        try:
            from hytwin.control.rl_controller import RLController
            scenario_rl = _load_scenario(args.config, args.dt, 0.0)
            # Reset random seed for same weather
            np.random.seed(args.seed)
            rl_ctrl = RLController(
                model_path=actual_path,
                grid_config=scenario_rl.grid_config,
                cost_model=EnergyCostModel(
                    scenario_rl.grid_config.get("energy_cost", {})
                ),
                dt_seconds=float(args.dt),
            )
            results_rl, kpis_rl = _run_simulation(
                scenario_rl, args.steps, args.dt, 0.0,
                label="rl", controller=rl_ctrl,
            )
            rl_available = True
        except Exception as e:
            print(f"  [WARN] Could not load RL model: {e}")
            print(f"         Run: python demos/demo_advanced.py --mode train_rl first.\n")
    else:
        print(f"  [INFO] No trained RL model found at {args.model_path}.zip")
        print(f"         Run: python demos/demo_advanced.py --mode train_rl first.\n")
        print(f"         Showing Classical-only results.\n")

    # ── Print comparison table ────────────────────────────────────────────
    _print_step_trace(results_cls, args.dt, max_rows=12)
    _print_comparison_table(kpis_cls, kpis_rl if rl_available else None)

    if args.plot:
        plots = {"Classical": results_cls}
        if results_rl:
            plots["RL"] = results_rl
        _plot_results(plots, args.dt)


def _print_comparison_table(
    kpis_cls: Dict[str, float],
    kpis_rl: Optional[Dict[str, float]] = None,
) -> None:
    print(f"\n  ── Controller Comparison ────────────────────────────────────────────")
    metrics = [
        ("Renewable fraction %",  "renewable_fraction_pct",  "{:.1f}", True),
        ("Self-sufficiency %",    "self_sufficiency_pct",    "{:.1f}", True),
        ("H₂ SOC mean",           "h2_soc_mean",             "{:.3f}", None),
        ("H₂ SOC min",            "h2_soc_min",              "{:.3f}", True),
        ("Total cost €",          "total_cost_eur",          "{:.2f}", False),
        ("Grid CO₂ kg",           "total_co2_kg",            "{:.2f}", False),
        ("Grid import kWh",       "grid_import_kwh",         "{:.1f}", False),
        ("Grid availability %",   "grid_availability_pct",   "{:.1f}", True),
        ("System health",         "health_mean",             "{:.3f}", True),
    ]
    col_w = 28
    if kpis_rl is not None:
        print(f"  {'Metric':<{col_w}}  {'Classical':>12}  {'RL':>12}  {'Δ':>8}")
        print("  " + "─" * 64)
        for name, key, fmt, higher_better in metrics:
            v_cls = kpis_cls.get(key, 0.0)
            v_rl  = kpis_rl.get(key, 0.0)
            delta = v_rl - v_cls
            if higher_better is True:
                marker = " ▲" if delta > 0.01 * abs(v_cls + 1e-9) else (
                    " ▼" if delta < -0.01 * abs(v_cls + 1e-9) else "  ")
            elif higher_better is False:
                marker = " ▲" if delta < -0.01 * abs(v_cls + 1e-9) else (
                    " ▼" if delta > 0.01 * abs(v_cls + 1e-9) else "  ")
            else:
                marker = ""
            print(
                f"  {name:<{col_w}}  "
                f"{fmt.format(v_cls):>12}  "
                f"{fmt.format(v_rl):>12}  "
                f"{delta:>+8.2f}{marker}"
            )
    else:
        print(f"  {'Metric':<{col_w}}  {'Classical':>12}")
        print("  " + "─" * 44)
        for name, key, fmt, _ in metrics:
            v = kpis_cls.get(key, 0.0)
            print(f"  {name:<{col_w}}  {fmt.format(v):>12}")
    print()


# ============================================================================
# Mode: dashboard
# ============================================================================

def mode_dashboard(args: argparse.Namespace) -> None:
    try:
        from dashboard.network_app import run_network_dashboard, DEFAULT_CONFIG, DEFAULT_RL_MODEL
        run_network_dashboard(
            config_path=args.config or DEFAULT_CONFIG,
            dt_seconds=args.dt,
            speed_factor=args.speed_factor,
            port=args.port,
            seed=args.seed,
            rl_model_path=DEFAULT_RL_MODEL,
        )
    except ImportError as e:
        print(f"[ERROR] Dashboard dependencies missing: {e}")
        print("  Install with:  pip install fastapi uvicorn websockets")


# ============================================================================
# Optional: matplotlib plots
# ============================================================================

def _plot_results(
    runs: Dict[str, List[GridState]], dt_seconds: int
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots")
        return

    n_runs = len(runs)
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("HyTwin — Advanced Simulation Results", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(4, n_runs, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"Classical": "#2563eb", "RL": "#e11d48", "sim": "#16a34a"}

    for col, (label, results) in enumerate(runs.items()):
        times = [gs_.timestamp for gs_ in results]
        color = colors.get(label, "#6b7280")

        # ── Row 0: Power flows ────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.stackplot(
            times,
            [gs_.wind_power_kw for gs_ in results],
            [gs_.pv_power_kw for gs_ in results],
            [gs_.fuel_cell_power_kw for gs_ in results],
            labels=["Wind", "PV", "Fuel Cell"],
            colors=["#3b82f6", "#f59e0b", "#10b981"],
            alpha=0.7,
        )
        ax0.plot(times, [gs_.load_kw for gs_ in results],
                 "r--", linewidth=1.5, label="Load")
        ax0.plot(times, [gs_.electrolyzer_power_kw for gs_ in results],
                 color="#7c3aed", linewidth=1, label="Electrolyzer", alpha=0.8)
        ax0.set_title(f"{label}: Power Flows", fontsize=10)
        ax0.set_ylabel("kW")
        ax0.legend(loc="upper right", fontsize=7, ncol=2)
        ax0.tick_params(labelbottom=False)

        # ── Row 1: Grid connection + price ────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        gc_kw = [gs_.grid_connection_kw for gs_ in results]
        ax1.fill_between(times, gc_kw, 0, where=[v > 0 for v in gc_kw],
                         color="#ef4444", alpha=0.6, label="Grid import")
        ax1.fill_between(times, gc_kw, 0, where=[v < 0 for v in gc_kw],
                         color="#22c55e", alpha=0.6, label="Grid export")
        ax1.axhline(0, color="gray", linewidth=0.5)
        ax1b = ax1.twinx()
        ax1b.plot(times, [gs_.energy_price_eur_kwh for gs_ in results],
                  color="#f97316", linewidth=1.5, label="Price €/kWh")
        ax1b.set_ylabel("€/kWh", fontsize=8)
        ax1.set_title("Grid Connection & Price", fontsize=10)
        ax1.set_ylabel("kW")
        ax1.legend(loc="upper left", fontsize=7)
        ax1.tick_params(labelbottom=False)

        # ── Row 2: H₂ state ───────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        ax2.plot(times, [gs_.h2_soc for gs_ in results],
                 color="#0ea5e9", linewidth=2, label="H₂ SOC")
        ax2.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")
        ax2.set_ylim(0, 1)
        ax2b = ax2.twinx()
        ax2b.plot(times, [gs_.h2_pressure_bar for gs_ in results],
                  color="#6366f1", linewidth=1, alpha=0.7, label="Pressure bar")
        ax2b.set_ylabel("bar", fontsize=8)
        ax2.set_title("H₂ Storage", fontsize=10)
        ax2.set_ylabel("SOC")
        ax2.legend(loc="upper left", fontsize=7)
        ax2.tick_params(labelbottom=False)

        # ── Row 3: KPIs ───────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[3, col])
        ax3.plot(times, [gs_.renewable_fraction * 100 for gs_ in results],
                 color="#16a34a", linewidth=1.5, label="Renew. fraction %")
        ax3.plot(times, [gs_.grid_self_sufficiency * 100 for gs_ in results],
                 color="#6b7280", linewidth=1, linestyle="--", label="Self-sufficiency %")
        ax3.set_ylim(0, 105)
        ax3.set_title("KPIs", fontsize=10)
        ax3.set_ylabel("%")
        ax3.legend(loc="lower right", fontsize=7)
        for tick in ax3.get_xticklabels():
            tick.set_rotation(30)
            tick.set_fontsize(7)

        # Mark outage steps in all subplots
        for ax in (ax0, ax1, ax2, ax3):
            for gs_ in results:
                if not gs_.grid_available:
                    ax.axvspan(
                        gs_.timestamp - timedelta(seconds=dt_seconds),
                        gs_.timestamp, color="#ef4444", alpha=0.08,
                    )

    plt.savefig(ROOT / "output" / "advanced_results.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved → output/advanced_results.png")
    plt.show()


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    (ROOT / "output").mkdir(exist_ok=True)

    if args.mode == "simulate":
        mode_simulate(args)
    elif args.mode == "train_rl":
        mode_train_rl(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "dashboard":
        mode_dashboard(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
