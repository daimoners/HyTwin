#!/usr/bin/env python3
"""
Run multi-scenario control comparisons (none / rule-based / RL) on HyTwin 2.0.

Outputs (for each horizon):
- CSV per scenario with full time series + objective terms
- Combined CSV for quick external analysis
- JSON summaries (final KPIs and objective values)
- PNG figures with aligned scales and units across scenarios,
  including dedicated H2/fuel-cell diagnostics

This script does not modify simulator modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hytwin.control.classical_controller import ClassicalController
from hytwin.control.fixed_policy_controller import FixedPolicyController
from hytwin.control.rl_controller import RLController
from hytwin.data.time_series import TimeSeriesRecorder
from hytwin.models.energy_cost import EnergyCostModel
from hytwin.simulation.engine import SimulationEngine
from hytwin.simulation.scenario import Scenario


@dataclass
class ObjectiveWeights:
    w_cost: float = 2.0
    w_curtailment: float = 0.2
    w_unmet_demand: float = 1.5
    w_co2: float = 0.5
    w_soc_deviation: float = 0.4
    w_anomaly: float = 1.0
    w_renewable_fraction: float = 1.0
    w_self_sufficiency: float = 1.2
    w_el_cost: float = 0.55
    w_h2_overproduction: float = 1.1
    w_efficiency: float = 0.9
    ref_cost_eur_step: float = 0.5
    ref_co2_kg_step: float = 5.0
    el_usage_norm_kw: float = 500.0
    h2_overproduction_soc_threshold: float = 0.70
    h2_overproduction_soc_span: float = 0.20


SCENARIOS = ("none", "rule", "rl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run control scenario comparisons and export data/plots."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "advanced_stress.yaml"),
        help="Scenario YAML path",
    )
    parser.add_argument(
        "--horizons",
        default="24h,7d",
        help="Comma-separated horizons, e.g. 24h,7d,72h",
    )
    parser.add_argument("--dt", type=int, default=600, help="Step size in seconds")
    parser.add_argument("--speed-factor", type=float, default=0.0, help="0 = as fast as possible")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rl-model",
        default=str(ROOT / "output" / "rl_models" / "advanced_ppo_final.zip"),
        help="Path to trained RL model .zip",
    )
    parser.add_argument(
        "--rl-history-window",
        type=int,
        default=8,
        help="RL inference history window (must match training)",
    )
    parser.add_argument(
        "--rl-forecast-horizon",
        type=int,
        default=3,
        help="RL inference forecast horizon (must match training)",
    )
    parser.add_argument(
        "--rl-forecast-step-mult",
        type=int,
        default=1,
        help="RL inference forecast step multiplier (must match training)",
    )
    parser.add_argument(
        "--diagnostic-lookahead-hours",
        type=float,
        default=6.0,
        help="Lookahead horizon [h] used by anticipatory diagnostics",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "output" / "control_comparison"),
        help="Output base directory",
    )
    parser.add_argument(
        "--scenarios",
        default="none,rule,rl",
        help="Comma-separated scenarios among: none,rule,rl",
    )
    parser.add_argument("--no-rl", action="store_true", help="Skip RL scenario")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def parse_horizons(tokens: str, dt_seconds: int) -> List[Tuple[str, int]]:
    horizons: List[Tuple[str, int]] = []
    for raw in [t.strip().lower() for t in tokens.split(",") if t.strip()]:
        if raw.endswith("h"):
            hours = float(raw[:-1])
            steps = int(round((hours * 3600.0) / dt_seconds))
            label = f"{int(hours)}h" if float(hours).is_integer() else f"{hours}h"
        elif raw.endswith("d"):
            days = float(raw[:-1])
            steps = int(round((days * 24 * 3600.0) / dt_seconds))
            label = f"{int(days)}d" if float(days).is_integer() else f"{days}d"
        else:
            raise ValueError(f"Unsupported horizon token: {raw} (use suffix h or d)")
        if steps <= 0:
            raise ValueError(f"Non-positive step count for horizon: {raw}")
        horizons.append((label, steps))
    return horizons


def parse_scenarios(tokens: str, no_rl: bool) -> List[str]:
    valid = set(SCENARIOS)
    scenarios = [t.strip().lower() for t in tokens.split(",") if t.strip()]
    if not scenarios:
        raise ValueError("No scenarios specified")
    bad = [s for s in scenarios if s not in valid]
    if bad:
        raise ValueError(f"Unsupported scenario(s): {bad}. Use only: {', '.join(SCENARIOS)}")
    if no_rl:
        scenarios = [s for s in scenarios if s != "rl"]
    if not scenarios:
        raise ValueError("No scenarios left after applying --no-rl")
    return scenarios


def resolve_rl_model(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if path.suffix != ".zip":
        with_zip = path.with_suffix(".zip")
        if with_zip.exists():
            return with_zip
    raise FileNotFoundError(
        f"RL model not found at {path} (or .zip variant). "
        f"Run train_rl_controller.py first."
    )


def build_controller(
    mode: str,
    scenario: Scenario,
    rl_model_path: Path | None,
    cost_model: Optional[EnergyCostModel] = None,
    rl_history_window: int = 8,
    rl_forecast_horizon: int = 3,
    rl_forecast_step_mult: int = 1,
):
    """Build controller with shared cost_model to ensure RNG synchronization."""
    if cost_model is None:
        cost_model = EnergyCostModel(scenario.grid_config.get("energy_cost", {}))
    if mode == "none":
        return FixedPolicyController(grid_config=scenario.grid_config)
    if mode == "rule":
        return ClassicalController(grid_config=scenario.grid_config, cost_model=cost_model)
    if mode == "rl":
        if rl_model_path is None:
            raise ValueError("rl_model_path is required for mode=rl")
        return RLController(
            model_path=rl_model_path,
            grid_config=scenario.grid_config,
            cost_model=cost_model,
            dt_seconds=float(scenario.dt_seconds),
            history_window=int(rl_history_window),
            forecast_horizon=int(rl_forecast_horizon),
            forecast_step_multiplier=int(rl_forecast_step_mult),
        )
    raise ValueError(f"Unknown controller mode: {mode}")


def run_one(
    mode: str,
    config_path: str,
    steps: int,
    dt: int,
    speed_factor: float,
    seed: int,
    rl_model_path: Path | None,
    rl_history_window: int = 8,
    rl_forecast_horizon: int = 3,
    rl_forecast_step_mult: int = 1,
) -> pd.DataFrame:
    """Run a single scenario with complete RNG isolation.
    
    Ensures that the environment (weather, load, prices) is identical across
    different controller modes when using the same seed. This is critical for
    fair comparison of control strategies.
    """
    # ─────────────────────────────────────────────────────────────────────
    # CRITICAL: Seed ALL random number generators to ensure reproducibility
    # ─────────────────────────────────────────────────────────────────────
    np.random.seed(seed)
    
    # Seed PyTorch (used by Stable-Baselines3 for neural networks)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not installed
    
    # Seed TensorFlow (alternative RL backend)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Also set numpy seed used by TensorFlow
        np.random.seed(seed)
    except ImportError:
        pass  # TensorFlow not installed
    
    # CRITICAL: Force lazy initialization of EnergyCostModel BEFORE controllers
    # This ensures RNG consumption is consistent across all scenario runs.
    # If not done, first controller call triggers _daily_factor resampling,
    # consuming RNG and causing divergence across scenarios.
    scenario = Scenario.from_yaml(config_path)
    scenario.dt_seconds = float(dt)
    
    # Create SHARED cost_model and force lazy initialization immediately after seed
    # This ensures all scenarios consume RNG in IDENTICAL order
    cost_model = EnergyCostModel(scenario.grid_config.get("energy_cost", {}))
    from datetime import datetime as dt_now
    cost_model.get_buy_price(dt_now.now())  # Force _maybe_resample_day()
    cost_model.reset()  # Reset to clean state

    controller = build_controller(
        mode,
        scenario,
        rl_model_path,
        cost_model=cost_model,
        rl_history_window=rl_history_window,
        rl_forecast_horizon=rl_forecast_horizon,
        rl_forecast_step_mult=rl_forecast_step_mult,
    )
    if hasattr(controller, "reset"):
        controller.reset()

    rl_diag_records: List[Dict[str, float | str]] = []

    def _schedule(step, ts, gs):
        if gs is None:
            return {}
        actions = controller.compute_actions(gs, timestamp=ts)
        if mode == "rl" and hasattr(controller, "get_last_diagnostics"):
            diag = controller.get_last_diagnostics() or {}
            if diag:
                row = {k: float(v) for k, v in diag.items()}
                row["timestamp"] = ts.isoformat()
                rl_diag_records.append(row)
        return actions

    scenario.set_schedule(_schedule)

    recorder = TimeSeriesRecorder()
    engine = SimulationEngine(
        scenario=scenario,
        dt_seconds=float(dt),
        speed_factor=float(speed_factor),
        recorder=recorder,
        cost_model=cost_model,
    )
    engine.run(steps=steps)

    df = recorder.to_dataframe().copy()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if mode == "rl" and rl_diag_records and not df.empty:
        diag_df = pd.DataFrame(rl_diag_records)
        diag_df["timestamp"] = pd.to_datetime(diag_df["timestamp"], errors="coerce")
        diag_df = diag_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df = df.merge(diag_df, on="timestamp", how="left")

    df["scenario"] = mode
    return df


def _verify_weather_reproducibility(
    config_path: str, dt: int, seed: int, steps_test: int = 50
) -> bool:
    """Verify that two runs with identical seed produce identical weather.
    
    This is a critical sanity check: if weather (wind, PV, load, price) differs
    between runs, any controller comparison is invalid.
    
    Returns:
        True if reproducible, False otherwise
    """
    try:
        print("\n[RNG TEST] Verifying environmental reproducibility...")
        
        # Run 1
        df1 = run_one(
            mode="none",
            config_path=config_path,
            steps=steps_test,
            dt=dt,
            speed_factor=0.0,
            seed=seed,
            rl_model_path=None,
            rl_history_window=8,
            rl_forecast_horizon=3,
            rl_forecast_step_mult=1,
        )
        
        # Run 2 (identical seed)
        df2 = run_one(
            mode="none",
            config_path=config_path,
            steps=steps_test,
            dt=dt,
            speed_factor=0.0,
            seed=seed,
            rl_model_path=None,
            rl_history_window=8,
            rl_forecast_horizon=3,
            rl_forecast_step_mult=1,
        )
        
        # Check environmental variables (should be identical)
        env_vars = ["wind_power_kw", "pv_power_kw", "load_kw", "energy_price_eur_kwh"]
        all_match = True
        
        for var in env_vars:
            if var in df1.columns and var in df2.columns:
                v1 = df1[var].values
                v2 = df2[var].values
                match = np.allclose(v1, v2, atol=1e-10, rtol=1e-10)
                status = "✓" if match else "✗"
                max_diff = np.max(np.abs(v1 - v2))
                print(f"  {status} {var:25s} max_diff={max_diff:.2e}")
                all_match = all_match and match
        
        if all_match:
            print("\n[RNG TEST] ✓ PASS: Environment is reproducible")
            print("          Fair comparison of controllers is possible.\n")
            return True
        else:
            print("\n[RNG TEST] ✗ FAIL: Environment differs across runs!")
            print("          RNG state not properly isolated. See implementation.\n")
            return False
            
    except Exception as exc:
        print(f"[RNG TEST] Error during reproducibility check: {exc}")
        return False


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    xv = x[valid].astype(float)
    yv = y[valid].astype(float)
    if float(xv.std()) < 1e-12 or float(yv.std()) < 1e-12:
        return float("nan")
    return float(xv.corr(yv))


def add_anticipatory_diagnostics(
    df: pd.DataFrame,
    dt_seconds: int,
    lookahead_hours: float,
) -> pd.DataFrame:
    out = df.copy()
    n_ahead = max(1, int(round((lookahead_hours * 3600.0) / max(1, dt_seconds))))

    out["future_price_lookahead"] = out["energy_price_eur_kwh"].shift(-n_ahead)
    out["future_load_lookahead_kw"] = out["load_kw"].shift(-n_ahead)
    out["future_renewable_lookahead_kw"] = (
        out["wind_power_kw"].shift(-n_ahead) + out["pv_power_kw"].shift(-n_ahead)
    )
    out["future_renew_surplus_lookahead_kw"] = (
        out["future_renewable_lookahead_kw"] - out["future_load_lookahead_kw"]
    )

    if "rl_used_f_price_norm_t1" in out.columns:
        out["used_future_price_signal"] = (out["rl_used_f_price_norm_t1"] + 1.0) * 0.5
    else:
        out["used_future_price_signal"] = np.clip(out["future_price_lookahead"] / 0.50, 0.0, 1.0)

    if "rl_used_f_renew_surplus_t1" in out.columns:
        out["used_future_renew_surplus_signal"] = out["rl_used_f_renew_surplus_t1"]
    else:
        out["used_future_renew_surplus_signal"] = np.clip(
            out["future_renew_surplus_lookahead_kw"] / (out["load_kw"] + 1e-9),
            -1.0,
            1.0,
        )

    out["h2_soc_delta_next"] = out["h2_soc"].shift(-1) - out["h2_soc"]
    out["el_power_norm"] = np.clip(out["electrolyzer_power_kw"] / 500.0, 0.0, 2.0)
    out["grid_import_norm"] = np.clip(np.maximum(0.0, out["real_grid_import_kw"]) / 500.0, 0.0, 2.0)

    out["diag_el_price_anticipation"] = np.clip(
        out["el_power_norm"] * (1.0 - out["used_future_price_signal"]),
        0.0,
        1.0,
    )
    out["diag_grid_price_anticipation"] = np.clip(
        out["grid_import_norm"] * (1.0 - out["used_future_price_signal"]),
        0.0,
        1.0,
    )
    out["diag_soc_renew_schedule"] = np.clip(
        (0.5 + 0.5 * out["used_future_renew_surplus_signal"]) * (0.5 + 5.0 * out["h2_soc_delta_next"]),
        0.0,
        1.0,
    )
    return out


def add_objective_terms(df: pd.DataFrame, w: ObjectiveWeights) -> pd.DataFrame:
    required = [
        "wind_power_kw", "pv_power_kw", "fuel_cell_power_kw", "load_kw",
        "grid_connection_kw", "grid_exchange_kw", "step_cost_eur", "step_co2_kg",
        "h2_soc", "overall_health", "renewable_fraction", "grid_self_sufficiency",
        "electrolyzer_power_kw",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for objective calculation: {missing}")

    out = df.copy()

    renewable_kw = out["wind_power_kw"] + out["pv_power_kw"]
    used_renewable_kw = np.maximum(0.0, renewable_kw + out["grid_exchange_kw"])
    out["curtailment_ratio"] = np.maximum(0.0, renewable_kw - used_renewable_kw) / (renewable_kw + 1e-9)

    out["real_grid_import_kw"] = np.maximum(0.0, out["grid_exchange_kw"])
    supply_to_load_kw = (
        out["wind_power_kw"]
        + out["pv_power_kw"]
        + out["fuel_cell_power_kw"]
        + out["real_grid_import_kw"]
    )
    out["real_supply_kw"] = supply_to_load_kw
    out["unmet_demand_ratio"] = np.maximum(0.0, out["load_kw"] - supply_to_load_kw) / (out["load_kw"] + 1e-9)

    out["cost_norm"] = np.clip(out["step_cost_eur"] / (w.ref_cost_eur_step + 1e-9), 0.0, 2.5)
    out["co2_norm"] = np.clip(out["step_co2_kg"] / (w.ref_co2_kg_step + 1e-9), 0.0, 1.0)
    out["soc_deviation"] = (out["h2_soc"] - 0.50).abs() ** 2
    out["anomaly"] = 1.0 - out["overall_health"]
    out["el_cost_norm"] = np.clip(out["electrolyzer_power_kw"] / (w.el_usage_norm_kw + 1e-9), 0.0, 2.0)
    out["overproduction_soc_ratio"] = np.clip(
        (out["h2_soc"] - w.h2_overproduction_soc_threshold) / max(1e-9, w.h2_overproduction_soc_span),
        0.0,
        1.0,
    )
    out["overproduction_penalty_norm"] = out["el_cost_norm"] * out["overproduction_soc_ratio"]
    useful_supply_kw = np.minimum(out["load_kw"], out["real_supply_kw"])
    total_energy_used_kw = (
        out["wind_power_kw"]
        + out["pv_power_kw"]
        + out["real_grid_import_kw"]
        + np.maximum(0.0, out["fuel_cell_power_kw"])
        + np.maximum(0.0, out["electrolyzer_power_kw"])
    )
    out["system_efficiency"] = np.clip(useful_supply_kw / (total_energy_used_kw + 1e-9), 0.0, 1.0)

    out["j_cost"] = w.w_cost * out["cost_norm"]
    out["j_curtailment"] = w.w_curtailment * out["curtailment_ratio"]
    out["j_unmet_demand"] = w.w_unmet_demand * out["unmet_demand_ratio"]
    out["j_co2"] = w.w_co2 * out["co2_norm"]
    out["j_soc_deviation"] = w.w_soc_deviation * out["soc_deviation"]
    out["j_anomaly"] = w.w_anomaly * out["anomaly"]
    out["j_renewable_fraction"] = -w.w_renewable_fraction * out["renewable_fraction"]
    out["j_self_sufficiency"] = -w.w_self_sufficiency * out["grid_self_sufficiency"]
    out["j_el_cost"] = w.w_el_cost * out["el_cost_norm"]
    out["j_h2_overproduction"] = w.w_h2_overproduction * out["overproduction_penalty_norm"]
    out["j_efficiency"] = -w.w_efficiency * out["system_efficiency"]

    out["j_total_step"] = (
        out["j_cost"]
        + out["j_curtailment"]
        + out["j_unmet_demand"]
        + out["j_co2"]
        + out["j_soc_deviation"]
        + out["j_anomaly"]
        + out["j_renewable_fraction"]
        + out["j_self_sufficiency"]
        + out["j_el_cost"]
        + out["j_h2_overproduction"]
        + out["j_efficiency"]
    )
    out["j_total_cumulative"] = out["j_total_step"].cumsum()

    return out


def summary_row(df: pd.DataFrame) -> Dict[str, float | str]:
    scenario = str(df["scenario"].iloc[0])
    corr_future_price_el = _safe_corr(
        df.get("future_price_lookahead", pd.Series(dtype=float)),
        df["electrolyzer_power_kw"],
    )
    corr_used_signal_price_el = _safe_corr(
        df.get("used_future_price_signal", pd.Series(dtype=float)),
        df["electrolyzer_power_kw"],
    )
    corr_future_price_grid = _safe_corr(
        df.get("future_price_lookahead", pd.Series(dtype=float)),
        df["real_grid_import_kw"],
    )
    corr_future_renew_soc_delta = _safe_corr(
        df.get("used_future_renew_surplus_signal", pd.Series(dtype=float)),
        df.get("h2_soc_delta_next", pd.Series(dtype=float)),
    )
    return {
        "scenario": scenario,
        "steps": int(len(df)),
        "renewable_fraction_mean": float(df["renewable_fraction"].mean()),
        "self_sufficiency_mean": float(df["grid_self_sufficiency"].mean()),
        "h2_soc_mean": float(df["h2_soc"].mean()),
        "h2_soc_min": float(df["h2_soc"].min()),
        "h2_soc_max": float(df["h2_soc"].max()),
        "cost_total_eur": float(df["step_cost_eur"].sum()),
        "co2_total_kg": float(df["step_co2_kg"].sum()),
        "unmet_mean": float(df["unmet_demand_ratio"].mean()),
        "grid_import_kwh": float(np.maximum(0.0, df["real_grid_import_kw"]).sum()),
        "h2_soc_variance": float(df["h2_soc"].var()),
        "system_efficiency_mean": float(df["system_efficiency"].mean()),
        "diag_el_price_anticipation_mean": float(df.get("diag_el_price_anticipation", pd.Series([np.nan])).mean()),
        "diag_grid_price_anticipation_mean": float(df.get("diag_grid_price_anticipation", pd.Series([np.nan])).mean()),
        "diag_soc_renew_schedule_mean": float(df.get("diag_soc_renew_schedule", pd.Series([np.nan])).mean()),
        "corr_future_price_el": corr_future_price_el,
        "corr_used_signal_price_el": corr_used_signal_price_el,
        "corr_future_price_grid_import": corr_future_price_grid,
        "corr_future_renew_surplus_soc_delta": corr_future_renew_soc_delta,
        "j_total_final": float(df["j_total_cumulative"].iloc[-1]),
    }


def _nice_name(s: str) -> str:
    return {"none": "Fixed baseline", "rule": "Rule-based", "rl": "RL"}.get(s, s)


def _plot_overlay(ax, series_by_scenario: Dict[str, pd.Series], ylabel: str, title: str) -> None:
    colors = {"none": "#6b7280", "rule": "#2563eb", "rl": "#dc2626"}
    all_min = min(float(v.min()) for v in series_by_scenario.values())
    all_max = max(float(v.max()) for v in series_by_scenario.values())
    if all_min == all_max:
        pad = 1.0 if all_min == 0 else abs(all_min) * 0.1
        all_min -= pad
        all_max += pad
    else:
        pad = (all_max - all_min) * 0.05
        all_min -= pad
        all_max += pad

    for scenario, series in series_by_scenario.items():
        ax.plot(series.index, series.values, label=_nice_name(scenario), lw=1.5, color=colors.get(scenario))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(all_min, all_max)
    ax.grid(alpha=0.25)


def _plot_grid(
    runs: Dict[str, pd.DataFrame],
    vars_meta: List[Tuple[str, str, str]],
    fig_title: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = int(math.ceil(len(vars_meta) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (column, unit, title) in zip(axes, vars_meta):
        series_by_scenario = {
            sc: df.set_index("timestamp")[column]
            for sc, df in runs.items()
            if column in df.columns
        }
        if not series_by_scenario:
            ax.set_title(f"{title} (not available)")
            ax.grid(alpha=0.25)
            continue
        _plot_overlay(ax, series_by_scenario, unit, title)

    for ax in axes[len(vars_meta):]:
        ax.axis("off")

    if len(axes) > 0:
        axes[0].legend(loc="upper right")
    fig.suptitle(fig_title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_core(runs: Dict[str, pd.DataFrame], output_path: Path, dpi: int) -> None:
    vars_meta = [
        ("wind_power_kw", "kW", "Wind power"),
        ("pv_power_kw", "kW", "PV power"),
        ("electrolyzer_power_kw", "kW", "Electrolyzer power"),
        ("fuel_cell_power_kw", "kW", "Fuel cell power"),
        ("unmet_demand_ratio", "ratio", "Unmet demand ratio"),
        ("h2_storage_kg", "kg", "H₂ tank level"),
        ("h2_soc", "ratio", "H₂ SOC"),
        ("h2_production_kg_h", "kg/h", "H₂ production flow"),
        ("h2_consumption_kg_h", "kg/h", "H₂ consumption flow"),
        ("load_kw", "kW", "Demand"),
        ("real_grid_import_kw", "kW", "Grid import (real)"),
        ("curtailment_ratio", "ratio", "Curtailment"),
        ("step_cost_eur", "€", "Instantaneous cost"),
        ("step_co2_kg", "kg", "Instantaneous CO2"),
        ("cumulative_cost_eur", "€", "Cumulative cost"),
    ]

    _plot_grid(
        runs=runs,
        vars_meta=vars_meta,
        fig_title="HyTwin control comparison — Core variables",
        output_path=output_path,
        dpi=dpi,
    )


def plot_h2_diagnostics(
    runs: Dict[str, pd.DataFrame],
    output_path: Path,
    dpi: int,
    dt_seconds: int,
) -> None:
    dt_h = float(dt_seconds) / 3600.0
    diag_runs: Dict[str, pd.DataFrame] = {}
    for scenario, df in runs.items():
        work = df.copy()
        work["h2_net_kg_step"] = (
            (work["h2_production_kg_h"] - work["h2_consumption_kg_h"]) * dt_h
        )
        work["h2_storage_delta_kg_step"] = work["h2_storage_kg"].diff().fillna(0.0)
        diag_runs[scenario] = work

    vars_meta = [
        ("electrolyzer_power_kw", "kW", "Electrolyzer power"),
        ("fuel_cell_power_kw", "kW", "Fuel cell power"),
        ("h2_production_kg_h", "kg/h", "H₂ production flow"),
        ("h2_consumption_kg_h", "kg/h", "H₂ consumption flow"),
        ("h2_storage_kg", "kg", "H₂ tank level"),
        ("h2_soc", "ratio", "H₂ SOC"),
        ("h2_net_kg_step", "kg/step", "Net H₂ balance (prod-cons)"),
        ("h2_storage_delta_kg_step", "kg/step", "Δ H₂ tank level"),
    ]

    _plot_grid(
        runs=diag_runs,
        vars_meta=vars_meta,
        fig_title="HyTwin control comparison — H2 & fuel-cell diagnostics",
        output_path=output_path,
        dpi=dpi,
    )


def plot_objective(runs: Dict[str, pd.DataFrame], output_path: Path, dpi: int) -> None:
    vars_meta = [
        ("j_cost", "weighted", "J term: cost"),
        ("j_curtailment", "weighted", "J term: curtailment"),
        ("j_unmet_demand", "weighted", "J term: unmet demand"),
        ("j_co2", "weighted", "J term: CO₂"),
        ("j_soc_deviation", "weighted", "J term: SOC deviation"),
        ("j_anomaly", "weighted", "J term: anomaly"),
        ("j_renewable_fraction", "weighted", "J term: renewable fraction"),
        ("j_self_sufficiency", "weighted", "J term: self-sufficiency"),
        ("j_el_cost", "weighted", "J term: EL energy cost"),
        ("j_h2_overproduction", "weighted", "J term: H2 overproduction"),
        ("j_efficiency", "weighted", "J term: system efficiency"),
        ("j_total_cumulative", "weighted", "J total cumulative"),
    ]

    _plot_grid(
        runs=runs,
        vars_meta=vars_meta,
        fig_title="HyTwin control comparison — Objective J terms",
        output_path=output_path,
        dpi=dpi,
    )


def plot_anticipatory_diagnostics(runs: Dict[str, pd.DataFrame], output_path: Path, dpi: int) -> None:
    vars_meta = [
        ("used_future_price_signal", "norm", "Used future-price signal"),
        ("used_future_renew_surplus_signal", "norm", "Used future renewable-surplus signal"),
        ("diag_el_price_anticipation", "score", "EL vs future-price anticipation"),
        ("diag_grid_price_anticipation", "score", "Grid import vs future-price anticipation"),
        ("diag_soc_renew_schedule", "score", "SOC scheduling vs future renewables"),
        ("h2_soc_delta_next", "ΔSOC", "Next-step H2 SOC delta"),
    ]
    _plot_grid(
        runs=runs,
        vars_meta=vars_meta,
        fig_title="HyTwin control comparison — Anticipatory diagnostics",
        output_path=output_path,
        dpi=dpi,
    )


def run_horizon(
    horizon_label: str,
    steps: int,
    args: argparse.Namespace,
    weights: ObjectiveWeights,
    selected: List[str],
) -> None:
    outdir = Path(args.outdir) / horizon_label
    data_dir = outdir / "data"
    fig_dir = outdir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rl_path = None
    if "rl" in selected:
        rl_path = resolve_rl_model(args.rl_model)

    runs: Dict[str, pd.DataFrame] = {}
    for scenario_name in selected:
        print(f"[run] horizon={horizon_label} scenario={scenario_name} steps={steps}")
        df = run_one(
            mode=scenario_name,
            config_path=args.config,
            steps=steps,
            dt=args.dt,
            speed_factor=args.speed_factor,
            seed=args.seed,
            rl_model_path=rl_path,
            rl_history_window=args.rl_history_window,
            rl_forecast_horizon=args.rl_forecast_horizon,
            rl_forecast_step_mult=args.rl_forecast_step_mult,
        )
        df = add_objective_terms(df, weights)
        df = add_anticipatory_diagnostics(
            df,
            dt_seconds=int(args.dt),
            lookahead_hours=float(args.diagnostic_lookahead_hours),
        )
        runs[scenario_name] = df

        csv_path = data_dir / f"timeseries_{scenario_name}.csv"
        df.to_csv(csv_path, index=False)

    combined = pd.concat(runs.values(), axis=0, ignore_index=True)
    combined.to_csv(data_dir / "timeseries_all_scenarios.csv", index=False)

    summary = pd.DataFrame([summary_row(df) for df in runs.values()])
    summary.to_csv(data_dir / "summary.csv", index=False)

    meta = {
        "config": args.config,
        "horizon": horizon_label,
        "steps": steps,
        "dt_seconds": args.dt,
        "speed_factor": args.speed_factor,
        "seed": args.seed,
        "scenarios": selected,
        "objective_weights": asdict(weights),
        "rl_history_window": int(args.rl_history_window),
        "rl_forecast_horizon": int(args.rl_forecast_horizon),
        "rl_forecast_step_mult": int(args.rl_forecast_step_mult),
        "diagnostic_lookahead_hours": float(args.diagnostic_lookahead_hours),
    }
    with open(data_dir / "run_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    plot_core(runs, fig_dir / "core_timeseries.png", dpi=args.dpi)
    plot_h2_diagnostics(
        runs,
        fig_dir / "h2_fuelcell_diagnostics.png",
        dpi=args.dpi,
        dt_seconds=args.dt,
    )
    plot_objective(runs, fig_dir / "objective_terms_timeseries.png", dpi=args.dpi)
    plot_anticipatory_diagnostics(runs, fig_dir / "anticipatory_diagnostics.png", dpi=args.dpi)

    print(f"[saved] {outdir}")


def main() -> None:
    args = parse_args()
    weights = ObjectiveWeights()

    horizons = parse_horizons(args.horizons, args.dt)
    selected = parse_scenarios(args.scenarios, args.no_rl)
    print("=" * 70)
    print("HyTwin 2.0 — Control scenarios comparison")
    print("=" * 70)
    print(f"config        : {args.config}")
    print(f"horizons      : {', '.join(h for h, _ in horizons)}")
    print(f"dt [s]        : {args.dt}")
    print(f"speed_factor  : {args.speed_factor}")
    print(f"seed          : {args.seed}")
    print(f"outdir        : {args.outdir}")
    print(f"rl_model      : {'SKIPPED' if args.no_rl else args.rl_model}")
    print(f"scenarios     : {', '.join(selected)}")
    print(f"rl_hist_win   : {args.rl_history_window}")
    print(f"rl_frc_hor    : {args.rl_forecast_horizon}")
    print(f"rl_frc_mult   : {args.rl_forecast_step_mult}")
    print(f"diag_lookahead: {args.diagnostic_lookahead_hours}h")
    
    # ────────────────────────────────────────────────────────────────────
    # SANITY CHECK: Verify environmental reproducibility before main run
    # ────────────────────────────────────────────────────────────────────
    reproducible = _verify_weather_reproducibility(
        config_path=args.config,
        dt=args.dt,
        seed=args.seed,
        steps_test=20,  # Short test run
    )
    
    if not reproducible:
        print("[WARNING] Environmental reproducibility test FAILED.")
        print("          Comparison results may be invalid.")
        print("          Consider debugging RNG state issues.\n")
    
    # Run actual scenarios
    for label, steps in horizons:
        run_horizon(label, steps, args, weights, selected)

    print("\nDone. Data and figures exported.")


if __name__ == "__main__":
    main()
