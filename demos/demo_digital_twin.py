"""
demo_digital_twin.py
====================
Shows the HyTwin 2.0 digital-twin layer in action:

  Phase A — Normal operation:  model + noisy sensor → fusion estimate
  Phase B — Sensor fault:      stuck pressure sensor, DT detects anomaly
  Phase C — Multi-component:   GridTwin 50-step run, system-level KPIs
  Phase D — Comparison:        virtual vs "measured" signal quality

Usage
-----
    python demos/demo_digital_twin.py [--steps 100] [--plot] [--save <path>]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.digital_twin.twin_node import TwinNode
from hytwin.digital_twin.grid_twin import GridTwin
from hytwin.sensors.base_sensor import SensorStatus
from hytwin.sensors.sensors import PressureSensor, PowerSensor
from hytwin.models.hydrogen_tank import HydrogenTankModel
from hytwin.models.wind_turbine import WindTurbineModel
from hytwin.weather.weather_model import WeatherModel


# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HyTwin 2.0 — digital-twin demo")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save", default="")
    return p.parse_args()


# ============================================================================
# Phase A — single TwinNode fusion (H2 tank pressure)
# ============================================================================
def _phase_a_fusion(steps: int):
    """Compare model-only vs noisy-sensor vs fused estimate."""
    print("  ── Phase A: Sensor-model fusion on H2 tank TwinNode")

    tank = HydrogenTankModel(
        "tk1",
        {
            "volume_m3": 10.0,
            "max_pressure_bar": 700.0,
            "min_pressure_bar": 10.0,
            "initial_soc": 0.45,
            "temperature_c": 20.0,
            "max_charge_rate_kg_s": 0.08,
            "max_discharge_rate_kg_s": 0.05,
            "boiloff_rate_per_day": 0.0,
        },
    )
    pressure_sensor = PressureSensor(
        sensor_id="tk1.pressure",
        noise_std_bar=8.0,      # noisy meter
        drift_rate_bar=0.2,
    )
    node = TwinNode("tk1", tank, state_keys=["pressure_bar"])

    model_vals, fused_vals, sensor_vals = [], [], []
    ts_base = datetime(2024, 6, 15, 6, 0, 0)
    rng = np.random.default_rng(0)

    for i in range(steps):
        ts = ts_base + timedelta(seconds=i * 600)
        # Alternate filling / discharging
        inflow  = 0.05 if i < steps // 2 else 0.0
        outflow = 0.0  if i < steps // 2 else 0.03
        context = {"h2_charge_kg": inflow * 600, "h2_discharge_kg": outflow * 600}
        ms = tank.step(dt=600.0, context=context)

        true_pressure = ms.values.get("pressure_bar", 0.0)
        reading = pressure_sensor.measure(true_pressure, ts)
        node.update(ms, [reading])
        fused_state = node.observe(keys=["pressure_bar"])

        model_vals.append(ms.values["pressure_bar"])
        sensor_vals.append(reading.value)
        fused_vals.append(fused_state[0] if len(fused_state) > 0 else float("nan"))

    model_arr  = np.array(model_vals)
    sensor_arr = np.array(sensor_vals, dtype=float)
    fused_arr  = np.array(fused_vals,  dtype=float)

    sensor_rmse = np.sqrt(np.nanmean((sensor_arr - model_arr) ** 2))
    fused_rmse  = np.sqrt(np.nanmean((fused_arr  - model_arr) ** 2))
    print(f"    Sensor RMSE = {sensor_rmse:.2f} bar")
    print(f"    Fused  RMSE = {fused_rmse:.2f} bar  (should be ≤ sensor RMSE)")

    return model_arr, sensor_arr, fused_arr


# ============================================================================
# Phase B — fault detection
# ============================================================================
def _phase_b_fault(steps: int):
    """Inject a stuck fault → TwinNode should detect the anomaly."""
    print("  ── Phase B: Anomaly detection under stuck-sensor fault")

    tank = HydrogenTankModel(
        "tk1_b",
        {
            "volume_m3": 10.0,
            "max_pressure_bar": 700.0,
            "min_pressure_bar": 10.0,
            "initial_soc": 0.50,
            "temperature_c": 20.0,
            "max_charge_rate_kg_s": 0.08,
            "max_discharge_rate_kg_s": 0.05,
            "boiloff_rate_per_day": 0.0,
        },
    )
    sensor = PressureSensor("tk1_b.pressure", noise_std_bar=3.0)
    node = TwinNode("tk1_b", tank, state_keys=["pressure_bar"])

    ts_base = datetime(2024, 6, 15, 0, 0)
    health_scores, anomalies = [], []
    fault_at = steps // 3

    for i in range(steps):
        ts = ts_base + timedelta(seconds=i * 600)
        # Tank drains continuously
        context = {"h2_charge_kg": 0.0, "h2_discharge_kg": 0.03 * 600}
        ms = tank.step(dt=600.0, context=context)

        # Inject stuck-fault halfway through
        if i == fault_at:
            sensor.inject_fault(SensorStatus.FAULT_STUCK, stuck_value=ms.values["pressure_bar"])
        elif i == fault_at + steps // 3:
            sensor.clear_fault()

        reading = sensor.measure(ms.values["pressure_bar"], ts)
        node.update(ms, [reading])
        health_scores.append(node.state.health_score)
        anomalies.append(node.state.anomaly_score)

    health_arr  = np.array(health_scores)
    anomaly_arr = np.array(anomalies)
    fault_window_health = health_arr[fault_at : fault_at + steps // 3].mean()
    normal_health       = health_arr[:fault_at].mean()
    print(f"    Normal  avg health = {normal_health:.3f}")
    print(f"    Fault   avg health = {fault_window_health:.3f}  (faulted sensors are bypassed; model used)")

    return health_arr, anomaly_arr, fault_at


# ============================================================================
# Phase C — GridTwin multi-component run
# ============================================================================
def _phase_c_gridtwin(steps: int):
    """Instantiate GridTwin and run several steps; print KPIs."""
    print("  ── Phase C: GridTwin 50-step run")

    config_path = ROOT / "config" / "default_grid.yaml"
    if not config_path.exists():
        print("    [SKIP] config/default_grid.yaml not found")
        return None

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    grid_twin = GridTwin(cfg["grid"])
    grid_twin.build()

    weather = WeatherModel(
        latitude_deg=40.5,
        longitude_deg=14.8,
        altitude_m=50.0,
        weibull_k=2.0,
        weibull_c=6.5,
    )

    ts = datetime(2024, 6, 21, 8, 0, 0)
    dt = 600.0
    renewable_fracs, self_suff, soc_vals = [], [], []

    for i in range(min(steps, 50)):
        ts_i = ts + timedelta(seconds=i * dt)
        wx   = weather.step(ts_i)
        control = {
            "el1": {"power_setpoint_kw": 50.0},
            "fc1": {"power_setpoint_kw": 0.0, "h2_available_kg": 100.0},
            "load1": {"demand_response": 0.0},
        }
        gs = grid_twin.step(dt=dt, weather=wx, control_actions=control, timestamp=ts_i)
        renewable_fracs.append(gs.renewable_fraction)
        self_suff.append(gs.grid_self_sufficiency)
        soc_vals.append(gs.h2_soc)

    rf_mean = np.mean(renewable_fracs)
    ss_mean = np.mean(self_suff)
    print(f"    Mean renewable fraction   = {rf_mean:.2%}")
    print(f"    Mean self-sufficiency     = {ss_mean:.2%}")
    print(f"    Final H2 SOC              = {soc_vals[-1]:.3f}")

    return np.array(renewable_fracs), np.array(self_suff), np.array(soc_vals)


# ============================================================================
# Plotting
# ============================================================================
def _make_plot(phase_a, phase_b, phase_c, save_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plots = 4
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("HyTwin 2.0 — Digital Twin Demo", fontsize=14, fontweight="bold")

    # Phase A — fusion
    ax = axes[0, 0]
    model_arr, sensor_arr, fused_arr = phase_a
    ax.plot(model_arr, lw=2, label="Model (ground truth)", color="tab:blue")
    ax.plot(sensor_arr, lw=1, alpha=0.7, label="Noisy sensor", color="tab:red")
    ax.plot(fused_arr, lw=2, ls="--", label="Fused estimate", color="tab:green")
    ax.set_title("Phase A — Sensor-Model Fusion (Pressure)")
    ax.set_xlabel("Step"); ax.set_ylabel("bar"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase B health
    ax = axes[0, 1]
    health_arr, anomaly_arr, fault_at = phase_b
    x = np.arange(len(health_arr))
    ax.plot(x, health_arr, lw=2, label="Health score", color="tab:green")
    ax.axvline(fault_at, color="red", ls="--", label=f"Fault injected @ step {fault_at}")
    ax.axvline(fault_at + len(health_arr) // 3, color="orange", ls=":", label="Fault cleared")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase B — Health Score under Sensor Fault")
    ax.set_xlabel("Step"); ax.set_ylabel("Health [0–1]"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase B anomaly
    ax = axes[1, 0]
    ax.plot(x, anomaly_arr, lw=1.5, color="tab:orange", label="Anomaly score")
    ax.axvline(fault_at, color="red", ls="--", label="Fault injected")
    ax.set_title("Phase B — Anomaly Score over Time")
    ax.set_xlabel("Step"); ax.set_ylabel("Anomaly"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase C GridTwin KPIs
    ax = axes[1, 1]
    if phase_c is not None:
        rf, ss, soc = phase_c
        steps_c = np.arange(len(rf))
        ax.plot(steps_c, rf * 100, label="Renewable fraction %", color="tab:orange", lw=2)
        ax.plot(steps_c, ss * 100, label="Self-sufficiency %",   color="tab:blue",   lw=2)
        ax2 = ax.twinx()
        ax2.plot(steps_c, soc, color="tab:green", lw=1.5, ls="--", label="H2 SOC")
        ax2.set_ylabel("H2 SOC", color="tab:green")
        ax.set_title("Phase C — GridTwin System KPIs")
        ax.set_xlabel("Step"); ax.set_ylabel("%"); ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Phase C skipped\n(config not found)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    print("=" * 64)
    print("  HyTwin 2.0 — Digital Twin Demo")
    print("=" * 64)

    phase_a = _phase_a_fusion(args.steps)
    print()
    phase_b = _phase_b_fault(args.steps)
    print()
    phase_c = _phase_c_gridtwin(args.steps)
    print()

    if args.plot or args.save:
        out_dir = ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        save = args.save if args.save else str(out_dir / "demo_digital_twin.png")
        _make_plot(phase_a, phase_b, phase_c, save)
    else:
        print("  (pass --plot or --save <path> to generate charts)")

    print("=" * 64)


if __name__ == "__main__":
    main()
