"""
demo_sensors.py
===============
Demonstrates HyTwin virtual-sensor capabilities:

  1. Gaussian noise + quantisation on a PV power sensor
  2. Slow drift on a H2 pressure sensor
  3. Fault injection (stuck value, spike, offline) on a wind-power sensor
  4. Anomaly / quality scoring

Run the demo then examine the console output and the generated plot.

Usage
-----
    python demos/demo_sensors.py [--steps 200] [--plot]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.sensors.base_sensor import SensorStatus
from hytwin.sensors.sensors import (
    PowerSensor,
    PressureSensor,
    WindSpeedSensor,
    IrradianceSensor,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HyTwin — virtual-sensor demo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: run a sensor for N steps against a synthetic true signal
# ---------------------------------------------------------------------------
def _run_sensor(sensor, true_values: np.ndarray, dt_s: float = 60.0):
    ts_base = datetime(2024, 6, 15, 8, 0, 0)
    readings, true_vals, qualities = [], [], []
    for i, v in enumerate(true_values):
        ts = ts_base + timedelta(seconds=i * dt_s)
        r = sensor.measure(v, ts)
        readings.append(r.value)
        true_vals.append(v)
        qualities.append(r.quality)
    return np.array(true_vals), np.array(readings), np.array(qualities)


# ---------------------------------------------------------------------------
# Scenario 1 — Noise + Quantisation (PV power)
# ---------------------------------------------------------------------------
def _scenario_noise(steps: int):
    print("  ── Scenario 1: Gaussian noise + quantisation on PV power sensor")
    sensor = PowerSensor(
        sensor_id="pv1.power",
        noise_std_kw=8.0,          # ±8 kW noise
        quantisation_step=5.0,  # 5 kW steps (coarse meter)
        drift_rate_kw=0.0,
    )
    # Trapezoid: ramp up to 250 kW, hold, ramp down
    t = np.linspace(0, np.pi, steps)
    true_vals = 250.0 * np.sin(t) ** 2  # smooth daily arc

    true_s, meas_s, qual_s = _run_sensor(sensor, true_vals)
    err = meas_s - true_s
    print(f"    RMSE (noise+quant) = {np.sqrt(np.mean(err**2)):.2f} kW")
    print(f"    Mean quality       = {qual_s.mean():.3f}")
    return true_s, meas_s, qual_s


# ---------------------------------------------------------------------------
# Scenario 2 — Drift (H2 pressure sensor)
# ---------------------------------------------------------------------------
def _scenario_drift(steps: int):
    print("  ── Scenario 2: Progressive drift on H2-tank pressure sensor")
    sensor = PressureSensor(
        sensor_id="tk1.pressure",
        noise_std_bar=1.5,
        drift_rate_bar=0.5,        # 0.5 bar per step → +50 bar error after 100 steps
    )
    true_vals = 200.0 + 30.0 * np.sin(np.linspace(0, 2 * np.pi, steps))

    true_s, meas_s, qual_s = _run_sensor(sensor, true_vals)
    err = meas_s - true_s
    print(f"    Max drift error    = {np.max(np.abs(err)):.2f} bar  (step {np.argmax(np.abs(err))})")
    print(f"    Final mean quality = {qual_s[-20:].mean():.3f}")

    # Reset drift → sensor returns to normal
    sensor.drift_accumulated = 0.0
    print("    [drift reset] sensor recalibrated")
    return true_s, meas_s, qual_s


# ---------------------------------------------------------------------------
# Scenario 3 — Fault injection (wind speed sensor)
# ---------------------------------------------------------------------------
def _scenario_faults(steps: int):
    print("  ── Scenario 3: Fault injection on wind-speed sensor")
    sensor = WindSpeedSensor(
        sensor_id="wt1.wind_speed",
        noise_std_ms=0.3,
    )
    rng = np.random.default_rng(42)
    true_vals = 8.0 + 3.0 * np.sin(np.linspace(0, 3 * np.pi, steps)) + rng.normal(0, 0.5, steps)
    true_vals = np.clip(true_vals, 0, 25)

    ts_base = datetime(2024, 6, 15, 0, 0, 0)
    meas, true_out, statuses = [], [], []

    fault_log = {}
    for i, v in enumerate(true_vals):
        ts = ts_base + timedelta(seconds=i * 60)

        # Inject stuck fault at step 60
        if i == 60:
            sensor.inject_fault(SensorStatus.FAULT_STUCK, stuck_value=8.5)
            fault_log[i] = "STUCK"
        # Inject spike at step 100
        elif i == 100:
            sensor.clear_fault()
            sensor.inject_fault(SensorStatus.FAULT_SPIKE)
            fault_log[i] = "SPIKE"
        # Back to normal at step 101
        elif i == 101:
            sensor.clear_fault()
            fault_log[i] = "CLEAR"
        # Offline at step 140
        elif i == 140:
            sensor.inject_fault(SensorStatus.FAULT_OFFLINE)
            fault_log[i] = "OFFLINE"
        elif i == 155:
            sensor.clear_fault()
            fault_log[i] = "CLEAR"

        r = sensor.measure(v, ts)
        meas.append(r.value if r.value is not None else float("nan"))
        true_out.append(v)
        statuses.append(r.status.name)

    meas = np.array(meas, dtype=float)
    true_out = np.array(true_out)

    print(f"    Fault events injected: {fault_log}")
    stuck_err = np.nanmean(np.abs(meas[60:100] - true_out[60:100]))
    print(f"    Mean stuck-fault error (steps 60–99) = {stuck_err:.2f} m/s")
    nan_pct = 100 * np.sum(np.isnan(meas)) / len(meas)
    print(f"    NaN (offline) fraction = {nan_pct:.1f}%")
    return np.array(true_out), meas, statuses


# ---------------------------------------------------------------------------
# Scenario 4 — Delay (irradiance sensor)
# ---------------------------------------------------------------------------
def _scenario_delay(steps: int):
    print("  ── Scenario 4: Measurement delay on irradiance sensor")
    sensor = IrradianceSensor(
        sensor_id="pv1.irradiance",
        noise_std_wm2=5.0,
        delay_steps=5,          # 5-step lag
    )
    t = np.linspace(0, np.pi, steps)
    true_vals = 800.0 * np.sin(t) ** 2

    true_s, meas_s, _ = _run_sensor(sensor, true_vals, dt_s=60.0)
    lag_corr = np.corrcoef(true_s[5:], meas_s[:-5])[0, 1]
    fwd_corr = np.corrcoef(true_s, meas_s)[0, 1]
    print(f"    Correlation (no lag)   = {fwd_corr:.4f}")
    print(f"    Correlation (5-lag)    = {lag_corr:.4f}  (should be higher)")
    return true_s, meas_s


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _make_plot(noise_trio, drift_trio, fault_trio, delay_duo, save_path: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg" if not save_path == "" else "Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("HyTwin — Virtual Sensor Showcase", fontsize=14, fontweight="bold")

    # ── Noise + Quantisation ─────────────────────────────────────────────────
    ax = axes[0, 0]
    true_s, meas_s, qual_s = noise_trio
    steps = len(true_s)
    ax.plot(true_s, label="True PV power", lw=2, color="tab:orange")
    ax.plot(meas_s, label="Sensor (noise+quant)", alpha=0.7, color="tab:red", lw=1)
    ax.set_title("Noise + Quantisation (PV Power)")
    ax.set_xlabel("Step"); ax.set_ylabel("kW"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Drift ────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    true_s, meas_s, qual_s = drift_trio
    ax.plot(true_s, label="True pressure", lw=2, color="tab:blue")
    ax.plot(meas_s, label="Measured (drifting)", alpha=0.7, color="tab:purple", lw=1)
    ax.fill_between(range(len(true_s)), true_s, meas_s, alpha=0.15, color="tab:purple", label="Drift error")
    ax.set_title("Drift (H2 Tank Pressure)")
    ax.set_xlabel("Step"); ax.set_ylabel("bar"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Fault injection ──────────────────────────────────────────────────────
    ax = axes[1, 0]
    true_s, meas_s, statuses = fault_trio
    ok_mask   = np.array([s == "OK"           for s in statuses])
    stuck_m   = np.array([s == "FAULT_STUCK"  for s in statuses])
    spike_m   = np.array([s == "FAULT_SPIKE"  for s in statuses])
    off_m     = np.array([s == "FAULT_OFFLINE" for s in statuses])
    x = np.arange(len(true_s))
    ax.plot(x, true_s, lw=2, color="tab:green", label="True wind speed")
    ax.scatter(x[ok_mask], meas_s[ok_mask], s=3, color="tab:green", label="OK")
    ax.scatter(x[stuck_m], meas_s[stuck_m], s=10, color="tab:orange", label="STUCK")
    ax.scatter(x[spike_m], meas_s[spike_m], s=50, color="tab:red", marker="^", label="SPIKE")
    ax.scatter(x[off_m],  [float("nan")] * np.sum(off_m), s=10, color="gray", label="OFFLINE")
    ax.set_title("Fault Injection (Wind Speed Sensor)")
    ax.set_xlabel("Step"); ax.set_ylabel("m/s"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Delay ────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    true_s, meas_s = delay_duo
    x = np.arange(len(true_s))
    ax.plot(x, true_s, lw=2, color="gold", label="True irradiance")
    ax.plot(x, meas_s, lw=1.5, color="darkorange", alpha=0.8, label="Delayed (5 steps)")
    ax.set_title("Measurement Delay (Irradiance Sensor)")
    ax.set_xlabel("Step"); ax.set_ylabel("W/m²"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    print("=" * 64)
    print("  HyTwin — Virtual Sensor Demo")
    print("=" * 64)

    noise_trio  = _scenario_noise(args.steps)
    drift_trio  = _scenario_drift(args.steps)
    fault_trio  = _scenario_faults(args.steps)
    delay_duo   = _scenario_delay(args.steps)

    print()
    if args.plot or args.save:
        out_path = Path(ROOT / "output")
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = args.save if args.save else str(out_path / "demo_sensors.png")
        _make_plot(noise_trio, drift_trio, fault_trio, delay_duo, save_path)
    else:
        print("  (pass --plot to display charts, --save <path> to save them)")

    print("=" * 64)


if __name__ == "__main__":
    main()
