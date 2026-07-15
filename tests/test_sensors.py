"""
test_sensors.py
===============
Unit tests for HyTwin virtual sensor layer.
Run: pytest tests/test_sensors.py -v
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.sensors.base_sensor import SensorStatus, SensorReading
from hytwin.sensors.sensors import (
    PowerSensor,
    VoltageSensor,
    CurrentSensor,
    PressureSensor,
    TemperatureSensor,
    FlowSensor,
    HydrogenLevelSensor,
    IrradianceSensor,
    WindSpeedSensor,
)
from hytwin.sensors.sensor_manager import SensorManager


TS0 = datetime(2024, 6, 15, 12, 0, 0)


def _ts(step: int, dt_s: float = 60.0) -> datetime:
    return TS0 + timedelta(seconds=step * dt_s)


# ===========================================================================
# Base sensor behaviour
# ===========================================================================
class TestBaseSensorNoise:

    def test_noiseless_sensor_exact(self):
        s = PowerSensor("p", noise_std_kw=0.0, drift_rate_kw=0.0, quantisation_step=0.0)
        r = s.measure(100.0, TS0)
        assert r.value == pytest.approx(100.0)

    def test_gaussian_noise_statistics(self):
        """300 readings from a sensor with σ=10 should have σ ≈ 10."""
        s = PowerSensor("p", noise_std_kw=10.0, drift_rate_kw=0.0, quantisation_step=0.0)
        vals = [s.measure(500.0, _ts(i)).value for i in range(300)]
        assert pytest.approx(10.0, rel=0.30) == np.std(vals)

    def test_quantisation_rounding(self):
        s = PowerSensor("p", noise_std_kw=0.0, drift_rate_kw=0.0, quantisation_step=5.0)
        r = s.measure(103.3, TS0)
        # Should round to nearest 5
        assert r.value % 5.0 == pytest.approx(0.0, abs=0.01)

    def test_quality_is_high_for_good_sensor(self):
        s = PowerSensor("p", noise_std_kw=1.0, drift_rate_kw=0.0)
        r = s.measure(100.0, TS0)
        assert r.quality > 0.8

    def test_reading_has_timestamp(self):
        s = TemperatureSensor("t", noise_std_c=0.5)
        r = s.measure(25.0, TS0)
        assert r.timestamp == TS0


# ===========================================================================
# Drift
# ===========================================================================
class TestDrift:

    def test_drift_accumulates(self):
        s = PressureSensor("p", noise_std_bar=0.0, drift_rate_bar=1.0)
        vals = [s.measure(200.0, _ts(i)).value for i in range(50)]
        # Drift is a random walk — after many steps, spread from true value should be nonzero
        deltas = np.array(vals) - 200.0
        # At least one reading should deviate significantly from zero
        assert np.max(np.abs(deltas)) > 0.5

    def test_zero_drift_is_stable(self):
        s = PressureSensor("p", noise_std_bar=0.0, drift_rate_bar=0.0)
        vals = [s.measure(200.0, _ts(i)).value for i in range(50)]
        assert all(v == pytest.approx(200.0) for v in vals)


# ===========================================================================
# Delay
# ===========================================================================
class TestDelay:

    def test_delay_returns_old_value(self):
        """Sensor with 3-step delay should return value from 3 steps ago."""
        s = PowerSensor("p", noise_std_kw=0.0, drift_rate_kw=0.0, delay_steps=3)
        readings = []
        true_vals = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        for i, v in enumerate(true_vals):
            r = s.measure(v, _ts(i))
            readings.append(r.value)
        # At step 3 (0-indexed), the delayed output should be from value at step 0
        # Buffer starts empty; first 3 steps return 0 or initial fill
        # Step 3 output = true_vals[0] = 100.0
        assert readings[3] == pytest.approx(true_vals[0], abs=0.01)

    def test_no_delay_is_immediate(self):
        s = PowerSensor("p", noise_std_kw=0.0, drift_rate_kw=0.0, delay_steps=0)
        r = s.measure(999.0, TS0)
        assert r.value == pytest.approx(999.0)


# ===========================================================================
# Fault injection
# ===========================================================================
class TestFaultInjection:

    def test_stuck_fault_constant_output(self):
        s = WindSpeedSensor("w", noise_std_ms=0.0)
        s.inject_fault(SensorStatus.FAULT_STUCK, stuck_value=7.5)
        for v in [5.0, 10.0, 3.0, 15.0]:
            r = s.measure(v, TS0)
            assert r.value == pytest.approx(7.5)
            assert r.status == SensorStatus.FAULT_STUCK

    def test_offline_fault_returns_nan(self):
        s = WindSpeedSensor("w", noise_std_ms=0.0)
        s.inject_fault(SensorStatus.FAULT_OFFLINE)
        r = s.measure(10.0, TS0)
        assert r.value is None or (isinstance(r.value, float) and math.isnan(r.value))
        assert r.status == SensorStatus.FAULT_OFFLINE

    def test_clear_fault_restores_normal(self):
        s = WindSpeedSensor("w", noise_std_ms=0.1)
        s.inject_fault(SensorStatus.FAULT_STUCK, stuck_value=0.0)
        s.clear_fault()
        r = s.measure(10.0, TS0)
        assert r.status == SensorStatus.OK
        assert r.value is not None and not math.isnan(r.value)

    def test_spike_fault_produces_spike(self):
        """With a spike fault the sensor should output an outlier on inject step."""
        s = PowerSensor("p", noise_std_kw=0.0)
        s.inject_fault(SensorStatus.FAULT_SPIKE)
        r = s.measure(100.0, TS0)
        assert r.status == SensorStatus.FAULT_SPIKE

    def test_quality_low_under_fault(self):
        s = PressureSensor("p", noise_std_bar=0.0)
        s.inject_fault(SensorStatus.FAULT_STUCK, stuck_value=0.0)
        r = s.measure(200.0, TS0)
        assert r.quality < 0.5


# ===========================================================================
# All concrete sensor types
# ===========================================================================
class TestAllSensorTypes:

    @pytest.mark.parametrize("SensorClass, true_val", [
        (PowerSensor,        100.0),
        (VoltageSensor,      380.0),
        (CurrentSensor,      50.0),
        (PressureSensor,     350.0),
        (TemperatureSensor,  25.0),
        (FlowSensor,         2.5),
        (HydrogenLevelSensor, 30.0),
        (IrradianceSensor,   700.0),
        (WindSpeedSensor,    8.0),
    ])
    def test_sensor_produces_reading(self, SensorClass, true_val):
        # Instantiate with no extra args — use defaults
        sensor = SensorClass(sensor_id="test")
        r = sensor.measure(true_val, TS0)
        assert isinstance(r, SensorReading)
        assert r.sensor_id == "test"
        assert r.status in SensorStatus
        assert r.value is not None or r.status == SensorStatus.FAULT_OFFLINE


# ===========================================================================
# SensorManager
# ===========================================================================
class TestSensorManager:

    def test_register_and_update(self):
        manager = SensorManager()
        sensor = PowerSensor("pv.power", noise_std_kw=5.0)
        manager.register(sensor, model_key="pv1.power_kw")

        snapshot = {"pv1.power_kw": 250.0}
        readings = manager.update(snapshot, TS0)

        assert len(readings) == 1
        assert readings[0].sensor_id == "pv.power"
        assert abs(readings[0].value - 250.0) < 50.0  # within ±3σ

    def test_missing_key_does_not_raise(self):
        manager = SensorManager()
        sensor = PowerSensor("ghost", noise_std_kw=1.0)
        manager.register(sensor, model_key="nonexistent.key")
        # Should not raise, just return no readings
        readings = manager.update({}, TS0)
        assert readings == []

    def test_multiple_sensors(self):
        manager = SensorManager()
        for i in range(5):
            manager.register(PowerSensor(f"s{i}", noise_std_kw=1.0), model_key=f"key{i}")
        snapshot = {f"key{i}": float(i * 10) for i in range(5)}
        readings = manager.update(snapshot, TS0)
        assert len(readings) == 5
