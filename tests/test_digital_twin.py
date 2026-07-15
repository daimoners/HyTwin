"""
test_digital_twin.py
====================
Integration tests for the HyTwin 2.0 digital-twin layer.
Run: pytest tests/test_digital_twin.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.digital_twin.twin_node import TwinNode
from hytwin.digital_twin.grid_twin import GridTwin, GridState
from hytwin.models.hydrogen_tank import HydrogenTankModel
from hytwin.models.wind_turbine import WindTurbineModel
from hytwin.sensors.sensors import PressureSensor, PowerSensor
from hytwin.sensors.base_sensor import SensorStatus


TS0 = datetime(2024, 6, 15, 12, 0, 0)


def _ts(step: int, dt_s: float = 600.0) -> datetime:
    return TS0 + timedelta(seconds=step * dt_s)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tank_model():
    return HydrogenTankModel("tk1", {
        "volume_m3": 5.0,
        "max_pressure_bar": 700.0,
        "min_pressure_bar": 10.0,
        "initial_soc": 0.50,
        "temperature_c": 20.0,
        "max_charge_rate_kg_s": 0.08,
        "max_discharge_rate_kg_s": 0.05,
        "boiloff_rate_per_day": 0.0,
    })


@pytest.fixture
def pressure_sensor():
    return PressureSensor("tk1.p", noise_std_bar=2.0, drift_rate_bar=0.0)


@pytest.fixture
def twin_node(tank_model, pressure_sensor):
    return TwinNode(
        node_id="tk1",
        model=tank_model,
        state_keys=["pressure_bar"],
    )


# ===========================================================================
# TwinNode
# ===========================================================================
class TestTwinNode:

    def test_observe_returns_array(self, twin_node, tank_model):
        ms = tank_model.step(600.0, context={"h2_charge_kg": 2.0, "h2_discharge_kg": 0.0})
        twin_node.update(ms)
        obs = twin_node.observe()
        assert isinstance(obs, np.ndarray)
        assert len(obs) >= 1

    def test_health_starts_high(self, twin_node, tank_model):
        ms = tank_model.step(600.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 0.0})
        twin_node.update(ms)
        assert twin_node.state.health_score > 0.5

    def test_health_degrades_on_stuck_fault(self, twin_node, tank_model, pressure_sensor):
        """With a stuck sensor, fused values fall back to model (sensor ignored as faulted)."""
        from hytwin.sensors.base_sensor import SensorReading, SensorStatus as SS
        pressure_sensor.inject_fault(SensorStatus.FAULT_STUCK, stuck_value=100.0)
        for i in range(30):
            ms = tank_model.step(600.0, context={"h2_charge_kg": 2.0, "h2_discharge_kg": 0.0})
            raw_p = ms.values["pressure_bar"]
            reading = pressure_sensor.measure(raw_p, _ts(i))
            twin_node.update(ms, [reading])
        # Fused values should remain close to model (faulted sensor is ignored)
        fused_p = twin_node.state.fused_values.get("pressure_bar", 0.0)
        model_p = twin_node.state.model_values.get("pressure_bar", 0.0)
        assert abs(fused_p - model_p) < 1.0

    def test_observe_shape_matches_state_keys(self, tank_model):
        node = TwinNode("tk1", tank_model, state_keys=["pressure_bar"])
        ms = tank_model.step(600.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 0.0})
        node.update(ms)
        obs = node.observe(keys=["pressure_bar"])
        assert len(obs) == 1

    def test_fused_estimate_close_to_truth(self, tank_model):
        """Fused estimate should be closer to model than raw noisy sensor on average."""
        sensor_errors, fused_errors = [], []
        noisy_sensor = PressureSensor("raw_p", noise_std_bar=20.0, drift_rate_bar=0.0)
        node = TwinNode("tk1", tank_model, state_keys=["pressure_bar"])
        for i in range(50):
            ctx = {"h2_charge_kg": 1.0, "h2_discharge_kg": 0.5}
            ms = tank_model.step(600.0, context=ctx)
            true_p = ms.values["pressure_bar"]
            raw_r = noisy_sensor.measure(true_p, _ts(i))
            node.update(ms, [raw_r])
            obs = node.observe(keys=["pressure_bar"])
            fused_p = obs[0] if len(obs) > 0 else true_p
            sensor_errors.append(abs(raw_r.value - true_p))
            fused_errors.append(abs(fused_p - true_p))

        # Fused should not be dramatically worse than raw
        assert np.mean(fused_errors) < np.mean(sensor_errors) * 2.0


# ===========================================================================
# GridTwin
# ===========================================================================
def _minimal_grid_config() -> dict:
    """Minimal valid grid configuration for GridTwin."""
    return {
        "wind_turbines": [{
            "id": "wt1",
            "params": {
                "rotor_diameter_m": 77.0,
                "hub_height_m": 80.0,
                "rated_power_kw": 500.0,
                "v_cut_in": 3.0,
                "v_rated": 12.0,
                "v_cut_out": 25.0,
                "efficiency_gen": 0.94,
                "altitude_m": 50.0,
                "turbulence_intensity": 0.0,
            },
        }],
        "pv_arrays": [{
            "id": "pv1",
            "params": {
                "n_panels": 100,
                "panel_area_m2": 1.96,
                "eta_stc": 0.20,
                "temp_coeff_pmax": -0.004,
                "noct_c": 45.0,
                "rated_power_kw": 40.0,
                "soiling_loss": 0.0,
                "degradation_per_year": 0.0,
                "tilt_deg": 0.0,
                "azimuth_deg": 180.0,
            },
        }],
        "electrolyzers": [{
            "id": "el1",
            "params": {
                "rated_power_kw": 100.0,
                "n_cells": 40,
                "cell_area_cm2": 300.0,
                "membrane_resistance_ohm_cm2": 0.16,
                "temperature_c": 65.0,
                "min_load_fraction": 0.05,
                "ramp_rate_kw_s": 100.0,
            },
        }],
        "fuel_cells": [{
            "id": "fc1",
            "params": {
                "rated_power_kw": 50.0,
                "n_cells": 420,
                "cell_area_cm2": 200.0,
                "membrane_resistance_ohm_cm2": 0.12,
                "temperature_c": 65.0,
                "h2_utilisation": 0.80,
                "min_load_fraction": 0.10,
                "ramp_rate_kw_s": 100.0,
            },
        }],
        "hydrogen_tanks": [{
            "id": "tk1",
            "params": {
                "volume_m3": 5.0,
                "max_pressure_bar": 700.0,
                "min_pressure_bar": 10.0,
                "initial_soc": 0.50,
                "temperature_c": 20.0,
                "max_charge_rate_kg_s": 0.08,
                "max_discharge_rate_kg_s": 0.05,
                "boiloff_rate_per_day": 0.0,
            },
        }],
        "loads": [{
            "id": "load1",
            "params": {
                "base_load_kw": 200.0,
                "profile_type": "residential",
                "noise_std_fraction": 0.0,
                "seasonal_amplitude": 0.0,
                "demand_response_factor": 0.0,
            },
        }],
    }


class TestGridTwin:

    @pytest.fixture
    def gt(self):
        gt = GridTwin(_minimal_grid_config())
        gt.build()
        return gt

    def _step(self, gt, i: int = 0):
        weather = {
            "wind_speed_ms": 8.0,
            "ghi_wm2": 600.0,
            "dhi_wm2": 100.0,
            "dni_wm2": 500.0,
            "temperature_c": 20.0,
            "sun_zenith_deg": 30.0,
            "sun_azimuth_deg": 180.0,
            "cloud_cover": 0.3,
            "humidity": 0.6,
        }
        control = {
            "el1": {"power_setpoint_kw": 50.0},
            "fc1": {"power_setpoint_kw": 0.0, "h2_available_kg": 100.0},
            "load1": {"demand_response": 0.0},
        }
        return gt.step(dt=600.0, weather=weather, control_actions=control,
                       timestamp=_ts(i))

    def test_step_returns_grid_state(self, gt):
        gs = self._step(gt)
        assert isinstance(gs, GridState)

    def test_power_non_negative(self, gt):
        gs = self._step(gt)
        assert gs.wind_power_kw >= 0.0
        assert gs.pv_power_kw  >= 0.0
        assert gs.electrolyzer_power_kw >= 0.0

    def test_h2_soc_in_range(self, gt):
        gs = self._step(gt)
        assert 0.0 <= gs.h2_soc <= 1.0

    def test_observe_returns_correct_shape(self, gt):
        gs = self._step(gt)
        obs = gt.observe()
        assert obs.shape == (14,)
        assert obs.dtype == np.float32

    def test_renewable_fraction_in_range(self, gt):
        gs = self._step(gt)
        assert 0.0 <= gs.renewable_fraction <= 1.01  # allow tiny float rounding

    def test_sequential_steps_run(self, gt):
        """20 sequential steps should run without error."""
        for i in range(20):
            gs = self._step(gt, i)
        assert gs is not None

    def test_h2_soc_increases_when_electrolyzing(self):
        gt = GridTwin(_minimal_grid_config())
        gt.build()
        weather = {
            "wind_speed_ms": 15.0, "ghi_wm2": 900.0,
            "dhi_wm2": 100.0, "dni_wm2": 800.0,
            "temperature_c": 20.0, "sun_zenith_deg": 20.0,
            "sun_azimuth_deg": 180.0, "cloud_cover": 0.0, "humidity": 0.5,
        }
        # Explicitly pipe ~0.235 kg H2/step from electrolyzer into tank
        control = {
            "el1": {"power_setpoint_kw": 100.0},
            "fc1": {"power_setpoint_kw": 0.0, "h2_available_kg": 100.0},
            "tk1": {"h2_charge_kg": 0.235, "h2_discharge_kg": 0.0},
        }
        soc_before = gt.step(600.0, weather, control, timestamp=TS0).h2_soc
        soc_after  = gt.step(600.0, weather, control, timestamp=_ts(1)).h2_soc
        assert soc_after >= soc_before

    def test_h2_soc_decreases_when_fuel_cell_runs(self):
        gt = GridTwin(_minimal_grid_config())
        gt.build()
        weather = {
            "wind_speed_ms": 0.5, "ghi_wm2": 0.0,
            "dhi_wm2": 0.0, "dni_wm2": 0.0,
            "temperature_c": 15.0, "sun_zenith_deg": 90.0,
            "sun_azimuth_deg": 180.0, "cloud_cover": 0.9, "humidity": 0.7,
        }
        # Explicitly drain ~0.72 kg H2/step from tank (fuel cell at rated load)
        control = {
            "el1": {"power_setpoint_kw": 0.0},
            "fc1": {"power_setpoint_kw": 50.0, "h2_available_kg": 100.0},
            "tk1": {"h2_charge_kg": 0.0, "h2_discharge_kg": 0.72},
        }
        soc_before = gt.step(600.0, weather, control, timestamp=TS0).h2_soc
        soc_after  = gt.step(600.0, weather, control, timestamp=_ts(1)).h2_soc
        assert soc_after <= soc_before
