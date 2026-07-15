"""
test_rl_environment.py
======================
Tests for the HyTwin 2.0 Gymnasium RL environment.
Run: pytest tests/test_rl_environment.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.rl.environment import H2GridEnv


# ---------------------------------------------------------------------------
# Shared grid config for all tests (small, fast)
# ---------------------------------------------------------------------------
GRID_CFG = {
    "wind_turbines": [{
        "id": "wt1",
        "params": {
            "rotor_diameter_m": 60.0,
            "hub_height_m": 60.0,
            "rated_power_kw": 300.0,
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
            "n_panels": 50,
            "panel_area_m2": 1.96,
            "eta_stc": 0.20,
            "temp_coeff_pmax": -0.004,
            "noct_c": 45.0,
            "rated_power_kw": 20.0,
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
            "cell_area_cm2": 200.0,
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
            "max_charge_rate_kg_s": 0.05,
            "max_discharge_rate_kg_s": 0.03,
            "boiloff_rate_per_day": 0.0,
        },
    }],
    "loads": [{
        "id": "load1",
        "params": {
            "base_load_kw": 150.0,
            "profile_type": "residential",
            "noise_std_fraction": 0.0,
            "seasonal_amplitude": 0.0,
            "demand_response_factor": 0.15,
        },
    }],
}

WEATHER_CFG = {
    "latitude_deg": 40.5,
    "longitude_deg": 14.8,
    "altitude_m": 50.0,
    "weibull_k": 2.0,
    "weibull_c": 6.5,
}


@pytest.fixture
def env():
    e = H2GridEnv(
        grid_config=GRID_CFG,
        weather_params=WEATHER_CFG,
        dt_seconds=600,
        episode_length=20,
    )
    yield e
    e.close()


# ===========================================================================
# Spaces
# ===========================================================================
class TestSpaces:

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (14,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (3,)

    def test_action_space_bounds(self, env):
        assert np.all(env.action_space.low  >= 0.0)
        assert np.all(env.action_space.high <= 1.0)


# ===========================================================================
# Reset
# ===========================================================================
class TestReset:

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (14,)
        assert isinstance(info, dict)

    def test_obs_in_bounds_after_reset(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= env.observation_space.low  - 1e-4)
        assert np.all(obs <= env.observation_space.high + 1e-4)

    def test_repeated_reset(self, env):
        for _ in range(3):
            obs, _ = env.reset(seed=1)
            assert obs.shape == (14,)

    def test_seed_reproducibility(self):
        e1 = H2GridEnv(GRID_CFG, WEATHER_CFG, dt_seconds=600, episode_length=10)
        e2 = H2GridEnv(GRID_CFG, WEATHER_CFG, dt_seconds=600, episode_length=10)
        obs1, _ = e1.reset(seed=7)
        obs2, _ = e2.reset(seed=7)
        np.testing.assert_array_almost_equal(obs1, obs2)
        e1.close(); e2.close()


# ===========================================================================
# Step
# ===========================================================================
class TestStep:

    def test_step_returns_correct_types(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_in_bounds_after_step(self, env):
        env.reset()
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)), "Observation contains NaN/Inf"
            if terminated or truncated:
                break

    def test_episode_terminates(self, env):
        """Episode should end within episode_length steps."""
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
            assert steps <= env._ep_len + 5, "Episode never terminated"
        assert done

    def test_info_contains_reward_components(self, env):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "reward_components" in info

    def test_zero_action_does_not_crash(self, env):
        env.reset()
        action = np.zeros(env.action_space.shape)
        obs, reward, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs))

    def test_full_action_does_not_crash(self, env):
        env.reset()
        action = np.ones(env.action_space.shape)
        obs, reward, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs))


# ===========================================================================
# Gymnasium API compliance (basic)
# ===========================================================================
class TestGymCompliance:

    def test_check_env(self):
        """gymnasium.utils.env_checker.check_env should pass without errors."""
        try:
            from gymnasium.utils.env_checker import check_env
        except ImportError:
            pytest.skip("gymnasium env_checker not available")

        small_env = H2GridEnv(
            grid_config=GRID_CFG,
            weather_params=WEATHER_CFG,
            dt_seconds=600,
            episode_length=10,
        )
        check_env(small_env, warn=True)
        small_env.close()
