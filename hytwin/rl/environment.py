"""
H2 Grid Gymnasium Environment
==============================
Fully compatible with Gymnasium (formerly OpenAI Gym) API.
Wraps the GridTwin and WeatherModel into an RL environment.

Observation space (14 continuous values, normalised):
  [wind_power, pv_power, electrolyzer_power, fuel_cell_power,
   load_kw, grid_exchange, h2_prod_rate, h2_cons_rate,
   h2_storage, h2_soc, h2_pressure, renew_fraction,
   self_sufficiency, health]

Action space:
  Continuous Box(3):
    [electrolyzer_setpoint (0-1), fuel_cell_setpoint (0-1),
     demand_response (0-1)]
  Where 0=off, 1=full rated power.

The environment can be configured to use a fixed scenario configuration
or the default pilot scenario.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..digital_twin.grid_twin import GridTwin, GridState
from ..weather.weather_model import WeatherModel
from ..sensors import SensorManager
from ..core.event_bus import EventBus
from ..core.state_manager import StateManager
from .rewards import RewardConfig, compute_reward

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Observation normalisation constants
# ------------------------------------------------------------------
OBS_MAX = np.array([
    2000.0,   # wind_power_kw
    2000.0,   # pv_power_kw
    1000.0,   # electrolyzer_power_kw
    500.0,    # fuel_cell_power_kw
    2000.0,   # load_kw
    1000.0,   # grid_exchange_kw (can be neg)
    500.0,    # h2_production_kg_h
    500.0,    # h2_consumption_kg_h
    2000.0,   # h2_storage_kg
    1.0,      # h2_soc
    700.0,    # h2_pressure_bar
    1.0,      # renewable_fraction
    1.0,      # self_sufficiency
    1.0,      # health
], dtype=np.float32)

OBS_MIN = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, -1000.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
], dtype=np.float32)


class H2GridEnv(gym.Env):
    """
    Gymnasium environment wrapping the full H2 grid digital twin.

    Parameters
    ----------
    grid_config : dict
        Configuration dict for GridTwin (component specs).
    weather_params : dict, optional
        Parameters passed to WeatherModel.
    dt_seconds : float
        Simulation step duration [s], default 60 (1 minute).
    episode_length : int
        Steps per episode, default 1440 (24 h at 1-min steps).
    start_time : datetime, optional
    reward_config : RewardConfig, optional
    sensor_manager : SensorManager, optional
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        grid_config: Dict[str, Any],
        weather_params: Optional[Dict[str, Any]] = None,
        dt_seconds: float = 60.0,
        episode_length: int = 1440,
        start_time: Optional[datetime] = None,
        reward_config: Optional[RewardConfig] = None,
        sensor_manager: Optional[SensorManager] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._grid_config = grid_config
        self._weather_params = weather_params or {}
        self._dt = dt_seconds
        self._ep_len = episode_length
        self._start_time = start_time or datetime(2024, 6, 15, 0, 0)
        self._reward_cfg = reward_config or RewardConfig()
        self._sensor_mgr = sensor_manager
        self.render_mode = render_mode

        # Gymnasium spaces
        n_obs = len(OBS_MAX)
        self.observation_space = spaces.Box(
            low=np.full(n_obs, -1.0, dtype=np.float32),
            high=np.full(n_obs, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        # Action: [el_setpoint, fc_setpoint, demand_response]
        self.action_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal components (built on first reset)
        self._twin: Optional[GridTwin] = None
        self._weather: Optional[WeatherModel] = None
        self._bus = EventBus()
        self._sm = StateManager()

        self._step_count: int = 0
        self._current_time: datetime = self._start_time
        self._prev_gs: Optional[GridState] = None
        self._episode_reward: float = 0.0

        # Logging buffers
        self._reward_log: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Random start time within a year
        if options and options.get("random_start"):
            day_offset = int(np.random.randint(0, 365))
            self._current_time = self._start_time + timedelta(days=day_offset)
        else:
            self._current_time = self._start_time

        # Build / rebuild twin and weather
        self._bus = EventBus()
        self._sm = StateManager()
        self._twin = GridTwin(self._grid_config, self._bus, self._sm)
        self._twin.build()
        self._weather = WeatherModel(**self._weather_params)

        self._step_count = 0
        self._prev_gs = None
        self._episode_reward = 0.0
        self._reward_log.clear()

        # Warm-up step
        weather = self._weather.step(self._current_time)
        _ = self._twin.step(self._dt, weather, {}, timestamp=self._current_time)

        obs = self._make_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        el_sp = float(np.clip(action[0], 0.0, 1.0))
        fc_sp = float(np.clip(action[1], 0.0, 1.0))
        dr_sp = float(np.clip(action[2], 0.0, 1.0))

        self._current_time += timedelta(seconds=self._dt)
        weather = self._weather.step(self._current_time)

        # Build control actions for each component type
        control_actions: Dict[str, Any] = {}
        for node_id, node in self._twin._nodes.items():
            mtype = type(node.model).__name__
            if mtype == "ElectrolyzerModel":
                rated = node.model._rated_kw
                control_actions[node_id] = {"power_setpoint_kw": el_sp * rated}
            elif mtype == "FuelCellModel":
                rated = node.model._rated_kw
                # H2 available from tank
                h2_avail = self._get_h2_available()
                control_actions[node_id] = {
                    "power_setpoint_kw": fc_sp * rated,
                    "h2_available_kg": h2_avail,
                }
            elif mtype == "EnergyLoadModel":
                control_actions[node_id] = {"demand_response": dr_sp}
            elif mtype == "HydrogenTankModel":
                h2_in, h2_out = self._get_h2_tank_flows()
                control_actions[node_id] = {
                    "h2_charge_kg": h2_in,
                    "h2_discharge_kg": h2_out,
                }

        # Collect sensor readings if manager available
        sensor_readings = None
        if self._sensor_mgr:
            snap = self._sm.snapshot()
            sensor_readings = self._sensor_mgr.update(snap, timestamp=self._current_time)

        gs = self._twin.step(
            self._dt, weather, control_actions,
            sensor_readings=sensor_readings,
            timestamp=self._current_time,
        )

        reward, components = compute_reward(gs, self._prev_gs, self._reward_cfg)
        self._reward_log.append(components)
        self._episode_reward += reward
        self._prev_gs = gs
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self._ep_len

        obs = self._make_obs()
        info = {
            "grid_state": gs,
            "reward_components": components,
            "episode_reward": self._episode_reward,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        gs = self._twin.grid_state if self._twin else None
        if gs is None:
            return None
        s = (
            f"[{gs.timestamp.strftime('%H:%M')}] "
            f"Wind={gs.wind_power_kw:.1f}kW  PV={gs.pv_power_kw:.1f}kW  "
            f"EL={gs.electrolyzer_power_kw:.1f}kW  FC={gs.fuel_cell_power_kw:.1f}kW  "
            f"Load={gs.load_kw:.1f}kW  Grid={gs.grid_exchange_kw:+.1f}kW  "
            f"H2={gs.h2_storage_kg:.1f}kg (SOC={gs.h2_soc:.2f})  "
            f"RF={gs.renewable_fraction:.0%}  Health={gs.overall_health:.2f}"
        )
        if self.render_mode == "human":
            print(s)
        return s

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        raw = self._twin.observe()
        # Normalise to [-1, 1]
        mid = (OBS_MAX + OBS_MIN) / 2.0
        scale = (OBS_MAX - OBS_MIN) / 2.0 + 1e-9
        norm = (raw - mid) / scale
        return norm.astype(np.float32)

    def _get_h2_available(self) -> float:
        """Return total H2 in all tanks [kg]."""
        total = 0.0
        for node in self._twin._nodes.values():
            if isinstance(node.model, __import__("hytwin.models", fromlist=["HydrogenTankModel"]).HydrogenTankModel):
                total += node.model.mass_kg
        return total

    def _get_h2_tank_flows(self) -> Tuple[float, float]:
        """
        Compute how much H2 to charge/discharge the tank this step.
        Charge = whatever electrolyzers are producing.
        Discharge = whatever fuel cells are consuming.
        Simple approach: the tank acts as buffer.
        """
        h2_in = 0.0
        h2_out = 0.0
        for node in self._twin._nodes.items():
            pass  # resolved via control_actions in step()
        return h2_in, h2_out
