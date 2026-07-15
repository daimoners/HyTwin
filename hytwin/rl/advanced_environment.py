"""
Advanced H2 Grid Gymnasium Environment
=======================================
Extended version of H2GridEnv that adds:

  • Explicit grid connection model (non-renewable, capacity-limited, with outages)
  • Time-varying electricity price (EnergyCostModel / Italian PUN structure)
  • Extended observation space (19-D):
      [wind, pv, el_power, fc_power, load, grid_exchange, grid_connection,
       h2_prod, h2_cons, h2_storage, h2_soc, h2_pressure,
       renew_frac, self_sufficiency, health,
       price_norm, grid_available, hour_sin, hour_cos]
  • Extended action space (4-D):
      [el_setpoint (0-1), fc_setpoint (0-1), demand_response (0-1),
       grid_import_fraction (0-1)]
  • Extended reward that penalises electricity cost and rewards
    renewable self-sufficiency and H₂ storage efficiency

This environment is designed for training agents that learn to
optimise long-run renewable utilisation across full plant lifetime
simulations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
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
from ..models.energy_cost import EnergyCostModel
from .forecast_utils import (
    N_FORECAST_FEATURES_PER_STEP,
    FORECAST_FEATURE_NAMES,
    build_forecast_features,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Observation normalisation
# ------------------------------------------------------------------
OBS_MAX = np.array([
    2000.0,   # wind_power_kw
    2000.0,   # pv_power_kw
    1000.0,   # electrolyzer_power_kw
    500.0,    # fuel_cell_power_kw
    2000.0,   # load_kw
    1000.0,   # grid_exchange_kw
    1000.0,   # grid_connection_kw
    500.0,    # h2_production_kg_h
    500.0,    # h2_consumption_kg_h
    2000.0,   # h2_storage_kg
    1.0,      # h2_soc
    700.0,    # h2_pressure_bar
    1.0,      # renewable_fraction
    1.0,      # grid_self_sufficiency
    1.0,      # overall_health
    1.0,      # energy_price_norm (price / 0.50)
    1.0,      # grid_available (0/1)
    1.0,      # hour_sin
    1.0,      # hour_cos
], dtype=np.float32)

OBS_MIN = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, -1000.0, -200.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, -1.0, -1.0,
], dtype=np.float32)

N_BASE_OBS = len(OBS_MAX)
N_ACT = 4


@dataclass
class AdvancedRewardConfig:
    """
    Weights for the extended reward signal.

    Positive weights add reward; negative weights subtract (penalty).
    """
    # Core objectives
    w_self_sufficiency: float = 0.25      # secondary objective (not dominant)
    w_renewable_fraction: float = 0.20    # secondary objective (not dominant)
    w_demand_met: float = 1.0            # always meet load
    w_unmet_demand_penalty: float = -14.0  # strong penalty on unmet demand (quadratic)
    w_unmet_hard_penalty: float = -30.0    # hard penalty once unmet exceeds threshold
    unmet_hard_threshold: float = 0.05
    unmet_hard_power: float = 2.0
    w_supply_priority: float = -6.0        # additional soft-priority penalty for unmet demand
    w_demand_response_penalty: float = -0.8  # penalise demand shedding usage
    # Cost minimisation
    w_grid_cost: float = -2.4            # penalise step energy cost (scaled, non-saturating)
    w_energy_cost_direct: float = -0.40  # direct linear €/step penalty (raw)
    w_grid_import_energy: float = -0.16  # penalise imported kWh per step
    w_grid_export_energy: float = -0.04  # discourage excessive export cycling
    w_grid_exchange_abs: float = -0.02   # penalise absolute exchange magnitude
    w_cost_inefficiency_gap: float = -1.20  # penalise systematic cost gap vs baseline EMA
    cost_inefficiency_gap_tolerance: float = 0.03  # allow small positive gap (3%)
    cost_ema_alpha: float = 0.08         # EMA update factor for baseline cost proxy
    ref_grid_energy_kwh_step: float = 250.0  # scaling for grid exchange penalties
    w_co2: float = -0.7                  # penalise CO₂ from grid
    # H₂ management
    w_h2_soc_target: float = 0.5         # reward proximity to target SOC
    w_h2_soc_deviation: float = -0.4     # penalise extreme SOC levels
    w_h2_depletion_rate_penalty: float = -4.5  # penalise rapid storage depletion
    w_h2_soc_drop_penalty: float = -5.0        # penalise aggressive negative SOC ramps
    w_soc_smoothing: float = 1.1         # reward smooth SOC trajectory
    depletion_norm_kg_per_step: float = 4.0
    soc_smoothing_norm: float = 0.03
    soc_drop_threshold_per_step: float = 0.02
    soc_drop_norm: float = 0.05
    h2_soc_target: float = 0.50          # target SOC for H₂ reservoir
    # H₂ usage incentives (NEW)
    w_h2_fc_usage: float = 1.5           # reward fuel cell activation during deficit
    w_h2_el_usage: float = 0.45          # reduced EL incentive; should not dominate system efficiency
    w_h2_accumulation: float = 0.8       # reward H₂ accumulation
    w_h2_waste_penalty: float = -1.5     # penalise unused H₂ storage during deficit
    w_h2_cheap_charge: float = 0.9        # reward EL charging when price is cheap
    w_el_power_cost: float = -0.55       # penalise EL power consumption continuously
    w_h2_overproduction_penalty: float = -1.1  # penalise EL use when SOC is already high
    w_system_efficiency: float = 0.9     # reward useful energy delivered per total energy used
    fc_usage_norm: float = 500.0         # normalise FC power (kW)
    el_usage_norm: float = 500.0         # normalise EL power (kW)
    el_opportunity_norm_kw: float = 200.0
    el_soc_soft_min: float = 0.10
    el_soc_soft_max: float = 0.92
    h2_overproduction_soc_threshold: float = 0.70
    h2_overproduction_soc_span: float = 0.20
    h2_accum_norm: float = 2.0           # normalise H₂ accumulation (kg/step)
    cheap_price_threshold_eur_kwh: float = 0.16
    # Operational stability
    w_operational_stability: float = 0.8  # reward smoother control actuation
    action_delta_norm: float = 0.25
    # Anomaly / health
    w_anomaly: float = -1.0              # penalise degraded health
    # Reference cost for normalising the grid-cost penalty
    ref_cost_eur_step: float = 0.50      # normalise cost penalty (~0.5 €/step at peak)
    ref_co2_kg_step: float = 5.0         # normalise CO2 penalty
    cost_log_epsilon: float = 1e-6       # numerical stability for log-scaled cost
    reward_dt_reference_seconds: float = 600.0


class AdvancedH2GridEnv(gym.Env):
    """
    Extended Gymnasium environment for the H2 energy grid.

    Parameters
    ----------
    grid_config : dict
        Configuration dict for GridTwin (must include ``grid_connections``
        and optionally ``energy_cost`` keys for the new behaviour).
    weather_params : dict, optional
    cost_params : dict, optional
        Passed to EnergyCostModel.
    dt_seconds : float
        Simulation step [s]. Default 600 (10 min).
    episode_length : int
        Steps per episode. Default 144 (24 h at 10-min steps).
    start_time : datetime, optional
    reward_config : AdvancedRewardConfig, optional
    sensor_manager : SensorManager, optional
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        grid_config: Dict[str, Any],
        weather_params: Optional[Dict[str, Any]] = None,
        cost_params: Optional[Dict[str, Any]] = None,
        dt_seconds: float = 600.0,
        episode_length: int = 144,
        start_time: Optional[datetime] = None,
        reward_config: Optional[AdvancedRewardConfig] = None,
        sensor_manager: Optional[SensorManager] = None,
        history_window: int = 8,
        forecast_horizon: int = 3,
        forecast_step_multiplier: int = 1,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._grid_config = grid_config
        self._weather_params = weather_params or {}
        self._cost_params = cost_params or {}
        self._dt = dt_seconds
        self._ep_len = episode_length
        self._start_time = start_time or datetime(2024, 6, 15, 0, 0)
        self._reward_cfg = reward_config or AdvancedRewardConfig()
        self._sensor_mgr = sensor_manager
        self._history_window = max(1, int(history_window))
        self._forecast_horizon = max(0, int(forecast_horizon))
        self._forecast_step_mult = max(1, int(forecast_step_multiplier))
        self.render_mode = render_mode

        # ─── Extract rated capacities from grid config for forecast normalisation ──
        _cfg = grid_config.get("grid", grid_config)
        self._wind_rated_kw: float = sum(
            float(wt["params"]["rated_power_kw"])
            for wt in _cfg.get("wind_turbines", [])
        ) or 520.0
        self._pv_rated_kw: float = sum(
            float(pv["params"]["rated_power_kw"])
            for pv in _cfg.get("pv_arrays", [])
        ) or 480.0
        self._load_rated_kw: float = max(
            (float(ld["params"]["base_load_kw"]) for ld in _cfg.get("loads", [])),
            default=760.0,
        )

        # ─── Weather parameters for physics-based forecasts ───────────────────────
        _wp = weather_params or {}
        self._fcast_lat: float   = float(_wp.get("latitude_deg",   40.5))
        self._fcast_lon: float   = float(_wp.get("longitude_deg",  14.8))
        self._fcast_alt: float   = float(_wp.get("altitude_m",     50.0))
        self._fcast_wind_autocorr: float = float(_wp.get("autocorr_wind", 0.88))
        self._fcast_cloud_mean: float    = float(_wp.get("cloud_cover_mean", 0.35))
        self._fcast_cloud_autocorr: float = 0.90  # matches WeatherModel default
        self._fcast_wind_cf_mean: float   = 0.28  # long-run cap. factor for Weibull c≈4.8

        # ─── Cloud cover tracking (updated each step from WeatherModel) ──────────
        self._cloud_cover_now: float = self._fcast_cloud_mean

        self._history_feature_names: List[str] = [
            "wind_power_kw",
            "pv_power_kw",
            "load_kw",
            "h2_soc",
            "energy_price_norm",
            "electrolyzer_power_kw",
            "grid_connection_kw",
        ]
        self._n_hist_features = len(self._history_feature_names)
        self._n_forecast_features_per_step = N_FORECAST_FEATURES_PER_STEP
        self._forecast_feature_size = self._forecast_horizon * self._n_forecast_features_per_step
        n_obs = (
            N_BASE_OBS
            + self._history_window * self._n_hist_features
            + self._forecast_feature_size
        )

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=np.full(n_obs, -1.0, dtype=np.float32),
            high=np.full(n_obs, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(N_ACT, dtype=np.float32),
            high=np.ones(N_ACT, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (built on reset)
        self._twin: Optional[GridTwin] = None
        self._weather: Optional[WeatherModel] = None
        self._cost_model: Optional[EnergyCostModel] = None
        self._bus = EventBus()
        self._sm = StateManager()

        self._step_count: int = 0
        self._current_time: datetime = self._start_time
        self._prev_gs: Optional[GridState] = None
        self._last_action = np.zeros(N_ACT, dtype=np.float32)
        self._episode_reward: float = 0.0
        self._reward_log: List[Dict[str, float]] = []
        self._reward_cumulative: Dict[str, float] = defaultdict(float)
        self._history_buffer: deque[np.ndarray] = deque(maxlen=self._history_window)
        self._last_forecast_used: Dict[str, float] = {}
        self._ema_step_cost_eur: float = 0.0
        self._cumulative_step_cost_eur: float = 0.0
        self._cumulative_grid_import_kwh: float = 0.0
        self._cumulative_grid_export_kwh: float = 0.0

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

        # Random episode start date (different seasons during training)
        random_start = True if options is None else bool(options.get("random_start", True))
        if random_start:
            day_offset = int(np.random.randint(0, 365))
            self._current_time = self._start_time + timedelta(days=day_offset)
        else:
            self._current_time = self._start_time

        # Rebuild components
        self._bus = EventBus()
        self._sm = StateManager()
        self._weather = WeatherModel(**self._weather_params)
        self._cost_model = EnergyCostModel(self._cost_params)
        self._twin = GridTwin(
            self._grid_config,
            self._bus,
            self._sm,
            cost_model=self._cost_model,
        )
        self._twin.build()

        self._step_count = 0
        self._prev_gs = None
        self._last_action = np.zeros(N_ACT, dtype=np.float32)
        self._episode_reward = 0.0
        self._reward_log.clear()
        self._reward_cumulative.clear()
        self._history_buffer.clear()
        self._ema_step_cost_eur = 0.0
        self._cumulative_step_cost_eur = 0.0
        self._cumulative_grid_import_kwh = 0.0
        self._cumulative_grid_export_kwh = 0.0

        # Warm-up step
        weather = self._weather.step(self._current_time)
        _ = self._twin.step(self._dt, weather, {}, timestamp=self._current_time)

        obs = self._make_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        el_sp   = float(np.clip(action[0], 0.0, 1.0))
        fc_sp   = float(np.clip(action[1], 0.0, 1.0))
        dr_sp   = float(np.clip(action[2], 0.0, 1.0))
        grd_frc = float(np.clip(action[3], 0.0, 1.0)) if len(action) > 3 else 0.3

        self._current_time += timedelta(seconds=self._dt)

        # Read current price (state advancement is handled by GridTwin.step)
        current_price = (
            self._cost_model.get_buy_price(self._current_time)
            if self._cost_model is not None else 0.15
        )

        # Fair demand-response policy (aligned with rule-based controller conditions)
        renewable_kw_now = float(self._twin.grid_state.wind_power_kw + self._twin.grid_state.pv_power_kw)
        load_kw_now = float(self._twin.grid_state.load_kw)
        net_renewable_now = renewable_kw_now - load_kw_now
        dr_allowed = (not bool(self._twin.grid_state.grid_available)) or (
            current_price >= 0.20 and net_renewable_now < -20.0
        )
        dr_cap = min(0.20, max(0.0, -net_renewable_now) / (load_kw_now + 1e-9))
        dr_effective = min(dr_sp * 0.20, dr_cap) if dr_allowed else 0.0

        weather = self._weather.step(self._current_time)
        # Track cloud cover for physics-based PV forecasts
        self._cloud_cover_now = float(weather.get("cloud_cover", self._fcast_cloud_mean))

        # Build control actions for each component type
        control_actions: Dict[str, Any] = {}
        h2_avail = self._get_h2_available()

        for node_id, node in self._twin._nodes.items():
            mtype = type(node.model).__name__
            if mtype == "ElectrolyzerModel":
                rated = node.model._rated_kw
                control_actions[node_id] = {"power_setpoint_kw": el_sp * rated}
            elif mtype == "FuelCellModel":
                rated = node.model._rated_kw
                control_actions[node_id] = {
                    "power_setpoint_kw": fc_sp * rated,
                    "h2_available_kg": h2_avail,
                }
            elif mtype == "EnergyLoadModel":
                control_actions[node_id] = {"demand_response": dr_effective}
            elif mtype == "HydrogenTankModel":
                h2_in, h2_out = self._get_h2_tank_flows()
                control_actions[node_id] = {
                    "h2_charge_kg": h2_in,
                    "h2_discharge_kg": h2_out,
                }
            elif mtype == "GridConnectionModel":
                max_imp = node.model.max_import_kw
                control_actions[node_id] = {
                    "power_setpoint_kw": grd_frc * max_imp
                }

        # Sensor readings
        sensor_readings = None
        if self._sensor_mgr is not None:
            snap = self._sm.snapshot()
            sensor_readings = self._sensor_mgr.update(snap, timestamp=self._current_time)

        gs = self._twin.step(
            self._dt, weather, control_actions,
            sensor_readings=sensor_readings,
            timestamp=self._current_time,
        )

        reward, components = self._compute_advanced_reward(
            gs,
            gs.energy_price_eur_kwh,
            action,
            dr_effective,
        )
        self._reward_log.append(components)
        for key, value in components.items():
            if key.startswith("cumulative_"):
                continue
            if isinstance(value, (int, float, np.floating)):
                self._reward_cumulative[key] += float(value)
        self._episode_reward += reward
        self._prev_gs = gs
        self._last_action = np.array(action, dtype=np.float32)
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self._ep_len

        obs = self._make_obs()
        info = {
            "grid_state": gs,
            "reward_components": components,
            "reward_components_cumulative": dict(self._reward_cumulative),
            "economic_cumulative": {
                "step_cost_eur_raw": float(self._cumulative_step_cost_eur),
                "grid_import_kwh": float(self._cumulative_grid_import_kwh),
                "grid_export_kwh": float(self._cumulative_grid_export_kwh),
                "cost_baseline_ema_eur": float(self._ema_step_cost_eur),
            },
            "episode_reward": self._episode_reward,
            "step": self._step_count,
            "energy_price": current_price,
        }
        # Record the forecast features used at this step for diagnostics
        info["forecast_features"] = self._last_forecast_used.copy()
        info["forecast_feature_names"] = FORECAST_FEATURE_NAMES
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        gs = self._twin.grid_state if self._twin else None
        if gs is None:
            return None
        lines = [
            f"[{gs.timestamp.strftime('%H:%M')}]",
            f"  Wind {gs.wind_power_kw:6.1f} kW  PV {gs.pv_power_kw:6.1f} kW",
            f"  EL   {gs.electrolyzer_power_kw:6.1f} kW  FC {gs.fuel_cell_power_kw:6.1f} kW",
            f"  Grid {gs.grid_connection_kw:+6.1f} kW  Price {gs.energy_price_eur_kwh:.3f} €/kWh",
            f"  H2 SOC {gs.h2_soc:.2f}  RF {gs.renewable_fraction:.1%}",
        ]
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        gs = self._twin.grid_state
        if gs is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        ts = self._current_time
        price = float(gs.energy_price_eur_kwh)
        hour_angle = 2 * np.pi * ts.hour / 24.0
        raw = np.array([
            gs.wind_power_kw,
            gs.pv_power_kw,
            gs.electrolyzer_power_kw,
            gs.fuel_cell_power_kw,
            gs.load_kw,
            gs.grid_exchange_kw,
            gs.grid_connection_kw,
            gs.h2_production_kg_h,
            gs.h2_consumption_kg_h,
            gs.h2_storage_kg,
            gs.h2_soc,
            gs.h2_pressure_bar,
            gs.renewable_fraction,
            gs.grid_self_sufficiency,
            gs.overall_health,
            price / 0.50,
            1.0 if gs.grid_available else 0.0,
            float(np.sin(hour_angle)),
            float(np.cos(hour_angle)),
        ], dtype=np.float32)

        base_obs = 2.0 * (raw - OBS_MIN) / (OBS_MAX - OBS_MIN + 1e-9) - 1.0
        base_obs = np.clip(base_obs, -1.0, 1.0).astype(np.float32)

        hist_vec = self._extract_history_features(gs, price)
        self._history_buffer.append(hist_vec)
        padded_history = self._padded_history_features()

        forecast_features = self._build_forecast_context(ts, gs, price)
        obs = np.concatenate([base_obs, padded_history, forecast_features], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _extract_history_features(self, gs: GridState, price: float) -> np.ndarray:
        values = np.array([
            gs.wind_power_kw,
            gs.pv_power_kw,
            gs.load_kw,
            gs.h2_soc,
            price / 0.50,
            gs.electrolyzer_power_kw,
            gs.grid_connection_kw,
        ], dtype=np.float32)
        max_vals = np.array([2000.0, 2000.0, 2000.0, 1.0, 1.0, 1000.0, 1000.0], dtype=np.float32)
        min_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -200.0], dtype=np.float32)
        scaled = 2.0 * (values - min_vals) / (max_vals - min_vals + 1e-9) - 1.0
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    def _padded_history_features(self) -> np.ndarray:
        if not self._history_buffer:
            return np.zeros(self._history_window * self._n_hist_features, dtype=np.float32)

        history = list(self._history_buffer)
        if len(history) < self._history_window:
            pad = [np.zeros(self._n_hist_features, dtype=np.float32)] * (self._history_window - len(history))
            history = pad + history

        return np.concatenate(history[-self._history_window:], dtype=np.float32)

    def _build_forecast_context(self, ts: datetime, gs: GridState, price: float) -> np.ndarray:
        if self._forecast_horizon <= 0:
            self._last_forecast_used = {}
            return np.zeros(0, dtype=np.float32)

        fvec = build_forecast_features(
            ts=ts,
            current_price=price,
            current_wind_kw=float(gs.wind_power_kw),
            current_pv_kw=float(gs.pv_power_kw),
            current_load_kw=float(gs.load_kw),
            n_steps=self._forecast_horizon,
            dt_seconds=self._dt,
            forecast_step_mult=self._forecast_step_mult,
            wind_rated_kw=self._wind_rated_kw,
            pv_rated_kw=self._pv_rated_kw,
            load_rated_kw=self._load_rated_kw,
            peek_price_fn=self._peek_buy_price if self._cost_model is not None else None,
            wind_autocorr=self._fcast_wind_autocorr,
            wind_capacity_factor_mean=self._fcast_wind_cf_mean,
            cloud_cover_now=self._cloud_cover_now,
            cloud_mean=self._fcast_cloud_mean,
            cloud_autocorr=self._fcast_cloud_autocorr,
            lat_deg=self._fcast_lat,
            lon_deg=self._fcast_lon,
            alt_m=self._fcast_alt,
        )

        # Store first-step forecast for logging (8 features of horizon-1 step only)
        self._last_forecast_used = {
            FORECAST_FEATURE_NAMES[j]: float(fvec[j])
            for j in range(min(N_FORECAST_FEATURES_PER_STEP, len(fvec)))
        }
        return fvec

    def _peek_buy_price(self, ts: datetime) -> float:
        """Return buy price at ts without mutating cost-model or global RNG state."""
        if self._cost_model is None:
            return 0.15

        cost_model = self._cost_model
        rng_state = np.random.get_state()
        model_state = (
            cost_model._current_day,
            cost_model._daily_factor,
            cost_model._spike_remaining,
            cost_model._spike_active,
        )
        try:
            return float(cost_model.get_buy_price(ts))
        finally:
            (
                cost_model._current_day,
                cost_model._daily_factor,
                cost_model._spike_remaining,
                cost_model._spike_active,
            ) = model_state
            np.random.set_state(rng_state)

    def _get_h2_available(self) -> float:
        if self._twin is None:
            return 0.0
        from ..models import HydrogenTankModel
        total_available = 0.0
        for node in self._twin._nodes.values():
            if isinstance(node.model, HydrogenTankModel):
                if node.state:
                    fused = node.state.fused_values
                    soc = float(fused.get("soc", 0.0))
                    max_capacity = float(fused.get("max_capacity_kg", 0.0))
                    total_available += max(0.0, soc * max_capacity)
        return total_available

    def _get_h2_tank_flows(self) -> Tuple[float, float]:
        gs = self._twin.grid_state if self._twin else None
        if gs is None:
            return 0.0, 0.0
        dt_h = self._dt / 3600.0
        return gs.h2_production_kg_h * dt_h, gs.h2_consumption_kg_h * dt_h

    def _compute_advanced_reward(
        self,
        gs: GridState,
        current_price: float,
        action: np.ndarray,
        effective_demand_response: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self._reward_cfg

        # Self-sufficiency
        self_sufficiency = float(np.clip(gs.grid_self_sufficiency, 0.0, 1.0))
        renewable_fraction = float(np.clip(gs.renewable_fraction, 0.0, 1.0))
        r_ss = cfg.w_self_sufficiency * self_sufficiency

        dt_scale = max(1e-9, self._dt / max(1e-9, cfg.reward_dt_reference_seconds))

        # Renewable fraction
        r_rf = cfg.w_renewable_fraction * renewable_fraction

        # Demand met (using real import from grid exchange, not setpoint)
        renewable_kw = float(gs.wind_power_kw + gs.pv_power_kw)
        fuel_cell_kw = float(gs.fuel_cell_power_kw)
        grid_import_kw = float(max(0.0, gs.grid_exchange_kw))
        load_kw = float(gs.load_kw)
        real_supply_kw = renewable_kw + fuel_cell_kw + grid_import_kw
        demand_met = min(1.0, real_supply_kw / (load_kw + 1e-9))
        r_demand = cfg.w_demand_met * demand_met
        unmet_ratio = max(0.0, 1.0 - demand_met)
        r_unmet_pen = cfg.w_unmet_demand_penalty * (unmet_ratio ** 2)
        excess_unmet = max(0.0, unmet_ratio - cfg.unmet_hard_threshold)
        r_unmet_hard = cfg.w_unmet_hard_penalty * (excess_unmet ** cfg.unmet_hard_power)
        r_supply_priority = cfg.w_supply_priority * unmet_ratio

        # ─────────────────────────────────────────────────────────────────
        # Economic terms (non-saturating, continuously sensitive)
        # ─────────────────────────────────────────────────────────────────
        step_cost_eur = float(gs.step_cost_eur)
        grid_exchange_kw = float(gs.grid_exchange_kw)
        grid_import_kw = float(max(0.0, grid_exchange_kw))
        grid_export_kw = float(max(0.0, -grid_exchange_kw))
        dt_h = max(1e-9, self._dt / 3600.0)
        grid_import_kwh_step = grid_import_kw * dt_h
        grid_export_kwh_step = grid_export_kw * dt_h
        grid_exchange_abs_kwh_step = abs(grid_exchange_kw) * dt_h

        # Non-saturating scaled cost (log1p keeps ordering and large-diff sensitivity)
        cost_scaled = np.log1p(max(0.0, step_cost_eur) / (cfg.ref_cost_eur_step * dt_scale + cfg.cost_log_epsilon))
        r_cost = cfg.w_grid_cost * float(cost_scaled)

        # Direct marginal economic penalties (raw units)
        r_cost_direct = cfg.w_energy_cost_direct * step_cost_eur
        r_grid_import = cfg.w_grid_import_energy * (grid_import_kwh_step / max(1e-9, cfg.ref_grid_energy_kwh_step * dt_h))
        r_grid_export = cfg.w_grid_export_energy * (grid_export_kwh_step / max(1e-9, cfg.ref_grid_energy_kwh_step * dt_h))
        r_grid_exchange_abs = cfg.w_grid_exchange_abs * (grid_exchange_abs_kwh_step / max(1e-9, cfg.ref_grid_energy_kwh_step * dt_h))

        # Economic inefficiency vs moving baseline (proxy baseline, no oracle leakage)
        if self._step_count <= 0 and self._ema_step_cost_eur <= 0.0:
            cost_baseline_ema = step_cost_eur
        else:
            cost_baseline_ema = self._ema_step_cost_eur
        allowed_cost = cost_baseline_ema * (1.0 + max(0.0, cfg.cost_inefficiency_gap_tolerance))
        cost_ineff_gap_eur = max(0.0, step_cost_eur - allowed_cost)
        cost_ineff_gap_scaled = cost_ineff_gap_eur / (cfg.ref_cost_eur_step * dt_scale + cfg.cost_log_epsilon)
        r_cost_inefficiency = cfg.w_cost_inefficiency_gap * cost_ineff_gap_scaled

        # Update EMA baseline after evaluating current gap
        ema_alpha = float(np.clip(cfg.cost_ema_alpha, 1e-4, 1.0))
        self._ema_step_cost_eur = (1.0 - ema_alpha) * cost_baseline_ema + ema_alpha * step_cost_eur

        # CO₂ penalty (log-scaled, non-saturating)
        co2_scaled = np.log1p(max(0.0, float(gs.step_co2_kg)) / (cfg.ref_co2_kg_step * dt_scale + 1e-9))
        r_co2 = cfg.w_co2 * float(co2_scaled)

        # H₂ SOC target
        soc_err = abs(gs.h2_soc - cfg.h2_soc_target)
        soc_dev_norm = float(np.clip(soc_err / 0.5, 0.0, 1.0))
        r_soc_dev = cfg.w_h2_soc_deviation * soc_dev_norm ** 2
        r_soc_tgt = cfg.w_h2_soc_target * max(0.0, 1.0 - soc_err * 2)

        delta_h2_kg = 0.0
        delta_soc = 0.0
        if self._prev_gs is not None:
            delta_h2_kg = gs.h2_storage_kg - self._prev_gs.h2_storage_kg
            delta_soc = gs.h2_soc - self._prev_gs.h2_soc

        depletion_norm = max(1e-6, cfg.depletion_norm_kg_per_step * dt_scale)
        rapid_depletion = max(0.0, -delta_h2_kg) / depletion_norm
        r_depletion = cfg.w_h2_depletion_rate_penalty * min(1.0, rapid_depletion ** 2)

        soc_drop = max(0.0, -delta_soc - cfg.soc_drop_threshold_per_step)
        soc_drop_norm = min(1.0, soc_drop / max(1e-9, cfg.soc_drop_norm))
        r_soc_drop_penalty = cfg.w_h2_soc_drop_penalty * (soc_drop_norm ** 2)

        smooth_norm = max(1e-6, cfg.soc_smoothing_norm * dt_scale)
        r_soc_smoothing = cfg.w_soc_smoothing * max(0.0, 1.0 - abs(delta_soc) / smooth_norm)

        action_delta = np.linalg.norm(np.asarray(action, dtype=np.float32) - self._last_action, ord=2)
        r_stability = cfg.w_operational_stability * max(
            0.0,
            1.0 - action_delta / max(1e-6, cfg.action_delta_norm),
        )

        # Anomaly / health
        r_anomaly = cfg.w_anomaly * (1.0 - gs.overall_health)

        # Demand-response usage penalty (keeps comparisons fair and avoids "easy" load shedding)
        r_dr_penalty = cfg.w_demand_response_penalty * float(np.clip(effective_demand_response, 0.0, 1.0))

        # === H₂ USAGE INCENTIVES (NEW) ===
        # Renewable excess / deficit calculation (before grid import)
        renewable_supply = renewable_kw
        total_demand = load_kw
        pre_grid_deficit = max(0.0, total_demand - renewable_supply)
        excess_renewable = max(0.0, renewable_supply - total_demand)

        # Fuel cell usage reward: incentivise FC when there's energy deficit
        el_power = gs.electrolyzer_power_kw
        fc_power = gs.fuel_cell_power_kw

        r_fc_usage = 0.0
        if pre_grid_deficit > 1e-6 and gs.h2_soc > 0.15:
            fc_cover = min(1.0, fc_power / (pre_grid_deficit + 1e-9))
            r_fc_usage = cfg.w_h2_fc_usage * fc_cover

        # Electrolyzer usage reward: continuous/proportional and active on wider SOC range
        el_ratio = min(1.0, el_power / (cfg.el_usage_norm + 1e-9))
        soc_span = max(1e-9, cfg.el_soc_soft_max - cfg.el_soc_soft_min)
        soc_room = np.clip((cfg.el_soc_soft_max - gs.h2_soc) / soc_span, 0.0, 1.0)
        cheap_drive = np.clip((cfg.cheap_price_threshold_eur_kwh - current_price) / max(1e-9, cfg.cheap_price_threshold_eur_kwh), 0.0, 1.0)
        excess_drive = np.clip(excess_renewable / max(1e-9, cfg.el_opportunity_norm_kw), 0.0, 1.0)
        low_soc_drive = np.clip((cfg.h2_soc_target - gs.h2_soc) / max(1e-9, cfg.h2_soc_target - cfg.el_soc_soft_min), 0.0, 1.0)
        el_drive = max(float(cheap_drive), float(excess_drive)) * float(low_soc_drive)
        r_el_usage = cfg.w_h2_el_usage * el_ratio * float(soc_room) * el_drive

        # Continuous EL energy-use penalty to reflect real energy consumption
        r_el_cost = cfg.w_el_power_cost * el_ratio

        # Penalise H2 overproduction when SOC is already high
        over_soc = max(0.0, gs.h2_soc - cfg.h2_overproduction_soc_threshold)
        over_soc_ratio = min(1.0, over_soc / max(1e-9, cfg.h2_overproduction_soc_span))
        r_overproduction = cfg.w_h2_overproduction_penalty * el_ratio * over_soc_ratio

        # Strategic cheap-price charging (important for long-horizon RL)
        r_cheap_charge = 0.0
        if current_price <= cfg.cheap_price_threshold_eur_kwh and gs.h2_soc < cfg.h2_soc_target:
            el_ratio = min(1.0, el_power / (cfg.el_usage_norm + 1e-9))
            r_cheap_charge = cfg.w_h2_cheap_charge * el_ratio

        # H₂ accumulation reward: encourage charging during renewable excess
        r_h2_accumulation = 0.0
        if delta_h2_kg > 0.0 and (excess_renewable > 50.0 or current_price <= cfg.cheap_price_threshold_eur_kwh) and gs.h2_soc < 0.85:
            accum_norm = min(1.0, delta_h2_kg / (cfg.h2_accum_norm * dt_scale + 1e-9))
            r_h2_accumulation = cfg.w_h2_accumulation * accum_norm

        # H₂ waste penalty: penalise not using FuelCell when deficit exists & H₂ available
        r_h2_waste = 0.0
        if pre_grid_deficit > 20.0 and gs.h2_soc > 0.20 and fc_power < 10.0:
            r_h2_waste = cfg.w_h2_waste_penalty

        # Reward global system efficiency: useful demand served over total energy actively used
        useful_supply_kw = min(load_kw, real_supply_kw)
        total_energy_used_kw = renewable_kw + grid_import_kw + max(0.0, fc_power) + max(0.0, el_power)
        efficiency = float(np.clip(useful_supply_kw / (total_energy_used_kw + 1e-9), 0.0, 1.0))
        r_efficiency = cfg.w_system_efficiency * efficiency

        total = (
            r_ss
            + r_rf
            + r_demand
            + r_unmet_pen
            + r_unmet_hard
            + r_supply_priority
            + r_cost
            + r_cost_direct
            + r_grid_import
            + r_grid_export
            + r_grid_exchange_abs
            + r_cost_inefficiency
            + r_co2
            + r_soc_dev
            + r_soc_tgt
            + r_depletion
            + r_soc_drop_penalty
            + r_soc_smoothing
            + r_stability
            + r_anomaly
            + r_dr_penalty
            + r_fc_usage           # NEW
            + r_el_usage           # NEW
            + r_el_cost
            + r_overproduction
            + r_cheap_charge
            + r_h2_accumulation    # NEW
            + r_h2_waste           # NEW
            + r_efficiency
        )

        economic_penalty_total = (
            r_cost
            + r_cost_direct
            + r_grid_import
            + r_grid_export
            + r_grid_exchange_abs
            + r_cost_inefficiency
        )
        h2_penalty_total = (
            r_el_cost
            + r_overproduction
            + r_h2_waste
            + r_depletion
            + r_soc_drop_penalty
        )

        self._cumulative_step_cost_eur += step_cost_eur
        self._cumulative_grid_import_kwh += grid_import_kwh_step
        self._cumulative_grid_export_kwh += grid_export_kwh_step

        components = {
            "self_sufficiency": r_ss,
            "renewable_fraction": r_rf,
            "demand_met": r_demand,
            "unmet_demand_penalty": r_unmet_pen,
            "unmet_hard_penalty": r_unmet_hard,
            "supply_priority_penalty": r_supply_priority,
            "grid_cost": r_cost,
            "energy_cost_direct": r_cost_direct,
            "grid_import_energy_penalty": r_grid_import,
            "grid_export_energy_penalty": r_grid_export,
            "grid_exchange_abs_penalty": r_grid_exchange_abs,
            "economic_inefficiency_penalty": r_cost_inefficiency,
            "economic_penalty_total": economic_penalty_total,
            "co2": r_co2,
            "soc_deviation": r_soc_dev,
            "soc_target": r_soc_tgt,
            "rapid_depletion_penalty": r_depletion,
            "soc_drop_penalty": r_soc_drop_penalty,
            "soc_smoothing": r_soc_smoothing,
            "operational_stability": r_stability,
            "anomaly": r_anomaly,
            "demand_response_penalty": r_dr_penalty,
            "h2_fc_usage": r_fc_usage,           # NEW
            "h2_el_usage": r_el_usage,          # NEW
            "el_power_cost": r_el_cost,
            "h2_overproduction_penalty": r_overproduction,
            "h2_cheap_charge": r_cheap_charge,
            "h2_accumulation": r_h2_accumulation,  # NEW
            "h2_waste_penalty": r_h2_waste,     # NEW
            "h2_penalty_total": h2_penalty_total,
            "system_efficiency_reward": r_efficiency,
            "real_supply_kw": real_supply_kw,
            "real_grid_import_kw": grid_import_kw,
            "real_grid_export_kw": grid_export_kw,
            "grid_exchange_kw": grid_exchange_kw,
            "load_kw": load_kw,
            "unmet_ratio": unmet_ratio,
            "cost_scaled": float(cost_scaled),
            "co2_scaled": float(co2_scaled),
            "step_cost_eur_raw": step_cost_eur,
            "grid_import_kwh_step": grid_import_kwh_step,
            "grid_export_kwh_step": grid_export_kwh_step,
            "grid_exchange_abs_kwh_step": grid_exchange_abs_kwh_step,
            "cost_baseline_ema_eur": float(cost_baseline_ema),
            "cost_inefficiency_gap_eur": float(cost_ineff_gap_eur),
            "cost_inefficiency_gap_scaled": float(cost_ineff_gap_scaled),
            "cumulative_step_cost_eur_raw": float(self._cumulative_step_cost_eur),
            "cumulative_grid_import_kwh": float(self._cumulative_grid_import_kwh),
            "cumulative_grid_export_kwh": float(self._cumulative_grid_export_kwh),
            "soc_error": soc_err,
            "soc_drop": soc_drop,
            "delta_soc": delta_soc,
            "delta_h2_kg": delta_h2_kg,
            "electrolyzer_power_kw": el_power,
            "fuel_cell_power_kw": fc_power,
            "overproduction_soc_ratio": over_soc_ratio,
            "system_efficiency": efficiency,
            "useful_supply_kw": useful_supply_kw,
            "total_energy_used_kw": total_energy_used_kw,
            "action_el": float(np.clip(action[0], 0.0, 1.0)),
            "action_fc": float(np.clip(action[1], 0.0, 1.0)),
            "action_dr": float(np.clip(action[2], 0.0, 1.0)),
            "action_grid": float(np.clip(action[3], 0.0, 1.0)) if len(action) > 3 else 0.3,
            "total": total,
        }
        return total, components
