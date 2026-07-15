"""
RL Controller Adapter
=====================
Wraps a trained Stable-Baselines3 model so it can be used as a drop-in
replacement for the ClassicalController inside the SimulationEngine.

The adapter translates:
  RL obs vector (normalised)  ←  GridState
  RL action vector (0-1)      →  component control dicts

It mirrors the observation normalisation used in AdvancedH2GridEnv so
that a model trained on that environment can be evaluated directly on
the SimulationEngine.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from ..digital_twin.grid_twin import GridState
from ..models.energy_cost import EnergyCostModel
from ..rl.forecast_utils import (
    N_FORECAST_FEATURES_PER_STEP as _N_FORECAST_FEATS,
    FORECAST_FEATURE_NAMES,
    build_forecast_features,
    _solar_elevation_deg,
    _ghi_clear_sky,
)

logger = logging.getLogger(__name__)

# Must match AdvancedH2GridEnv.OBS_MAX / OBS_MIN exactly
_OBS_MAX = np.array([
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
    1.0,      # energy_price_eur_kwh (normalised by 0.50)
    1.0,      # grid_available (0/1)
    1.0,      # hour_sin
    1.0,      # hour_cos
], dtype=np.float32)

_OBS_MIN = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, -1000.0, -200.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, -1.0, -1.0,
], dtype=np.float32)

_OBS_RANGE = _OBS_MAX - _OBS_MIN
_HIST_FEATURE_DIM = 7
_N_FORECAST_FEATURES_PER_STEP = _N_FORECAST_FEATS  # must equal env value


class RLController:
    """
    Wraps a trained SB3 model as a controller.

    Parameters
    ----------
    model_path : str | Path
        Path to a saved SB3 `.zip` model.
    grid_config : dict
        Same config used for GridTwin (to read rated powers).
    cost_model : EnergyCostModel, optional
    name : str
    """

    def __init__(
        self,
        model_path,
        grid_config: Dict[str, Any],
        cost_model: Optional[EnergyCostModel] = None,
        dt_seconds: float = 600.0,
        history_window: int = 8,
        forecast_horizon: int = 3,
        forecast_step_multiplier: int = 1,
        enable_unmet_safety: bool = True,
        unmet_safety_margin_kw: float = 5.0,
        enable_h2_first_safety: bool = True,
        h2_soc_min_for_fc_safety: float = 0.22,
        enable_el_charge_safety: bool = True,
        el_charge_cheap_price_threshold: float = 0.16,
        el_charge_soc_target: float = 0.50,
        el_charge_min_fraction: float = 0.12,
        el_charge_renewable_excess_kw: float = 50.0,
        demand_response_max_fraction: float = 0.20,
        demand_response_price_high_threshold: float = 0.20,
        demand_response_deficit_threshold_kw: float = -20.0,
        name: str = "RLController",
    ) -> None:
        self.name = name
        self._cost_model = cost_model
        self._dt_seconds = float(dt_seconds)
        self._history_window = max(1, int(history_window))
        self._forecast_horizon = max(0, int(forecast_horizon))
        self._forecast_step_mult = max(1, int(forecast_step_multiplier))
        self._enable_unmet_safety = bool(enable_unmet_safety)
        self._unmet_safety_margin_kw = max(0.0, float(unmet_safety_margin_kw))
        self._enable_h2_first_safety = bool(enable_h2_first_safety)
        self._h2_soc_min_for_fc_safety = float(np.clip(h2_soc_min_for_fc_safety, 0.0, 1.0))
        self._enable_el_charge_safety = bool(enable_el_charge_safety)
        self._el_charge_cheap_price_threshold = float(el_charge_cheap_price_threshold)
        self._el_charge_soc_target = float(np.clip(el_charge_soc_target, 0.0, 1.0))
        self._el_charge_min_fraction = float(np.clip(el_charge_min_fraction, 0.0, 1.0))
        self._el_charge_renewable_excess_kw = float(el_charge_renewable_excess_kw)
        self._dr_max_fraction = float(np.clip(demand_response_max_fraction, 0.0, 1.0))
        self._dr_price_high_threshold = float(demand_response_price_high_threshold)
        self._dr_deficit_threshold_kw = float(demand_response_deficit_threshold_kw)
        self._history_buffer: deque[np.ndarray] = deque(maxlen=self._history_window)
        self._lstm_state = None
        self._episode_start = np.array([True], dtype=bool)
        self._last_timestamp: Optional[datetime] = None
        self._last_forecast_vector = np.zeros(0, dtype=np.float32)
        self._last_forecast_first_step: Dict[str, float] = {}
        self._last_predicted_action = np.zeros(4, dtype=np.float32)

        # Extract rated powers from config
        self._el_rated: Dict[str, float] = {}
        self._fc_rated: Dict[str, float] = {}
        self._gc_max_import: Dict[str, float] = {}
        self._load_ids: list[str] = []
        cfg = grid_config.get("grid", grid_config)
        for ec in cfg.get("electrolyzers", []):
            self._el_rated[ec["id"]] = float(ec["params"]["rated_power_kw"])
        for fc in cfg.get("fuel_cells", []):
            self._fc_rated[fc["id"]] = float(fc["params"]["rated_power_kw"])
        for gc in cfg.get("grid_connections", []):
            self._gc_max_import[gc["id"]] = float(
                gc["params"].get("max_import_kw", 800.0)
            )
        for ld in cfg.get("loads", []):
            self._load_ids.append(str(ld["id"]))

        # ─── Rated capacities for forecast normalisation ───────────────────────
        self._wind_rated_kw: float = sum(
            float(wt["params"]["rated_power_kw"])
            for wt in cfg.get("wind_turbines", [])
        ) or 520.0
        self._pv_rated_kw: float = sum(
            float(pv["params"]["rated_power_kw"])
            for pv in cfg.get("pv_arrays", [])
        ) or 480.0
        self._load_rated_kw_ref: float = max(
            (float(ld["params"]["base_load_kw"]) for ld in cfg.get("loads", [])),
            default=760.0,
        )

        # ─── Weather params for physics-based PV/wind forecast ─────────────────
        _wp = grid_config.get("weather", {})
        self._fcast_lat: float   = float(_wp.get("latitude_deg",   40.5))
        self._fcast_lon: float   = float(_wp.get("longitude_deg",  14.8))
        self._fcast_alt: float   = float(_wp.get("altitude_m",     50.0))
        self._fcast_wind_autocorr: float  = float(_wp.get("autocorr_wind",    0.88))
        self._fcast_cloud_mean: float     = float(_wp.get("cloud_cover_mean", 0.35))
        self._fcast_cloud_autocorr: float = 0.90
        self._fcast_wind_cf_mean: float   = 0.28

        # Load SB3 model
        self._model = None
        self._is_recurrent = False
        try:
            from stable_baselines3 import PPO, SAC, TD3
            for cls in (PPO, SAC, TD3):
                try:
                    self._model = cls.load(model_path)
                    logger.info("Loaded RL model (%s) from %s", cls.__name__, model_path)
                    break
                except Exception:
                    continue

            if self._model is None:
                try:
                    from sb3_contrib import RecurrentPPO
                    self._model = RecurrentPPO.load(model_path)
                    self._is_recurrent = True
                    logger.info("Loaded RL model (%s) from %s", "RecurrentPPO", model_path)
                except Exception:
                    pass

            if self._model is None:
                raise RuntimeError(f"Could not load model from {model_path}")
        except ImportError:
            raise ImportError("stable-baselines3 (and sb3-contrib for recurrent models) is required for RLController.")

        self._obs_dim = int(self._model.observation_space.shape[0])

    # ------------------------------------------------------------------
    # Main interface (mirrors ClassicalController)
    # ------------------------------------------------------------------

    def compute_actions(
        self,
        grid_state: GridState,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Predict actions from the RL model and convert to control dicts."""
        ts = timestamp or grid_state.timestamp

        if self._last_timestamp is not None and ts <= self._last_timestamp:
            self.reset()
        self._last_timestamp = ts

        obs = self._make_obs(grid_state, ts)
        if self._is_recurrent:
            action, self._lstm_state = self._model.predict(
                obs,
                state=self._lstm_state,
                episode_start=self._episode_start,
                deterministic=True,
            )
            self._episode_start[:] = False
        else:
            action, _ = self._model.predict(obs, deterministic=True)
        self._last_predicted_action = np.asarray(action, dtype=np.float32).reshape(-1)
        return self._decode_action(action, grid_state)

    def reset(self) -> None:
        self._history_buffer.clear()
        self._lstm_state = None
        self._episode_start[:] = True
        self._last_timestamp = None
        self._last_forecast_vector = np.zeros(0, dtype=np.float32)
        self._last_forecast_first_step = {}
        self._last_predicted_action = np.zeros(4, dtype=np.float32)

    def get_last_diagnostics(self) -> Dict[str, float]:
        """Return last forecast/action diagnostics used by the RL policy."""
        diag: Dict[str, float] = {}
        for key, value in self._last_forecast_first_step.items():
            diag[f"rl_used_{key}_t1"] = float(value)
        if self._last_predicted_action.size >= 4:
            diag["rl_action_el"] = float(np.clip(self._last_predicted_action[0], 0.0, 1.0))
            diag["rl_action_fc"] = float(np.clip(self._last_predicted_action[1], 0.0, 1.0))
            diag["rl_action_dr"] = float(np.clip(self._last_predicted_action[2], 0.0, 1.0))
            diag["rl_action_grid"] = float(np.clip(self._last_predicted_action[3], 0.0, 1.0))
        return diag

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self, gs: GridState, ts: datetime) -> np.ndarray:
        """Build the normalised observation array from a GridState."""
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
            price / 0.50,           # normalise price to [0-1] roughly
            1.0 if gs.grid_available else 0.0,
            float(np.sin(hour_angle)),
            float(np.cos(hour_angle)),
        ], dtype=np.float32)

        base_obs = 2.0 * (raw - _OBS_MIN) / (_OBS_RANGE + 1e-9) - 1.0
        base_obs = np.clip(base_obs, -1.0, 1.0).astype(np.float32)

        hist_features = self._extract_history_features(gs, price)
        self._history_buffer.append(hist_features)
        padded_hist = self._padded_history_features()
        forecast_ctx = self._build_forecast_context(ts, gs, price)

        merged = np.concatenate([base_obs, padded_hist, forecast_ctx], dtype=np.float32)

        if merged.shape[0] < self._obs_dim:
            pad = np.zeros(self._obs_dim - merged.shape[0], dtype=np.float32)
            merged = np.concatenate([merged, pad], dtype=np.float32)
        elif merged.shape[0] > self._obs_dim:
            merged = merged[:self._obs_dim]

        return np.clip(merged, -1.0, 1.0).astype(np.float32)

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
        history = list(self._history_buffer)
        if len(history) < self._history_window:
            pad = [np.zeros(_HIST_FEATURE_DIM, dtype=np.float32)] * (self._history_window - len(history))
            history = pad + history
        return np.concatenate(history[-self._history_window:], dtype=np.float32)

    def _build_forecast_context(self, ts: datetime, gs: GridState, price: float) -> np.ndarray:
        if self._forecast_horizon <= 0:
            return np.zeros(0, dtype=np.float32)

        # Estimate current cloud cover from PV / clear-sky ratio at inference.
        # When solar elevation is low / zero, we fall back to climatological mean.
        elev_now = _solar_elevation_deg(ts, self._fcast_lat, self._fcast_lon)
        ghi_now  = _ghi_clear_sky(elev_now, self._fcast_alt)
        if ghi_now > 5.0:
            pv_cf = float(np.clip(gs.pv_power_kw / max(1.0, self._pv_rated_kw), 0.0, 1.0))
            # Rough inversion: P_pv ≈ P_rated × (1 - 0.75 × cloud^3.4) × ghi_ratio
            # → cloud ≈ (1 - pv_cf)^(1/3.4)  as a first approximation
            cloud_inferred = float(np.clip((1.0 - pv_cf) ** (1.0 / 3.4), 0.0, 1.0))
        else:
            cloud_inferred = self._fcast_cloud_mean

        fvec = build_forecast_features(
            ts=ts,
            current_price=price,
            current_wind_kw=float(gs.wind_power_kw),
            current_pv_kw=float(gs.pv_power_kw),
            current_load_kw=float(gs.load_kw),
            n_steps=self._forecast_horizon,
            dt_seconds=self._dt_seconds,
            forecast_step_mult=self._forecast_step_mult,
            wind_rated_kw=self._wind_rated_kw,
            pv_rated_kw=self._pv_rated_kw,
            load_rated_kw=self._load_rated_kw_ref,
            peek_price_fn=self._peek_buy_price if self._cost_model is not None else None,
            wind_autocorr=self._fcast_wind_autocorr,
            wind_capacity_factor_mean=self._fcast_wind_cf_mean,
            cloud_cover_now=cloud_inferred,
            cloud_mean=self._fcast_cloud_mean,
            cloud_autocorr=self._fcast_cloud_autocorr,
            lat_deg=self._fcast_lat,
            lon_deg=self._fcast_lon,
            alt_m=self._fcast_alt,
        )
        self._last_forecast_vector = fvec.copy()
        self._last_forecast_first_step = {
            FORECAST_FEATURE_NAMES[j]: float(fvec[j])
            for j in range(min(_N_FORECAST_FEATURES_PER_STEP, len(fvec)))
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

    def _decode_action(self, action: np.ndarray, gs: GridState) -> Dict[str, Dict[str, Any]]:
        """Convert 4-D RL action vector to component control dicts."""
        el_sp  = float(np.clip(action[0], 0.0, 1.0))
        fc_sp  = float(np.clip(action[1], 0.0, 1.0))
        dr_sp  = float(np.clip(action[2], 0.0, 1.0))
        grid_fraction = float(np.clip(action[3], 0.0, 1.0)) if len(action) > 3 else 0.3

        actions: Dict[str, Dict[str, Any]] = {}

        renewable_kw = float(gs.wind_power_kw + gs.pv_power_kw)
        total_load_kw = float(gs.load_kw)
        excess_renewable_kw = max(0.0, renewable_kw - total_load_kw)
        price = float(getattr(gs, "energy_price_eur_kwh", 0.15))

        total_el_cap_kw = float(sum(self._el_rated.values()))
        requested_el_kw = float(sum(el_sp * rated for rated in self._el_rated.values()))
        if self._enable_el_charge_safety and total_el_cap_kw > 1e-9:
            cheap_price = price <= self._el_charge_cheap_price_threshold
            low_soc = float(gs.h2_soc) < self._el_charge_soc_target
            renewable_excess = excess_renewable_kw > self._el_charge_renewable_excess_kw
            if low_soc and (cheap_price or renewable_excess):
                requested_el_kw = max(requested_el_kw, self._el_charge_min_fraction * total_el_cap_kw)
        requested_el_kw = min(requested_el_kw, total_el_cap_kw)

        if total_el_cap_kw > 1e-9:
            for el_id, rated in self._el_rated.items():
                share = rated / total_el_cap_kw
                actions[el_id] = {"power_setpoint_kw": requested_el_kw * share}
        else:
            for el_id in self._el_rated:
                actions[el_id] = {"power_setpoint_kw": 0.0}

        total_fc_cap_kw = float(sum(self._fc_rated.values()))
        requested_fc_kw = float(sum(fc_sp * rated for rated in self._fc_rated.values()))

        if self._enable_h2_first_safety and total_fc_cap_kw > 1e-9 and float(gs.h2_soc) > self._h2_soc_min_for_fc_safety:
            deficit_wo_fc = max(0.0, float(gs.load_kw) - renewable_kw)
            fc_needed_kw = min(deficit_wo_fc + self._unmet_safety_margin_kw, total_fc_cap_kw)
            requested_fc_kw = max(requested_fc_kw, fc_needed_kw)

        requested_fc_kw = min(requested_fc_kw, total_fc_cap_kw)
        if total_fc_cap_kw > 1e-9:
            for fc_id, rated in self._fc_rated.items():
                share = rated / total_fc_cap_kw
                actions[fc_id] = {"power_setpoint_kw": requested_fc_kw * share}
        else:
            for fc_id in self._fc_rated:
                actions[fc_id] = {"power_setpoint_kw": 0.0}

        requested_grid_kw = 0.0
        gc_ids = list(self._gc_max_import.keys())
        gc_caps = [self._gc_max_import[gcid] for gcid in gc_ids]
        total_grid_cap_kw = float(sum(gc_caps))

        rl_grid_kw = grid_fraction * total_grid_cap_kw
        requested_grid_kw = rl_grid_kw

        if self._enable_unmet_safety and bool(getattr(gs, "grid_available", True)):
            expected_supply_wo_grid = renewable_kw + requested_fc_kw
            required_grid_kw = max(
                0.0,
                float(gs.load_kw) - expected_supply_wo_grid + self._unmet_safety_margin_kw,
            )
            requested_grid_kw = max(rl_grid_kw, required_grid_kw)

        requested_grid_kw = min(requested_grid_kw, total_grid_cap_kw)

        if total_grid_cap_kw > 1e-9:
            for gc_id, max_imp in self._gc_max_import.items():
                share = max_imp / total_grid_cap_kw
                actions[gc_id] = {"power_setpoint_kw": requested_grid_kw * share}
        else:
            for gc_id in self._gc_max_import:
                actions[gc_id] = {"power_setpoint_kw": 0.0}

        # Fair demand-response policy (same enabling conditions as rule-based)
        net_renewable_kw = renewable_kw - float(gs.load_kw)
        dr_allowed = (not bool(getattr(gs, "grid_available", True))) or (
            price >= self._dr_price_high_threshold and net_renewable_kw < self._dr_deficit_threshold_kw
        )
        dr_cap = min(self._dr_max_fraction, max(0.0, -net_renewable_kw) / (float(gs.load_kw) + 1e-9))
        dr_factor = min(dr_sp * self._dr_max_fraction, dr_cap) if dr_allowed else 0.0

        for load_id in self._load_ids:
            actions[load_id] = {"demand_response": dr_factor}

        return actions
