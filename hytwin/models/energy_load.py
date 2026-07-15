"""
Energy Load Model
=================
Stochastic energy consumption profile with:
  - Diurnal patterns (hour-of-day, profile-specific)
  - Weekday/weekend structure — profile-specific: industrial drops sharply
    on weekends, commercial moderately, residential mildly
  - Italian national holidays treated as weekend-equivalent
  - Seasonal pattern — winter heating peak for all profiles, PLUS a secondary
    summer air-conditioning bump for commercial (and a smaller one for
    residential), matching real Italian consumption seasonality
  - August industrial shutdown ("ferie estive") — many Italian plants close
    for 1-3 weeks in August; modelled as a configurable output dip
  - Gaussian noise on top of the deterministic profile
  - Programmable peak shaving / demand response signals

Physical reference
------------------
IEEE Std 1459-2010 — Definitions for the Measurement of Electric Power
ENTSO-E consumption profiles (simplified)
Terna "Rapporto Mensile sul Sistema Elettrico" — Italian load seasonality reference
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .base_model import BaseModel, ModelState
from .energy_cost import _is_italian_holiday
from ..core.rng import resolve_rng


def _diurnal_profile(hour: float, profile_type: str = "residential") -> float:
    """
    Returns a normalised load fraction [0..1] based on hour of day.
    """
    t = hour % 24
    if profile_type == "residential":
        # Two peaks: morning ~7 h, evening ~20 h
        morning = 0.6 * math.exp(-0.5 * ((t - 7.0) / 1.5) ** 2)
        evening = 1.0 * math.exp(-0.5 * ((t - 20.0) / 2.0) ** 2)
        base = 0.25
        return base + morning + evening
    elif profile_type == "industrial":
        # Flat 07-22, reduced nights
        if 7 <= t < 22:
            return 1.0 - 0.1 * math.sin(math.pi * (t - 7) / 15)
        else:
            return 0.40
    else:  # commercial
        if 9 <= t < 18:
            return 0.80 + 0.10 * math.sin(math.pi * (t - 9) / 9)
        elif 18 <= t < 22:
            return 0.50
        else:
            return 0.15


# Weekday/weekend/holiday multipliers, profile-specific — industrial plants
# largely shut on Sunday/holidays and run reduced on Saturday; commercial
# (shops, offices) drops moderately on Sunday; residential is only mildly
# affected (people are still home, just on a different daily rhythm).
_WEEKEND_FACTORS = {
    "industrial":  {"sat": 0.55, "sun": 0.30},
    "commercial":  {"sat": 0.95, "sun": 0.55},
    "residential": {"sat": 0.92, "sun": 0.85},
}


class EnergyLoadModel(BaseModel):
    """
    Aggregated load model for a grid node.

    Parameters (``params`` dict)
    ----------------------------
    base_load_kw : float       — mean load during peak hour [kW]
    profile_type : str         — 'residential' | 'industrial' | 'commercial'
    noise_std_fraction : float — noise standard deviation as fraction, default 0.03
    seasonal_amplitude : float — winter-heating seasonal amplitude, default 0.10
    summer_ac_amplitude : float — secondary summer air-conditioning bump,
        default 0.12 for 'commercial', 0.05 for 'residential', 0 for 'industrial'
        (overridable per-load via this param)
    weekday_sensitive : bool   — apply weekend/holiday reduction, default True
    august_shutdown : bool     — model the Italian industrial August summer
        shutdown ("ferie estive"), default True only for 'industrial'
    august_shutdown_factor : float — output fraction during the shutdown window
        (day 213-231, ~Aug 1-19), default 0.35
    demand_response_factor : float — max DR reduction fraction, default 0.15
    """

    def __init__(self, component_id: str, params: Dict[str, Any],
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(component_id, params)
        self._rng = resolve_rng(rng)
        self._base_kw = float(params["base_load_kw"])
        self._profile = str(params.get("profile_type", "residential"))
        self._noise_std = float(params.get("noise_std_fraction", 0.03))
        self._seasonal_amp = float(params.get("seasonal_amplitude", 0.10))
        _default_ac = {"commercial": 0.12, "residential": 0.05, "industrial": 0.0}.get(self._profile, 0.0)
        self._summer_ac_amp = float(params.get("summer_ac_amplitude", _default_ac))
        self._dr_factor = float(params.get("demand_response_factor", 0.15))
        self._weekday_sensitive = bool(params.get("weekday_sensitive", True))
        self._august_shutdown = bool(params.get("august_shutdown", self._profile == "industrial"))
        self._august_shutdown_factor = float(params.get("august_shutdown_factor", 0.35))

        self._load_kw: float = self._base_kw
        self._energy_kwh: float = 0.0

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        context keys:
          timestamp        : datetime — simulation timestamp
          demand_response  : float   — DR signal [0..1] (fraction to shed)
        """
        ts = context.get("timestamp", datetime.utcnow())
        dr = float(np.clip(context.get("demand_response", 0.0), 0.0, 1.0))

        hour = ts.hour + ts.minute / 60.0
        day_of_year = ts.timetuple().tm_yday

        # Winter-heating seasonal factor + secondary summer AC bump (peaks in
        # opposite halves of the year: heating ~mid-Jan, AC ~late-Jul).
        seasonal = (
            1.0
            + self._seasonal_amp * math.cos(2 * math.pi * (day_of_year - 15) / 365)
            + self._summer_ac_amp * max(0.0, math.cos(2 * math.pi * (day_of_year - 205) / 365))
        )

        # Weekday/weekend/holiday factor — profile-specific (industrial drops
        # most on Sunday/holidays, commercial moderately, residential mildly).
        weekend_factor = 1.0
        if self._weekday_sensitive:
            dow = ts.weekday()  # 0=Mon, 6=Sun
            factors = _WEEKEND_FACTORS.get(self._profile, _WEEKEND_FACTORS["residential"])
            if dow == 6 or _is_italian_holiday(ts):
                weekend_factor = factors["sun"]
            elif dow == 5:
                weekend_factor = factors["sat"]

        # Italian industrial "ferie estive" — many plants close for ~2-3
        # weeks in August (here: day-of-year 213-231, i.e. roughly Aug 1-19).
        august_factor = 1.0
        if self._august_shutdown and 213 <= day_of_year <= 231:
            august_factor = self._august_shutdown_factor

        diurnal = _diurnal_profile(hour, self._profile)

        load = (
            self._base_kw
            * diurnal
            * seasonal
            * weekend_factor
            * august_factor
            * (1.0 + self._rng.normal(0, self._noise_std))
        )
        load = max(0.0, load)

        # Apply demand response (shed up to dr_factor)
        dr_shed = min(dr, self._dr_factor) * load
        load_final = max(0.0, load - dr_shed)

        energy_kwh = load_final * dt / 3600.0
        self._load_kw = load_final
        self._energy_kwh += energy_kwh

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "load_kw": load_final,
                "base_load_kw": self._base_kw,
                "diurnal_factor": diurnal,
                "seasonal_factor": seasonal,
                "weekend_factor": weekend_factor,
                "august_factor": august_factor,
                "dr_shed_kw": dr_shed,
                "energy_kwh_step": energy_kwh,
                "energy_kwh_total": self._energy_kwh,
            },
        )
        return self._state

    def reset(self) -> None:
        self._load_kw = self._base_kw
        self._energy_kwh = 0.0
        self._state = None
