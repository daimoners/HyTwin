"""
Energy Cost Model
=================
Time-varying electricity price model for the national power grid.

Price Structure (Italian PUN day-ahead market)
----------------------------------------------
F1  Peak:    08:00–19:00 Mon–Fri                     → high price
F2  Shoulder: 07:00–08:00, 19:00–23:00 Mon–Fri; 07:00–23:00 Sat → medium
F3  Off-peak: 23:00–07:00 all days; all day Sunday and national
    holidays (treated as Sunday-equivalent — see ``_is_italian_holiday``) → low

On top of the base tariff the model super-imposes:
  • A daily price factor drawn from N(1, σ) resampled at midnight
    (simulates day-ahead market clearing)
  • Seasonal sinusoidal trend (peak in summer for Italy — AC-driven demand)
  • A **merit-order discount**: when renewable generation is abundant
    (``renewable_cf`` fed in via :meth:`step`, e.g. a GHI/wind-speed-based
    capacity-factor proxy), zero-marginal-cost renewables push the day-ahead
    clearing price down — the same real effect behind negative/near-zero PUN
    prices on very sunny/windy days. Smoothed with an EMA since day-ahead
    prices reflect a forecast, not an instantaneous reading.
  • Rare price spike events (sudden surge due to congestion / gas crisis)

The model is *not* a BaseModel — it is a stateful service called by the
SimulationEngine or the controller, not a TwinNode.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np

from ..core.rng import resolve_rng

# Fixed-date Italian national holidays (month, day).
_FIXED_HOLIDAYS = {
    (1, 1),    # Capodanno
    (1, 6),    # Epifania
    (4, 25),   # Festa della Liberazione
    (5, 1),    # Festa del Lavoro
    (6, 2),    # Festa della Repubblica
    (8, 15),   # Ferragosto
    (11, 1),   # Ognissanti
    (12, 8),   # Immacolata Concezione
    (12, 25),  # Natale
    (12, 26),  # Santo Stefano
}


def _easter_date(year: int) -> date:
    """Gregorian Easter Sunday (anonymous/Meeus algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _is_italian_holiday(ts: datetime) -> bool:
    """True for fixed-date national holidays plus Easter Sunday/Monday."""
    if (ts.month, ts.day) in _FIXED_HOLIDAYS:
        return True
    easter = _easter_date(ts.year)
    d = date(ts.year, ts.month, ts.day)
    return d == easter or d == easter + timedelta(days=1)


class EnergyCostModel:
    """
    Time-varying electricity price service.

    Parameters
    ----------
    f1_price : float
        Peak buy price [€/kWh]. Default 0.25.
    f2_price : float
        Shoulder buy price [€/kWh]. Default 0.17.
    f3_price : float
        Off-peak buy price [€/kWh]. Default 0.09.
    price_volatility : float
        Standard deviation of the daily price factor. Default 0.08.
    seasonal_amplitude : float
        Peak-to-mean seasonal swing (fraction). Default 0.20.
    spike_prob_per_day : float
        Probability of a price spike event per day. Default 0.05.
    spike_multiplier : float
        Price spike factor (e.g. 3 → price triples). Default 3.0.
    spike_duration_steps : int
        Steps a spike lasts (step = engine dt). Default 3.
    sell_ratio : float
        Feed-in tariff = buy_price × sell_ratio. Default 0.28.
    merit_order_gain : float
        Max fractional price discount when renewable capacity factor (fed via
        ``step(..., renewable_cf=...)``) is at its highest. Default 0.22
        (up to ~22% cheaper on very sunny/windy days).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        p = params or {}
        self._rng = resolve_rng(rng)
        self._f1 = float(p.get("f1_price", 0.25))
        self._f2 = float(p.get("f2_price", 0.17))
        self._f3 = float(p.get("f3_price", 0.09))
        self._volatility = float(p.get("price_volatility", 0.08))
        self._seasonal_amp = float(p.get("seasonal_amplitude", 0.20))
        self._spike_prob = float(p.get("spike_prob_per_day", 0.05))
        self._spike_mult = float(p.get("spike_multiplier", 3.0))
        self._spike_dur = int(p.get("spike_duration_steps", 3))
        self._sell_ratio = float(p.get("sell_ratio", 0.28))
        self._merit_order_gain = float(p.get("merit_order_gain", 0.22))

        # Mutable state
        self._current_day: int = -1
        self._daily_factor: float = 1.0
        self._is_holiday: bool = False
        self._spike_remaining: int = 0
        self._spike_active: bool = False
        self._renewable_cf_ema: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_buy_price(self, timestamp: datetime) -> float:
        """Return current buy price [€/kWh]."""
        self._maybe_resample_day(timestamp)
        base = self._base_price(timestamp)
        seasonal = self._seasonal_factor(timestamp)
        spike = self._spike_mult if self._spike_active else 1.0
        merit_order = 1.0 - self._merit_order_gain * self._renewable_cf_ema
        return round(base * seasonal * self._daily_factor * spike * merit_order, 4)

    def get_sell_price(self, timestamp: datetime) -> float:
        """Return feed-in tariff [€/kWh]."""
        return round(self.get_buy_price(timestamp) * self._sell_ratio, 4)

    def step(self, timestamp: datetime, dt_seconds: float = 60.0,
              renewable_cf: Optional[float] = None) -> None:
        """
        Advance the price model by one simulation step.

        Call this *once per simulation step* before reading prices
        so that spike durations are tracked correctly.

        Parameters
        ----------
        timestamp : datetime
        dt_seconds : float
            Step duration (used to compute spike prob per step).
        renewable_cf : float, optional
            Current renewable capacity-factor proxy [0..1] (e.g. blended
            GHI/wind-speed favorability) — feeds the merit-order discount via
            an EMA (day-ahead prices reflect a forecast, not an instantaneous
            reading, so a single noisy step shouldn't swing the price).
        """
        self._maybe_resample_day(timestamp)

        if renewable_cf is not None:
            alpha = 0.15
            self._renewable_cf_ema = (
                (1 - alpha) * self._renewable_cf_ema + alpha * float(np.clip(renewable_cf, 0.0, 1.0))
            )

        # Decrement active spike
        if self._spike_active:
            self._spike_remaining -= 1
            if self._spike_remaining <= 0:
                self._spike_active = False
        else:
            # Probability of a new spike this step
            p_spike = self._spike_prob * dt_seconds / 86_400.0
            if self._rng.random() < p_spike:
                self._spike_active = True
                self._spike_remaining = self._spike_dur

    def reset(self) -> None:
        self._current_day = -1
        self._daily_factor = 1.0
        self._is_holiday = False
        self._spike_remaining = 0
        self._spike_active = False
        self._renewable_cf_ema = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spike_active(self) -> bool:
        return self._spike_active

    @property
    def nominal_f1(self) -> float:
        return self._f1

    @property
    def nominal_f3(self) -> float:
        return self._f3

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_resample_day(self, ts: datetime) -> None:
        """Resample daily price factor + holiday flag at calendar-day rollover."""
        day = ts.timetuple().tm_yday
        if day != self._current_day:
            self._current_day = day
            self._daily_factor = float(
                np.clip(self._rng.normal(1.0, self._volatility), 0.5, 2.0)
            )
            self._is_holiday = _is_italian_holiday(ts)

    def _base_price(self, ts: datetime) -> float:
        """Italian F1/F2/F3 time-of-use base price."""
        h = ts.hour
        wd = ts.weekday()   # 0 = Monday, 6 = Sunday

        if wd == 6 or self._is_holiday:
            # Sunday / national holiday — all F3
            return self._f3

        if wd < 5:
            # Monday – Friday
            if 8 <= h < 19:
                return self._f1
            if 7 <= h < 8 or 19 <= h < 23:
                return self._f2
            return self._f3

        # Saturday (wd == 5)
        if 7 <= h < 23:
            return self._f2
        return self._f3

    def _seasonal_factor(self, ts: datetime) -> float:
        """Sinusoidal seasonal modifier — peaks in July for Italy."""
        doy = ts.timetuple().tm_yday
        # Phase: peak at doy ≈ 190 (July 9)
        factor = 1.0 + self._seasonal_amp * math.sin(
            2 * math.pi * (doy - 100) / 365
        )
        return max(0.4, factor)
