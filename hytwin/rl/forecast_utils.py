"""
Physics-based multi-step forecast utilities for RL observation construction.
=============================================================================

Provides noise-free expected-value forecasts of:
  - Energy price      : EnergyCostModel peeks (stateless, no RNG mutation)
  - Wind power        : AR(1) mean reversion in capacity-factor space
  - PV power          : solar geometry (clear-sky model) + cloud persistence
  - Load              : daily sinusoidal industrial pattern
  - Renewable surplus : (wind + pv - load) as normalised balance signal

All helpers are pure functions (no internal state). This module is imported
by both AdvancedH2GridEnv (training) and RLController (inference) to guarantee
identical observation encoding.

Feature layout per forecast step (N_FORECAST_FEATURES_PER_STEP = 8):
  [0] f_price_norm    – future buy price, scaled to [-1,+1]
  [1] f_price_delta   – (price[t+k] - price[t]) / ref_delta, clip [-1,+1]
  [2] f_wind_norm     – AR(1) expected wind power, scaled to [-1,+1]
  [3] f_pv_norm       – solar-geometry PV forecast, scaled to [-1,+1]
  [4] f_load_norm     – daily-pattern load forecast, scaled to [-1,+1]
  [5] f_renew_surplus – (wind+pv-load) / load_rated, clip [-1,+1]
  [6] f_sin_hour      – sin of future fractional hour angle
  [7] f_cos_hour      – cos of future fractional hour angle

No feature must ever convey information that would be unavailable to a real
operator (e.g., exact future noise realisations). AR(1) and solar geometry
represent operationally realistic short-horizon forecasts.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Feature-space constants
# ---------------------------------------------------------------------------

N_FORECAST_FEATURES_PER_STEP: int = 8

FORECAST_FEATURE_NAMES: List[str] = [
    "f_price_norm",       # future price (normalised)
    "f_price_delta",      # price trend (future - present)
    "f_wind_norm",        # AR(1) wind power forecast (normalised)
    "f_pv_norm",          # solar-geometry PV forecast (normalised)
    "f_load_norm",        # daily-pattern load forecast (normalised)
    "f_renew_surplus",    # renewable surplus or deficit (normalised)
    "f_sin_hour",         # sin(hour angle at future step)
    "f_cos_hour",         # cos(hour angle at future step)
]

# ---------------------------------------------------------------------------
# Solar geometry helpers (inline, no external deps)
# ---------------------------------------------------------------------------

def _solar_elevation_deg(ts: datetime, lat_deg: float, lon_deg: float) -> float:
    """Solar elevation angle [deg] using the Spencer (1971) declination formula."""
    doy = ts.timetuple().tm_yday
    hour_utc = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    B = 2.0 * math.pi * (doy - 1) / 365.0
    decl = (180.0 / math.pi) * (
        0.006918
        - 0.399912 * math.cos(B) + 0.070257 * math.sin(B)
        - 0.006758 * math.cos(2 * B) + 0.000907 * math.sin(2 * B)
        - 0.002697 * math.cos(3 * B) + 0.00148  * math.sin(3 * B)
    )
    eot = 229.18 * (
        0.000075 + 0.001868 * math.cos(B) - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B) - 0.04089 * math.sin(2 * B)
    )
    lstm = round(lon_deg / 15.0) * 15.0
    tc = 4.0 * (lon_deg - lstm) + eot
    lct = hour_utc + tc / 60.0
    ha_deg = 15.0 * (lct - 12.0)

    lat_r = math.radians(lat_deg)
    decl_r = math.radians(decl)
    ha_r = math.radians(ha_deg)
    cos_z = max(-1.0, min(1.0,
        math.sin(lat_r) * math.sin(decl_r)
        + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r)
    ))
    zenith = math.degrees(math.acos(cos_z))
    return max(0.0, 90.0 - zenith)


def _ghi_clear_sky(elevation_deg: float, altitude_m: float = 50.0) -> float:
    """Simplified Ineichen clear-sky GHI [W/m²] from solar elevation."""
    if elevation_deg <= 0.0:
        return 0.0
    zenith_deg = 90.0 - elevation_deg
    cos_z = math.cos(math.radians(zenith_deg))
    alt_factor = math.exp(-altitude_m / 8500.0)
    TL = 2.5
    AM = 1.0 / (cos_z + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364))
    tau = math.exp(-0.8662 * TL * AM * alt_factor * 0.0296)
    return max(0.0, 1361.0 * cos_z * tau)


# Reference clear-sky GHI at peak elevation (~45°), used for night normalisation
_GHI_PEAK_REF: float = _ghi_clear_sky(45.0, 50.0)


# ---------------------------------------------------------------------------
# Individual forecast functions
# ---------------------------------------------------------------------------

def ar1_wind_power_forecast(
    current_wind_kw: float,
    lag_k: int,
    wind_rated_kw: float,
    autocorr: float = 0.88,
    capacity_factor_mean: float = 0.28,
) -> float:
    """
    AR(1) expected wind power at lag *lag_k* steps ahead.

    Works in capacity-factor space to avoid power-curve inversion.
    Mean reverts toward *capacity_factor_mean* (long-run Weibull CF).
    """
    cf_now = float(np.clip(current_wind_kw / max(1.0, wind_rated_kw), 0.0, 1.0))
    alpha = autocorr ** lag_k
    cf_fcast = alpha * cf_now + (1.0 - alpha) * capacity_factor_mean
    return float(np.clip(cf_fcast, 0.0, 1.0)) * wind_rated_kw


def solar_pv_power_forecast(
    ts_now: datetime,
    ts_future: datetime,
    current_pv_kw: float,
    pv_rated_kw: float,
    cloud_cover_now: float,
    cloud_mean: float,
    cloud_autocorr: float,
    lag_k: int,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
) -> float:
    """
    PV power forecast using solar geometry + AR(1) cloud persistence.

    At inference, *cloud_cover_now* can be estimated from the current
    PV/clear-sky ratio; at training it is the exact WeatherModel state.
    """
    elev_now = _solar_elevation_deg(ts_now, lat_deg, lon_deg)
    elev_fcast = _solar_elevation_deg(ts_future, lat_deg, lon_deg)

    ghi_now = _ghi_clear_sky(elev_now, alt_m)
    ghi_fcast = _ghi_clear_sky(elev_fcast, alt_m)

    # Cloud AR(1) mean reversion
    alpha_c = cloud_autocorr ** lag_k
    cloud_fcast = float(np.clip(
        alpha_c * cloud_cover_now + (1.0 - alpha_c) * cloud_mean,
        0.0, 1.0,
    ))
    cloud_atten_now   = 1.0 - 0.75 * cloud_cover_now ** 3.4
    cloud_atten_fcast = 1.0 - 0.75 * cloud_fcast    ** 3.4

    ghi_actual_now   = ghi_now   * cloud_atten_now
    ghi_actual_fcast = ghi_fcast * cloud_atten_fcast

    if ghi_actual_now > 5.0:
        # Proportional scaling from current actual production
        pv_fcast = current_pv_kw * (ghi_actual_fcast / max(1.0, ghi_actual_now))
    else:
        # Night or deep overcast: estimate from rated power × clear-sky fraction
        pv_fcast = pv_rated_kw * cloud_atten_fcast * (ghi_fcast / max(1.0, _GHI_PEAK_REF))

    return float(np.clip(pv_fcast, 0.0, pv_rated_kw))


def load_day_pattern_forecast(
    ts_now: datetime,
    ts_future: datetime,
    current_load_kw: float,
    load_rated_kw: float = 760.0,
    amplitude: float = 0.25,
) -> float:
    """
    Daily sinusoidal load forecast for an industrial profile.

    Profile: peak at ~14:00, trough at ~02:00.
    The base load is back-calculated from the current observation.
    """
    def _scale(h: int, m: int) -> float:
        angle = 2.0 * math.pi * (h + m / 60.0 - 2.0) / 24.0
        return 1.0 + amplitude * math.cos(angle)

    scale_now   = _scale(ts_now.hour,    ts_now.minute)
    scale_fcast = _scale(ts_future.hour, ts_future.minute)
    base_load   = current_load_kw / max(1e-9, scale_now)
    return float(np.clip(base_load * scale_fcast, 0.0, load_rated_kw * 1.5))


# ---------------------------------------------------------------------------
# Main feature builder (called identically from env and controller)
# ---------------------------------------------------------------------------

def build_forecast_features(
    ts: datetime,
    current_price: float,
    current_wind_kw: float,
    current_pv_kw: float,
    current_load_kw: float,
    n_steps: int,
    dt_seconds: float,
    forecast_step_mult: int,
    wind_rated_kw: float,
    pv_rated_kw: float,
    load_rated_kw: float,
    peek_price_fn: Optional[Callable[[datetime], float]],
    # AR(1) wind params
    wind_autocorr: float = 0.88,
    wind_capacity_factor_mean: float = 0.28,
    # Cloud / PV params
    cloud_cover_now: float = 0.35,
    cloud_mean: float = 0.35,
    cloud_autocorr: float = 0.90,
    # Location
    lat_deg: float = 40.5,
    lon_deg: float = 14.8,
    alt_m: float = 50.0,
    # Normalisation references
    price_ref: float = 0.50,
    price_delta_ref: float = 0.20,
) -> np.ndarray:
    """
    Build the physics-based multi-step forecast feature vector.

    Returns an array of shape ``(n_steps * N_FORECAST_FEATURES_PER_STEP,)``
    with every element clipped to ``[-1, +1]``.

    This function **must be called with identical arguments** in
    ``AdvancedH2GridEnv`` and ``RLController`` to ensure consistent
    observation encoding between training and inference.

    Parameters
    ----------
    ts : datetime
        Current simulation timestamp.
    current_price : float
        Current energy buy price [€/kWh].
    current_wind_kw, current_pv_kw, current_load_kw : float
        Current measured power values [kW].
    n_steps : int
        Number of forecast horizons.
    dt_seconds : float
        Base simulation step [s].
    forecast_step_mult : int
        Forecast step multiplier (forecast stride = dt × mult).
    wind_rated_kw, pv_rated_kw, load_rated_kw : float
        Installed capacity / reference power for normalisation [kW].
    peek_price_fn : callable or None
        Stateless function ``f(timestamp) -> float`` returning buy price
        without advancing RNG. Pass ``None`` to use constant current price.
    wind_autocorr : float
        Lag-1 autocorrelation for the wind AR(1) model.
    wind_capacity_factor_mean : float
        Long-run wind capacity factor for mean reversion.
    cloud_cover_now : float  [0, 1]
        Current cloud-cover fraction (exact at training, inferred at inference).
    cloud_mean, cloud_autocorr : float
        Cloud AR(1) parameters for PV persistence forecast.
    lat_deg, lon_deg, alt_m : float
        Site coordinates for solar geometry.
    price_ref, price_delta_ref : float
        Normalisation denominators for price features.
    """
    if n_steps <= 0:
        return np.zeros(0, dtype=np.float32)

    dt_fcast = dt_seconds * float(forecast_step_mult)
    feats: List[float] = []

    for k in range(1, n_steps + 1):
        ft = ts + timedelta(seconds=k * dt_fcast)
        angle = 2.0 * math.pi * (ft.hour + ft.minute / 60.0) / 24.0

        # ── 0: price forecast ─────────────────────────────────────────
        f_price = peek_price_fn(ft) if peek_price_fn is not None else current_price
        f_price_norm  = 2.0 * float(np.clip(f_price / (price_ref + 1e-9), 0.0, 1.0)) - 1.0
        f_price_delta = float(np.clip(
            (f_price - current_price) / (price_delta_ref + 1e-9), -1.0, 1.0
        ))

        # ── 2: wind AR(1) ─────────────────────────────────────────────
        f_wind_kw = ar1_wind_power_forecast(
            current_wind_kw, k, wind_rated_kw, wind_autocorr, wind_capacity_factor_mean
        )
        f_wind_norm = 2.0 * float(np.clip(f_wind_kw / (wind_rated_kw + 1e-9), 0.0, 1.0)) - 1.0

        # ── 3: PV solar geometry ──────────────────────────────────────
        f_pv_kw = solar_pv_power_forecast(
            ts, ft, current_pv_kw, pv_rated_kw,
            cloud_cover_now, cloud_mean, cloud_autocorr, k,
            lat_deg, lon_deg, alt_m,
        )
        f_pv_norm = 2.0 * float(np.clip(f_pv_kw / (pv_rated_kw + 1e-9), 0.0, 1.0)) - 1.0

        # ── 4: load daily pattern ─────────────────────────────────────
        f_load_kw = load_day_pattern_forecast(ts, ft, current_load_kw, load_rated_kw)
        f_load_norm = 2.0 * float(np.clip(f_load_kw / (load_rated_kw + 1e-9), 0.0, 1.0)) - 1.0

        # ── 5: renewable surplus signal ───────────────────────────────
        f_renew_surplus = float(np.clip(
            (f_wind_kw + f_pv_kw - f_load_kw) / (load_rated_kw + 1e-9), -1.0, 1.0
        ))

        # ── 6-7: time encoding ────────────────────────────────────────
        feats.extend([
            f_price_norm,
            f_price_delta,
            f_wind_norm,
            f_pv_norm,
            f_load_norm,
            f_renew_surplus,
            float(math.sin(angle)),
            float(math.cos(angle)),
        ])

    return np.clip(np.asarray(feats, dtype=np.float32), -1.0, 1.0)
