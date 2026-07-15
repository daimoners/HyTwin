"""
test_weather_and_cost.py
========================
Realism-focused tests for the weather (seasonal/diurnal climatology, spatial
correlation) and energy-cost (merit-order, Italian calendar) improvements.
Run: pytest tests/test_weather_and_cost.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.weather.weather_model import WeatherModel
from hytwin.weather.weather_field import WeatherField
from hytwin.models.energy_cost import EnergyCostModel, _is_italian_holiday


# ===========================================================================
# WeatherModel — climatology realism
# ===========================================================================

def test_cloud_cover_reverts_to_configured_mean_not_to_half():
    """
    Regression test for a real bug: the cloud AR(1) process had no mean-
    reversion target, so any site's cloud fraction slowly drifted toward 0.5
    regardless of its configured cloud_cover_mean. A 'sunny' site (mean=0.15)
    run for a long stretch must stay clearly below 0.5, not converge to it.
    """
    wm = WeatherModel(latitude_deg=37.98, longitude_deg=12.51,
                       cloud_cover_mean=0.15, cloud_seasonal_amplitude=0.0,
                       rng=np.random.default_rng(3))
    ts = datetime(2024, 6, 1)  # fixed season (amplitude=0) to isolate the mean
    clouds = []
    for _ in range(3000):
        w = wm.step(ts)
        ts += timedelta(seconds=600)
        clouds.append(w["cloud_cover"])
    long_run_mean = float(np.mean(clouds[-1500:]))
    assert long_run_mean < 0.30, f"cloud fraction drifted to {long_run_mean:.3f}, expected ~0.15"


def test_wind_seasonal_climatology_winter_windier_than_summer():
    """Italy: winter (frontal systems) is climatologically windier than summer."""
    def mean_wind(month_day, n=800):
        wm = WeatherModel(weibull_c=7.0, wind_seasonal_amplitude=0.20,
                           wind_diurnal_amplitude=0.0, rng=np.random.default_rng(11))
        ts = datetime(2024, *month_day)
        vals = []
        for _ in range(n):
            vals.append(wm.step(ts)["wind_speed_ms"])
            ts += timedelta(seconds=600)
        return np.mean(vals)

    winter = mean_wind((1, 10))
    summer = mean_wind((7, 10))
    assert winter > summer


def test_synoptic_argument_couples_wind_and_cloud_upward():
    """A positive (frontal/stormy) synoptic index must boost both wind and cloud."""
    wm_calm = WeatherModel(rng=np.random.default_rng(5))
    wm_storm = WeatherModel(rng=np.random.default_rng(5))
    ts = datetime(2024, 3, 1, 12, 0)
    calm = wm_calm.step(ts, synoptic=0.0)
    storm = wm_storm.step(ts, synoptic=2.5)
    assert storm["wind_speed_ms"] > calm["wind_speed_ms"]
    assert storm["cloud_cover"] > calm["cloud_cover"]


def test_reset_preserves_rng_isolation_and_base_scale():
    """
    Regression test: WeatherModel.reset() used to rebuild WindModel without
    passing rng= (silently falling back to the legacy global RNG, breaking
    the per-instance isolation from NetworkTwin(seed=...)) and used the
    seasonally-drifted scale instead of the original base value.
    """
    rng = np.random.default_rng(9)
    wm = WeatherModel(weibull_c=6.0, rng=rng)
    wind_rng_before_reset = wm._wind._rng  # the per-instance child spawned at construction
    ts = datetime(2024, 1, 15)  # winter -> seasonal modulation shifts _wind._c
    for _ in range(50):
        wm.step(ts)
        ts += timedelta(seconds=600)
    assert wm._wind._c != pytest.approx(6.0, abs=1e-6)  # drifted, as expected pre-reset
    wm.reset()
    assert wm._wind._c == pytest.approx(6.0)  # back to the true base scale
    assert wm._wind._rng is wind_rng_before_reset  # still the SAME isolated generator, not a fresh global proxy


# ===========================================================================
# WeatherField — spatial correlation
# ===========================================================================

def test_nearby_sites_more_correlated_than_distant_sites():
    params = {
        "A": {"latitude_deg": 40.85, "longitude_deg": 14.27, "weibull_c": 6.0},
        "B": {"latitude_deg": 40.68, "longitude_deg": 14.76, "weibull_c": 6.0},  # ~50 km from A
        "C": {"latitude_deg": 45.46, "longitude_deg": 9.19, "weibull_c": 5.0},   # ~700 km from A
    }
    field = WeatherField(params, rng=np.random.default_rng(7), correlation_length_km=300.0)
    ts = datetime(2024, 3, 1)
    wa, wb, wc = [], [], []
    for _ in range(500):
        out = field.step(ts)
        ts += timedelta(seconds=600)
        wa.append(out["A"]["wind_speed_ms"])
        wb.append(out["B"]["wind_speed_ms"])
        wc.append(out["C"]["wind_speed_ms"])
    corr_near = np.corrcoef(wa, wb)[0, 1]
    corr_far = np.corrcoef(wa, wc)[0, 1]
    assert corr_near > corr_far
    assert corr_near > 0.3  # meaningfully correlated, not coincidental noise


def test_weather_field_reproducible_given_seed():
    params = {"A": {"latitude_deg": 41.0, "longitude_deg": 15.0},
              "B": {"latitude_deg": 45.0, "longitude_deg": 9.0}}
    f1 = WeatherField(params, rng=np.random.default_rng(123))
    f2 = WeatherField(params, rng=np.random.default_rng(123))
    ts = datetime(2024, 5, 1)
    for _ in range(20):
        o1, o2 = f1.step(ts), f2.step(ts)
        assert o1["A"]["wind_speed_ms"] == o2["A"]["wind_speed_ms"]
        assert o1["B"]["cloud_cover"] == o2["B"]["cloud_cover"]
        ts += timedelta(seconds=600)


def test_single_site_field_does_not_crash():
    field = WeatherField({"only": {"latitude_deg": 41.9, "longitude_deg": 12.5}})
    out = field.step(datetime(2024, 6, 1))
    assert "only" in out


# ===========================================================================
# EnergyCostModel — merit-order + Italian calendar
# ===========================================================================

def test_italian_holidays_fixed_and_easter():
    assert _is_italian_holiday(datetime(2024, 1, 1))    # Capodanno
    assert _is_italian_holiday(datetime(2024, 8, 15))   # Ferragosto
    assert _is_italian_holiday(datetime(2024, 12, 25))  # Natale
    assert _is_italian_holiday(datetime(2024, 3, 31))   # Pasqua 2024 (verified date)
    assert _is_italian_holiday(datetime(2024, 4, 1))    # Pasquetta 2024
    assert not _is_italian_holiday(datetime(2024, 4, 2))  # ordinary Tuesday


def test_holiday_priced_as_sunday_equivalent():
    """A weekday national holiday must fall back to the F3 (off-peak) band."""
    cm = EnergyCostModel({"f1_price": 0.28, "f2_price": 0.18, "f3_price": 0.09},
                          rng=np.random.default_rng(2))
    ts = datetime(2024, 8, 15, 12, 0)  # Ferragosto 2024 was a Thursday
    cm.step(ts, 600)
    price = cm.get_buy_price(ts)
    assert price < 0.15  # clearly F3-band, not the F1 peak this weekday-hour would imply


def test_merit_order_discount_lowers_price_with_high_renewable_favorability():
    ts = datetime(2024, 6, 17, 12, 0)  # Monday noon, F1 peak hour
    cm_low = EnergyCostModel({"f1_price": 0.28}, rng=np.random.default_rng(1))
    cm_high = EnergyCostModel({"f1_price": 0.28}, rng=np.random.default_rng(1))
    for _ in range(25):
        cm_low.step(ts, 600, renewable_cf=0.0)
        cm_high.step(ts, 600, renewable_cf=1.0)
    assert cm_high.get_buy_price(ts) < cm_low.get_buy_price(ts)


def test_reset_clears_holiday_and_merit_order_state():
    cm = EnergyCostModel(rng=np.random.default_rng(1))
    cm.step(datetime(2024, 8, 15, 12, 0), 600, renewable_cf=1.0)
    cm.reset()
    assert cm._is_holiday is False
    assert cm._renewable_cf_ema == 0.0
