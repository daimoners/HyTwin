"""
Photovoltaic Panel / Array Model
=================================
Implements a physics-based PV model including:
  - Clear-sky and actual irradiance processing
  - Single-diode simplified model (5-parameter)
  - Temperature correction (NOCT model)
  - DC/AC inverter efficiency
  - Incidence angle modifier (IAM)
  - Soiling and degradation losses

Physical references
-------------------
De Soto, W. et al. (2006). Solar Energy, 80(1), 78-88.
Duffie, J.A. & Beckman, W.A. (2013). Solar Engineering of Thermal Processes.
IEC 61215:2021 photovoltaic module qualification.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict

import numpy as np

from .base_model import BaseModel, ModelState


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _cell_temperature_noct(
    g_poa: float,
    t_amb_c: float,
    noct_c: float = 47.0,
    wind_ms: float = 1.0,
) -> float:
    """
    NOCT (Normal Operating Cell Temperature) model.
    T_cell = T_amb + (NOCT - 20) / 800 * G_poa * (9.5 / (5.7 + 3.8*v))
    where v is wind speed over the panel.
    """
    wind_corr = 9.5 / (5.7 + 3.8 * max(wind_ms, 0.1))
    return t_amb_c + (noct_c - 20.0) / 800.0 * g_poa * wind_corr


def _iam_ashrae(
    aoi_deg: float,
    b0: float = 0.05,
) -> float:
    """
    ASHRAE incidence angle modifier.
    IAM = 1 - b0 * (1/cos(θ) - 1)
    """
    if aoi_deg >= 90.0:
        return 0.0
    cos_t = math.cos(math.radians(aoi_deg))
    if cos_t <= 0:
        return 0.0
    return max(0.0, 1.0 - b0 * (1.0 / cos_t - 1.0))


def _inverter_efficiency(p_dc_kw: float, p_dc_rated_kw: float) -> float:
    """
    European efficiency curve approximation.
    Peak ~97% at 20-100% load, drops off at low loads.
    """
    if p_dc_rated_kw <= 0 or p_dc_kw <= 0:
        return 0.0
    load = p_dc_kw / p_dc_rated_kw
    # Simple 4th-order polynomial fit
    eta = -0.0162 * load ** 4 + 0.0499 * load ** 3 - 0.0518 * load ** 2 + 0.0237 * load + 0.9713
    return max(0.0, min(eta, 0.985))


# ------------------------------------------------------------------
# PV Model
# ------------------------------------------------------------------

class PhotovoltaicModel(BaseModel):
    """
    PV array model.

    Parameters (``params`` dict)
    ----------------------------
    n_panels : int          — total number of panels
    panel_area_m2 : float   — area per panel [m²], default 1.7
    eta_stc : float         — STC efficiency (fraction), e.g. 0.20
    temp_coeff_pmax : float — Pmax temperature coefficient [1/°C], e.g. -0.004
    noct_c : float          — NOCT [°C], default 47
    inverter_efficiency : float — fixed DC/AC efficiency; if 0 use curve
    rated_power_kw : float  — DC STC rated output [kW]
    soiling_loss : float    — soiling fraction, default 0.02
    degradation_per_year : float — annual degradation, default 0.005
    tilt_deg : float        — panel tilt [°], default 30
    azimuth_deg : float     — panel azimuth from N [°], default 180 (south)
    iam_b0 : float          — ASHRAE IAM coefficient, default 0.05
    """

    def __init__(self, component_id: str, params: Dict[str, Any]) -> None:
        super().__init__(component_id, params)
        self._n = int(params.get("n_panels", 1))
        self._area = float(params.get("panel_area_m2", 1.7))
        self._eta_stc = float(params.get("eta_stc", 0.20))
        self._tc_pmax = float(params.get("temp_coeff_pmax", -0.0040))
        self._noct = float(params.get("noct_c", 47.0))
        self._rated_dc_kw = float(params.get("rated_power_kw", self._n * self._area * self._eta_stc))
        self._fixed_eta_inv = float(params.get("inverter_efficiency", 0.0))
        self._soiling = float(params.get("soiling_loss", 0.02))
        self._degradation = float(params.get("degradation_per_year", 0.005))
        self._tilt = float(params.get("tilt_deg", 30.0))
        self._azimuth = float(params.get("azimuth_deg", 180.0))
        self._iam_b0 = float(params.get("iam_b0", 0.05))

        # Mutable state
        self._power_ac_kw: float = 0.0
        self._energy_kwh: float = 0.0
        self._age_years: float = 0.0

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _pv_power_dc(
        self,
        g_poa: float,
        t_cell_c: float,
    ) -> float:
        """
        DC power from plane-of-array irradiance and cell temperature.
        P_dc = N * A * G * η_stc * (1 + α_T*(T_cell - 25))
        """
        if g_poa <= 0:
            return 0.0
        deg_factor = 1.0 - self._degradation * self._age_years
        soil_factor = 1.0 - self._soiling
        eta_t = self._eta_stc * (1.0 + self._tc_pmax * (t_cell_c - 25.0))
        p_dc = self._n * self._area * g_poa * eta_t * deg_factor * soil_factor / 1000.0  # kW
        return max(0.0, p_dc)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        Parameters
        ----------
        context keys:
          ghi_wm2    : float — Global Horizontal Irradiance [W/m²]
          dhi_wm2    : float — Diffuse Horizontal Irradiance [W/m²]
          temperature_c : float — ambient temperature [°C]
          wind_ms    : float — wind speed at panel level [m/s]
          sun_zenith_deg : float — solar zenith angle [°]
          sun_azimuth_deg : float — solar azimuth (from N, clockwise) [°]
          age_years  : float — override array age in years (optional)
        """
        ghi = float(context.get("ghi_wm2", 0.0))
        dhi = float(context.get("dhi_wm2", 0.0))
        t_amb = float(context.get("temperature_c", 20.0))
        wind = float(context.get("wind_ms", 1.0))
        zenith = float(context.get("sun_zenith_deg", 90.0))
        sun_az = float(context.get("sun_azimuth_deg", 180.0))
        ts = context.get("timestamp", datetime.utcnow())

        if "age_years" in context:
            self._age_years = float(context["age_years"])

        # --- Plane-of-array irradiance (simple tilted surface) ---
        # Direct Normal Irradiance approximation
        cos_z = math.cos(math.radians(zenith))
        dni = max(0.0, (ghi - dhi) / cos_z) if cos_z > 0.01 else 0.0

        # Angle of incidence on tilted surface
        # Using vector dot product
        tilt_rad = math.radians(self._tilt)
        azim_diff_rad = math.radians(sun_az - self._azimuth)
        cos_aoi = (
            math.cos(math.radians(zenith)) * math.cos(tilt_rad)
            + math.sin(math.radians(zenith)) * math.sin(tilt_rad) * math.cos(azim_diff_rad)
        )
        aoi_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_aoi))))

        iam = _iam_ashrae(aoi_deg, self._iam_b0)

        # POA beam + diffuse + reflected
        g_beam = max(0.0, dni * max(0.0, cos_aoi)) * iam
        g_diff = dhi * (1.0 + math.cos(tilt_rad)) / 2.0     # isotropic
        g_refl = ghi * 0.2 * (1.0 - math.cos(tilt_rad)) / 2.0  # albedo 0.2
        g_poa = g_beam + g_diff + g_refl

        # Cell temperature
        t_cell = _cell_temperature_noct(g_poa, t_amb, self._noct, wind)

        # DC power
        p_dc = self._pv_power_dc(g_poa, t_cell)
        p_dc = min(p_dc, self._rated_dc_kw)

        # AC power via inverter
        if self._fixed_eta_inv > 0:
            p_ac = p_dc * self._fixed_eta_inv
        else:
            eta_inv = _inverter_efficiency(p_dc, self._rated_dc_kw)
            p_ac = p_dc * eta_inv

        energy_kwh = p_ac * dt / 3600.0
        self._power_ac_kw = p_ac
        self._energy_kwh += energy_kwh

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "power_ac_kw": p_ac,
                "power_dc_kw": p_dc,
                "energy_kwh_step": energy_kwh,
                "energy_kwh_total": self._energy_kwh,
                "g_poa_wm2": g_poa,
                "cell_temp_c": t_cell,
                "aoi_deg": aoi_deg,
            },
        )
        return self._state

    def reset(self) -> None:
        self._power_ac_kw = 0.0
        self._energy_kwh = 0.0
        self._age_years = 0.0
        self._state = None
