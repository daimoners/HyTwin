"""
Wind Turbine Physics Model
==========================
Implements a realistic wind turbine power curve with:
 - Betz / power coefficient model
 - Cut-in / rated / cut-out speed envelope
 - Air density correction for altitude and temperature
 - Optional stochastic turbulence on wind speed
 - Tower shadow and wake losses (simplified)

Physical reference
------------------
Burton, T. et al. (2011). "Wind Energy Handbook", Wiley.
IEC 61400-12-1:2017 power performance standard.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .base_model import BaseModel, ModelState
from ..core.rng import resolve_rng

# Standard atmospheric constants
_RHO_STP = 1.225       # kg/m³ at sea level, 15 °C
_R_AIR = 287.05        # J/(kg·K) specific gas constant for dry air
_g = 9.80665           # m/s²
_T_STP = 288.15        # K


def _air_density(altitude_m: float, temperature_c: float) -> float:
    """ISA pressure and density at altitude with local temperature."""
    P0 = 101_325.0  # Pa
    T_K = temperature_c + 273.15
    # Hypsometric formula
    P = P0 * (1 - 0.0065 * altitude_m / _T_STP) ** 5.2561
    return P / (_R_AIR * T_K)


def _power_coefficient(tsr: float, pitch_deg: float = 0.0) -> float:
    """
    Simplified Cp(λ, β) using the analytical approximation from
    Heier (2014) "Grid Integration of Wind Energy Systems".

    λ = tip-speed ratio, β = pitch angle (degrees)
    """
    beta = pitch_deg
    lambda_i = 1.0 / (1.0 / (tsr + 0.08 * beta) - 0.035 / (beta ** 3 + 1))
    Cp = (
        0.5176
        * (116.0 / lambda_i - 0.4 * beta - 5.0)
        * math.exp(-21.0 / lambda_i)
        + 0.0068 * tsr
    )
    return max(0.0, min(Cp, 0.593))   # Betz limit


class WindTurbineModel(BaseModel):
    """
    Variable-speed wind turbine model.

    Parameters (in ``params`` dict)
    --------------------------------
    rotor_diameter_m : float   — rotor diameter [m]
    hub_height_m : float       — hub height above ground [m], default 80
    rated_power_kw : float     — rated electrical output [kW]
    v_cut_in : float           — cut-in wind speed [m/s], default 3.0
    v_rated : float            — rated wind speed [m/s], default 12.0
    v_cut_out : float          — cut-out wind speed [m/s], default 25.0
    efficiency_gen : float     — generator + gearbox electrical efficiency
    altitude_m : float         — installation altitude ASL [m], default 0
    wake_loss_factor : float   — fractional wake/shadow loss, default 0.0
    turbulence_intensity : float — wind speed turbulence σ/μ, default 0.05
    """

    def __init__(self, component_id: str, params: Dict[str, Any],
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(component_id, params)
        self._rng = resolve_rng(rng)
        self._rotor_radius = params["rotor_diameter_m"] / 2.0
        self._swept_area = math.pi * self._rotor_radius ** 2
        self._rated_kw = float(params["rated_power_kw"])
        self._v_ci = float(params.get("v_cut_in", 3.0))
        self._v_r = float(params.get("v_rated", 12.0))
        self._v_co = float(params.get("v_cut_out", 25.0))
        self._eta_gen = float(params.get("efficiency_gen", 0.94))
        self._altitude = float(params.get("altitude_m", 0.0))
        self._wake_loss = float(params.get("wake_loss_factor", 0.0))
        self._turb_I = float(params.get("turbulence_intensity", 0.05))
        self._hub_height = float(params.get("hub_height_m", 80.0))
        # Nominal tip-speed ratio for max Cp
        self._tsr_opt = 7.0
        self._omega_rated = self._tsr_opt * self._v_r / self._rotor_radius  # rad/s

        # Internal state
        self._power_kw: float = 0.0
        self._energy_kwh: float = 0.0   # accumulated

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _wind_to_power(self, v_hub: float, temperature_c: float = 15.0) -> float:
        """
        Convert hub-height wind speed to electrical power [kW].
        Includes: power curve envelope, air density correction, wake loss.
        """
        if v_hub < self._v_ci or v_hub > self._v_co:
            return 0.0

        rho = _air_density(self._altitude, temperature_c)
        rho_corr = rho / _RHO_STP  # density correction factor

        if v_hub >= self._v_r:
            p_kw = self._rated_kw

        else:
            # Optimal tip-speed ratio operation below rated
            tsr = self._tsr_opt
            Cp = _power_coefficient(tsr, pitch_deg=0.0)
            p_aero = 0.5 * rho * self._swept_area * v_hub ** 3 * Cp  # W
            p_kw = min(p_aero * self._eta_gen / 1000.0 * rho_corr, self._rated_kw)

        # Wake / shadow losses
        p_kw *= 1.0 - self._wake_loss
        return max(0.0, p_kw)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        Parameters
        ----------
        dt : float
            Time step [s].
        context : dict
            Expected keys:
            - ``wind_speed_ms`` : float  — wind speed at reference height
            - ``wind_ref_height_m`` : float — anemometer height (default 10)
            - ``temperature_c`` : float  — ambient temperature (default 15)
            - ``action_pitch_deg`` : float — optional pitch override (default 0)
        """
        v_ref = float(context.get("wind_speed_ms", 0.0))
        z_ref = float(context.get("wind_ref_height_m", 10.0))
        temp_c = float(context.get("temperature_c", 15.0))

        # Wind shear: power-law profile α ≈ 0.14 (onshore)
        alpha = 0.14
        v_hub = v_ref * (self._hub_height / z_ref) ** alpha if z_ref > 0 else v_ref

        # Stochastic turbulence
        if self._turb_I > 0:
            v_hub = max(0.0, v_hub + self._rng.normal(0, self._turb_I * v_hub))

        p_kw = self._wind_to_power(v_hub, temp_c)
        energy_kwh = p_kw * dt / 3600.0
        self._power_kw = p_kw
        self._energy_kwh += energy_kwh

        ts = context.get("timestamp", datetime.utcnow())
        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "power_kw": p_kw,
                "energy_kwh_step": energy_kwh,
                "energy_kwh_total": self._energy_kwh,
                "wind_speed_hub_ms": v_hub,
                "wind_speed_ref_ms": v_ref,
                "available": float(self._v_ci <= v_hub <= self._v_co),
            },
        )
        return self._state

    def reset(self) -> None:
        self._power_kw = 0.0
        self._energy_kwh = 0.0
        self._state = None
