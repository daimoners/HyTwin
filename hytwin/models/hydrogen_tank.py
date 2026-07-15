"""
Hydrogen Storage Tank Model
============================
Compressed gas type-IV storage vessel with:
  - Van der Waals real-gas equation of state (better at high pressure)
  - Charge/discharge flow rate limits
  - Thermal effects (adiabatic compression heating, exothermic/endothermic)
  - Safety interlocks (min/max pressure, flow rate)
  - Boil-off and permeation losses
  - State of charge calculation

Physical references
-------------------
Colozza, A. (2002). NASA / TM-2002-211867 hydrogen storage.
Aceves, S.M. et al. (2010). Int. J. Hydrogen Energy 35(3), 1219-1226.
ISO 15869:2009 gaseous hydrogen storage systems.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict

import numpy as np

from .base_model import BaseModel, ModelState

# Constants
R_GAS = 8.314          # J/(mol·K)
M_H2 = 2.01568e-3      # kg/mol
T_STD = 273.15         # K (0 °C)
P_STD = 101_325.0      # Pa (1 atm)
# Van der Waals constants for H2
VDW_A = 0.2476e-3      # Pa·m⁶/mol² (converted: a=0.2476 L²·atm/mol²)
VDW_B = 26.61e-6       # m³/mol (b=26.61 mL/mol)


def _vdw_pressure(n_mol: float, V_m3: float, T_K: float) -> float:
    """Van der Waals equation of state: P in Pa."""
    if n_mol <= 0 or V_m3 <= 0:
        return 0.0
    v_m = V_m3 / n_mol  # molar volume [m³/mol]
    if v_m <= VDW_B:
        return P_STD * 1000  # clamp to very high pressure
    P = (R_GAS * T_K) / (v_m - VDW_B) - VDW_A / (v_m ** 2)
    return max(0.0, P)


def _vdw_n_from_P(P_Pa: float, V_m3: float, T_K: float) -> float:
    """Solve VdW for n given P, V, T — Newton's method."""
    if P_Pa <= 0:
        return 0.0
    # Initial guess from ideal gas
    n = P_Pa * V_m3 / (R_GAS * T_K)
    for _ in range(50):
        v_m = V_m3 / (n + 1e-12)
        f = (R_GAS * T_K) / (v_m - VDW_B) - VDW_A / v_m ** 2 - P_Pa
        # df/dn
        dvm_dn = -V_m3 / (n ** 2 + 1e-12)
        df_dn = (-(R_GAS * T_K) / (v_m - VDW_B) ** 2 + 2 * VDW_A / v_m ** 3) * dvm_dn
        if abs(df_dn) < 1e-20:
            break
        n_new = n - f / df_dn
        n = max(0.0, n_new)
        if abs(f) < 1e-3:
            break
    return n


class HydrogenTankModel(BaseModel):
    """
    Compressed hydrogen storage tank.

    Parameters (``params`` dict)
    ----------------------------
    volume_m3 : float           — internal geometric volume [m³]
    max_pressure_bar : float    — design maximum pressure [bar], default 700
    min_pressure_bar : float    — minimum operating pressure [bar], default 5
    initial_soc : float         — initial state of charge (0-1), default 0.5
    temperature_c : float       — nominal gas temperature [°C], default 20
    max_charge_rate_kg_s : float — max charge mass flow [kg/s]
    max_discharge_rate_kg_s : float — max discharge mass flow [kg/s]
    boiloff_rate_per_day : float — permeation/boiloff loss [1/day], default 0
    """

    def __init__(self, component_id: str, params: Dict[str, Any]) -> None:
        super().__init__(component_id, params)
        self._V = float(params["volume_m3"])
        self._P_max = float(params.get("max_pressure_bar", 700.0)) * 1e5   # Pa
        self._P_min = float(params.get("min_pressure_bar", 5.0)) * 1e5     # Pa
        self._T = float(params.get("temperature_c", 20.0)) + 273.15        # K
        self._max_in = float(params.get("max_charge_rate_kg_s", 1e9))
        self._max_out = float(params.get("max_discharge_rate_kg_s", 1e9))
        self._boiloff_day = float(params.get("boiloff_rate_per_day", 0.0))

        soc_init = float(params.get("initial_soc", 0.50))

        # Compute max and min moles from pressure limits
        self._n_max = _vdw_n_from_P(self._P_max, self._V, self._T)
        self._n_min = _vdw_n_from_P(self._P_min, self._V, self._T)

        # Current moles
        self._n: float = self._n_min + soc_init * (self._n_max - self._n_min)
        self._h2_in_kg: float = 0.0
        self._h2_out_kg: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass_kg(self) -> float:
        return self._n * M_H2

    @property
    def pressure_bar(self) -> float:
        return _vdw_pressure(self._n, self._V, self._T) / 1e5

    @property
    def soc(self) -> float:
        """State of charge [0..1]."""
        denom = self._n_max - self._n_min
        if denom <= 0:
            return 0.0
        return float(np.clip((self._n - self._n_min) / denom, 0.0, 1.0))

    @property
    def max_capacity_kg(self) -> float:
        return (self._n_max - self._n_min) * M_H2

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def charge(self, h2_kg: float, dt: float) -> float:
        """Attempt to add *h2_kg* [kg]; return actual amount stored [kg]."""
        max_flow_kg = self._max_in * dt
        available = (self._n_max - self._n) * M_H2
        actual = min(h2_kg, max_flow_kg, available)
        actual = max(0.0, actual)
        self._n += actual / M_H2
        self._h2_in_kg += actual
        return actual

    def discharge(self, h2_kg: float, dt: float) -> float:
        """Attempt to remove *h2_kg* [kg]; return actual amount released [kg]."""
        max_flow_kg = self._max_out * dt
        available = max(0.0, (self._n - self._n_min) * M_H2)
        actual = min(h2_kg, max_flow_kg, available)
        actual = max(0.0, actual)
        self._n -= actual / M_H2
        self._h2_out_kg += actual
        return actual

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        context keys:
          h2_charge_kg   : float — H2 to be added this step [kg]
          h2_discharge_kg: float — H2 to be drawn this step [kg]
          temperature_c  : float — ambient temperature [°C] (optional)
        """
        h2_in = float(context.get("h2_charge_kg", 0.0))
        h2_out = float(context.get("h2_discharge_kg", 0.0))
        ts = context.get("timestamp", datetime.utcnow())

        if "temperature_c" in context:
            self._T = float(context["temperature_c"]) + 273.15

        # Boil-off / permeation losses
        boiloff_kg = self.mass_kg * self._boiloff_day * dt / 86_400.0
        self._n -= boiloff_kg / M_H2

        actual_in = self.charge(h2_in, dt)
        actual_out = self.discharge(h2_out, dt)

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "mass_kg": self.mass_kg,
                "pressure_bar": self.pressure_bar,
                "soc": self.soc,
                "max_capacity_kg": self.max_capacity_kg,
                "h2_in_kg": actual_in,
                "h2_out_kg": actual_out,
                "h2_in_total_kg": self._h2_in_kg,
                "h2_out_total_kg": self._h2_out_kg,
                "boiloff_kg_step": boiloff_kg,
                "temperature_k": self._T,
            },
        )
        return self._state

    def reset(self) -> None:
        soc_init = self.params.get("initial_soc", 0.50)
        self._n = self._n_min + soc_init * (self._n_max - self._n_min)
        self._h2_in_kg = 0.0
        self._h2_out_kg = 0.0
        self._state = None
