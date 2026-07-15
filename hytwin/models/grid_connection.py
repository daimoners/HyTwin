"""
Grid Connection Model
=====================
Models the physical connection to the national power grid,
representing the non-renewable portion of the energy mix.

Features
--------
- Capacity-limited import and export
- Stochastic availability (random grid outages, maintenance windows)
- CO₂ emission tracking
- Ramp-rate limiting on grid power changes
- Per-step energy and cost tracking
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .base_model import BaseModel, ModelState
from ..core.rng import resolve_rng


class GridConnectionModel(BaseModel):
    """
    Non-renewable grid connection point.

    Parameters (in ``params`` dict)
    --------------------------------
    max_import_kw : float
        Maximum power that can be drawn from the grid [kW]. Default 800.
    max_export_kw : float
        Maximum power that can be fed back to the grid [kW]. Default 200.
    base_availability : float
        Long-run probability that the grid is up (0–1). Default 0.98.
    outage_mean_h : float
        Mean duration of a single outage [h]. Default 2.0.
    outage_rate_per_day : float
        Expected number of outages per simulated day. Default 0.1.
    ramp_rate_kw_s : float
        Maximum power ramp rate [kW/s]. Default 50.0.
    grid_carbon_factor : float
        CO₂ intensity of grid electricity [kgCO₂/kWh]. Default 0.4.
    """

    def __init__(self, component_id: str, params: Dict[str, Any],
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(component_id, params)
        self._rng = resolve_rng(rng)
        self._max_import = float(params.get("max_import_kw", 800.0))
        self._max_export = float(params.get("max_export_kw", 200.0))
        self._base_avail = float(params.get("base_availability", 0.98))
        self._outage_mean_h = float(params.get("outage_mean_h", 2.0))
        self._outage_rate = float(params.get("outage_rate_per_day", 0.1))
        self._ramp_kw_s = float(params.get("ramp_rate_kw_s", 50.0))
        self._carbon_factor = float(params.get("grid_carbon_factor", 0.4))

        # Mutable state
        self._available: bool = True
        self._outage_remaining_s: float = 0.0
        self._power_kw: float = 0.0          # current actual power (+ import, – export)
        self._energy_imported_kwh: float = 0.0
        self._energy_exported_kwh: float = 0.0
        self._co2_kg_total: float = 0.0

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        Advance by *dt* seconds.

        Required context key
        --------------------
        power_setpoint_kw : float
            Requested grid power.  Positive = import, negative = export.
        """
        ts: datetime = context.get("timestamp", datetime.utcnow())

        # ---- Outage logic ----
        if self._outage_remaining_s > 0:
            self._outage_remaining_s -= dt
            self._available = False
        else:
            self._available = True
            # Stochastic new outage?
            p_outage = self._outage_rate * dt / 86_400.0
            if self._rng.random() < p_outage:
                duration_h = self._rng.exponential(self._outage_mean_h)
                self._outage_remaining_s = max(dt, duration_h * 3600.0)
                self._available = False

        # ---- Power dispatch ----
        requested_kw = float(context.get("power_setpoint_kw", 0.0))

        if not self._available:
            target_kw = 0.0
        else:
            # Clamp to capacity
            target_kw = float(np.clip(requested_kw, -self._max_export, self._max_import))

        # Ramp-rate limiting
        max_delta = self._ramp_kw_s * dt
        target_kw = float(
            np.clip(target_kw, self._power_kw - max_delta, self._power_kw + max_delta)
        )
        self._power_kw = target_kw

        # ---- Cumulative energy ----
        dt_h = dt / 3600.0
        if self._power_kw > 0:
            self._energy_imported_kwh += self._power_kw * dt_h
            self._co2_kg_total += self._power_kw * dt_h * self._carbon_factor
        else:
            self._energy_exported_kwh += abs(self._power_kw) * dt_h

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "power_kw": self._power_kw,
                "available": 1.0 if self._available else 0.0,
                "energy_imported_kwh": self._energy_imported_kwh,
                "energy_exported_kwh": self._energy_exported_kwh,
                "co2_kg_step": max(0.0, self._power_kw) * dt_h * self._carbon_factor,
                "co2_kg_total": self._co2_kg_total,
            },
        )
        return self._state

    def reset(self) -> None:
        self._available = True
        self._outage_remaining_s = 0.0
        self._power_kw = 0.0
        self._energy_imported_kwh = 0.0
        self._energy_exported_kwh = 0.0
        self._co2_kg_total = 0.0
        self._state = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def max_import_kw(self) -> float:
        return self._max_import

    @property
    def max_export_kw(self) -> float:
        return self._max_export
