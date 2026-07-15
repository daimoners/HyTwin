"""
Electric Line Model
===================
Physical model of an inter-site electrical interconnection (transmission /
distribution line) transporting power between two network nodes.

Features
--------
- Capacity-limited power transfer [MW]
- Distance-dependent resistive losses (loss ∝ length)
- Bidirectional flow (configurable)
- Ramp-rate limiting on flow changes
- Optional stochastic availability (line faults)

Sign convention
---------------
``power_request_kw`` (context) and the reported ``flow_kw`` are **signed**:
positive = power flowing from the ``from`` site to the ``to`` site,
negative = reverse.  ``delivered_kw`` keeps the same sign but its magnitude is
reduced by transmission losses (energy is lost *in transit*, so the receiving
end gets less than the sending end injects).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .base_model import BaseModel, ModelState
from ..core.rng import resolve_rng


class ElectricLineModel(BaseModel):
    """
    Inter-site electric interconnection.

    Parameters (in ``params`` dict)
    --------------------------------
    max_power_mw : float
        Thermal transfer capacity [MW]. Default 50.  (``max_power_kw`` also
        accepted and takes precedence if given.)
    length_km : float
        Line length [km]. Default 100.  Usually auto-filled from site geodata.
    loss_per_1000km : float
        Fractional resistive loss over 1000 km at rated flow. Default 0.06.
    bidirectional : bool
        Allow reverse (to→from) flow. Default True.
    ramp_rate_kw_s : float
        Max change of transferred power [kW/s]. Default large (≈no limit).
    base_availability, outage_mean_h, outage_rate_per_day : float
        Optional stochastic line-fault model (same semantics as the grid
        connection). Outages disabled by default (``outage_rate_per_day``=0).
    """

    def __init__(self, component_id: str, params: Dict[str, Any],
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(component_id, params)
        self._rng = resolve_rng(rng)
        if "max_power_kw" in params:
            self._max_kw = float(params["max_power_kw"])
        else:
            self._max_kw = float(params.get("max_power_mw", 50.0)) * 1000.0
        self._length_km = float(params.get("length_km", 100.0))
        self._loss_per_1000km = float(params.get("loss_per_1000km", 0.06))
        self._bidirectional = bool(params.get("bidirectional", True))
        self._ramp_kw_s = float(params.get("ramp_rate_kw_s", 1.0e9))
        self._outage_rate = float(params.get("outage_rate_per_day", 0.0))
        self._outage_mean_h = float(params.get("outage_mean_h", 2.0))

        # Fractional loss applied to the transferred magnitude.
        self._loss_fraction = float(
            np.clip(self._loss_per_1000km * self._length_km / 1000.0, 0.0, 0.9)
        )

        # Mutable state
        self._flow_kw: float = 0.0
        self._available: bool = True
        self._outage_remaining_s: float = 0.0
        self._energy_loss_kwh: float = 0.0

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        Advance by *dt* seconds.

        Context keys
        ------------
        power_request_kw : float
            Requested transfer (signed: + = from→to). Default 0.
        """
        ts: datetime = context.get("timestamp", datetime.utcnow())

        # ---- Availability / faults ----
        if self._outage_remaining_s > 0:
            self._outage_remaining_s -= dt
            self._available = False
        else:
            self._available = True
            if self._outage_rate > 0:
                p_outage = self._outage_rate * dt / 86_400.0
                if self._rng.random() < p_outage:
                    duration_h = self._rng.exponential(self._outage_mean_h)
                    self._outage_remaining_s = max(dt, duration_h * 3600.0)
                    self._available = False

        # ---- Requested flow ----
        req = float(context.get("power_request_kw", 0.0))
        if not self._bidirectional and req < 0.0:
            req = 0.0
        if not self._available:
            req = 0.0

        # Capacity clamp
        target = float(np.clip(req, -self._max_kw, self._max_kw))

        # Ramp limiting
        max_delta = self._ramp_kw_s * dt
        target = float(np.clip(target, self._flow_kw - max_delta, self._flow_kw + max_delta))
        self._flow_kw = target

        # ---- Losses ----
        magnitude = abs(self._flow_kw)
        loss_kw = magnitude * self._loss_fraction
        delivered_kw = np.sign(self._flow_kw) * (magnitude - loss_kw)

        dt_h = dt / 3600.0
        self._energy_loss_kwh += loss_kw * dt_h

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "flow_kw": self._flow_kw,               # source-side, signed
                "delivered_kw": float(delivered_kw),    # dest-side, signed
                "loss_kw": loss_kw,
                "loss_fraction": self._loss_fraction,
                "utilization": magnitude / (self._max_kw + 1e-9),
                "available": 1.0 if self._available else 0.0,
                "energy_loss_kwh": self._energy_loss_kwh,
            },
        )
        return self._state

    def reset(self) -> None:
        self._flow_kw = 0.0
        self._available = True
        self._outage_remaining_s = 0.0
        self._energy_loss_kwh = 0.0
        self._state = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_power_kw(self) -> float:
        return self._max_kw

    @property
    def loss_fraction(self) -> float:
        return self._loss_fraction

    @property
    def available(self) -> bool:
        return self._available
