"""
Hydrogen Pipeline Model
=======================
Physical model of an inter-site hydrogen pipeline transporting H₂ (in gaseous
phase) between two network nodes.

Features
--------
- Capacity-limited mass flow [kg/h]
- **Compression energy**: injecting H₂ into the pipeline costs electricity at
  the sending node (``compressor_spec_kwh_per_kg``) — reported as an electrical
  load the dispatch must cover.
- **Transport delay**: gas injected now arrives at the far end after a delay
  proportional to line length (``length_km`` / ``transport_velocity_ms``).
- **Line-pack**: the mass currently in transit acts as buffered storage,
  capped at ``line_pack_capacity_kg``; injection is throttled if the buffer
  would overflow.

Sign convention (F2)
--------------------
The pipeline is modelled as **unidirectional** (producer → consumer,
``from`` → ``to``).  ``flow_request_kg_h`` is clamped to ``[0, max_flow_kg_h]``
unless ``bidirectional`` is set.  Reverse flow with an in-transit buffer is a
later refinement; for now a negative request yields zero flow.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict

import numpy as np

from .base_model import BaseModel, ModelState


class H2PipelineModel(BaseModel):
    """
    Inter-site hydrogen pipeline.

    Parameters (in ``params`` dict)
    --------------------------------
    max_flow_kg_h : float
        Maximum injection flow [kg/h]. Default 300.
    length_km : float
        Pipeline length [km]. Default 100.  Usually auto-filled from geodata.
    diameter_m : float
        Informative only (not used in the F2 flow physics). Default 0.4.
    compressor_spec_kwh_per_kg : float
        Electric energy to compress/inject 1 kg of H₂ [kWh/kg]. Default 2.0.
    line_pack_capacity_kg : float
        Max mass buffered in the pipeline (line-pack). Default 800.
    transport_velocity_ms : float
        Effective gas transport speed [m/s], sets the delay. Default 15.
    bidirectional : bool
        Reserved (F2 treats the pipe as unidirectional). Default False.
    """

    def __init__(self, component_id: str, params: Dict[str, Any]) -> None:
        super().__init__(component_id, params)
        self._max_flow_kg_h = float(params.get("max_flow_kg_h", 300.0))
        self._length_km = float(params.get("length_km", 100.0))
        self._diameter_m = float(params.get("diameter_m", 0.4))
        self._compressor_spec = float(params.get("compressor_spec_kwh_per_kg", 2.0))
        self._linepack_cap = float(params.get("line_pack_capacity_kg", 800.0))
        self._velocity_ms = float(params.get("transport_velocity_ms", 15.0))
        self._bidirectional = bool(params.get("bidirectional", False))

        # Transport delay [s]; converted to whole steps at step time.
        self._delay_s = (self._length_km * 1000.0) / max(1e-6, self._velocity_ms)

        # Mutable state — FIFO of in-transit mass chunks [kg], one per step.
        self._pipeline: Deque[float] = deque()
        self._linepack_kg: float = 0.0
        self._h2_delivered_total_kg: float = 0.0
        self._compressor_energy_kwh: float = 0.0

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        Advance by *dt* seconds.

        Context keys
        ------------
        flow_request_kg_h : float
            Requested injection flow at the sending node. Default 0.
        """
        ts: datetime = context.get("timestamp", datetime.utcnow())
        dt_h = dt / 3600.0
        delay_steps = max(0, int(round(self._delay_s / max(1e-9, dt))))

        # ---- Requested injection ----
        req = float(context.get("flow_request_kg_h", 0.0))
        if not self._bidirectional and req < 0.0:
            req = 0.0
        req = float(np.clip(abs(req), 0.0, self._max_flow_kg_h)) * (1.0 if req >= 0 else -1.0)

        inject_kg = req * dt_h  # mass injected this step

        # ---- Line-pack overflow throttling ----
        headroom = max(0.0, self._linepack_cap - self._linepack_kg)
        if inject_kg > headroom:
            inject_kg = headroom
        actual_flow_kg_h = inject_kg / dt_h if dt_h > 0 else 0.0

        # ---- Advance the transport buffer ----
        self._pipeline.append(inject_kg)
        self._linepack_kg += inject_kg

        if len(self._pipeline) > delay_steps:
            delivered_kg = self._pipeline.popleft()
        else:
            delivered_kg = 0.0
        self._linepack_kg = max(0.0, self._linepack_kg - delivered_kg)
        delivered_kg_h = delivered_kg / dt_h if dt_h > 0 else 0.0

        # ---- Compression energy (electric load at sending node) ----
        compressor_power_kw = actual_flow_kg_h * self._compressor_spec  # kWh/kg × kg/h = kW
        self._compressor_energy_kwh += compressor_power_kw * dt_h
        self._h2_delivered_total_kg += delivered_kg

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "flow_kg_h": actual_flow_kg_h,          # injected at source
                "delivered_kg_h": delivered_kg_h,       # arriving at dest (delayed)
                "compressor_power_kw": compressor_power_kw,
                "linepack_kg": self._linepack_kg,
                "utilization": abs(actual_flow_kg_h) / (self._max_flow_kg_h + 1e-9),
                "delivered_total_kg": self._h2_delivered_total_kg,
                "compressor_energy_kwh": self._compressor_energy_kwh,
            },
        )
        return self._state

    def reset(self) -> None:
        self._pipeline.clear()
        self._linepack_kg = 0.0
        self._h2_delivered_total_kg = 0.0
        self._compressor_energy_kwh = 0.0
        self._state = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_flow_kg_h(self) -> float:
        return self._max_flow_kg_h

    @property
    def delay_s(self) -> float:
        return self._delay_s

    @property
    def linepack_kg(self) -> float:
        return self._linepack_kg
