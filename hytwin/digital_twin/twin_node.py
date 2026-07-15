"""
Digital Twin Node
=================
A ``TwinNode`` represents the digital counterpart of one physical
component (wind turbine, electrolyzer, etc.).  It:

  1. Holds the physics model
  2. Fuses sensor readings with model predictions via a state estimator
  3. Maintains an anomaly score
  4. Exposes a ``observe()`` method for the RL agent

The twin node is intentionally decoupled from the simulation engine so
it can be used equally in *simulation mode* (where a model drives state)
and in *live mode* (where sensor stream drives state).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.base_model import BaseModel, ModelState
from ..sensors.base_sensor import SensorReading, SensorStatus

logger = logging.getLogger(__name__)


@dataclass
class TwinNodeState:
    """Fused state of a twin node at one instant."""
    node_id: str
    timestamp: datetime
    model_values: Dict[str, float] = field(default_factory=dict)
    fused_values: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    health_score: float = 1.0          # 0 = broken, 1 = perfect
    sensor_quality: float = 1.0
    # Diagnostic detail for the worst-offending sensor this step (None if none
    # flagged) — lets callers (e.g. the dashboard event log) report *which*
    # measurement is at fault and *why*, instead of a bare aggregate score.
    fault_key: Optional[str] = None
    fault_status: Optional[str] = None   # SensorStatus name, e.g. "FAULT_SPIKE"
    fault_sensor_value: Optional[float] = None
    fault_model_value: Optional[float] = None


class TwinNode:
    """
    Digital twin counterpart for a single grid component.

    Parameters
    ----------
    node_id : str
    model : BaseModel
    state_keys : list of str
        Which model output keys are associated with this node.
    anomaly_threshold : float
        Relative sensor/model deviation (fraction of the model's expected
        value) above which the anomaly score saturates to 1.0. Scale-free,
        so it applies uniformly regardless of the component's engineering
        unit (kW, bar, SOC fraction, ...).
    """

    def __init__(
        self,
        node_id: str,
        model: BaseModel,
        state_keys: Optional[List[str]] = None,
        anomaly_threshold: float = 0.5,
    ) -> None:
        self.node_id = node_id
        self.model = model
        self._keys = state_keys or []
        self._anomaly_threshold = anomaly_threshold

        self._current_state: Optional[TwinNodeState] = None
        self._history: List[TwinNodeState] = []
        self._max_history = 1_440   # 24 h at 1-min steps

    # ------------------------------------------------------------------
    # Update cycle
    # ------------------------------------------------------------------

    def update(
        self,
        model_state: ModelState,
        sensor_readings: Optional[List[SensorReading]] = None,
    ) -> TwinNodeState:
        """
        Fuse model prediction with sensor readings.

        Simple fusion: weighted average. If a sensor is faulted, use
        model value 100 %.  If both agree, use average.
        """
        ts = model_state.timestamp
        model_vals = dict(model_state.values)

        fused_vals: Dict[str, float] = {}
        rel_errs: List[float] = []
        sensor_qualities: List[float] = []
        # (relative_error, key, reading, model_value) for every sensor that
        # is explicitly faulted or numerically off — used to report *which*
        # measurement is the worst offender, not just an aggregate score.
        fault_candidates: List[tuple] = []

        if sensor_readings:
            # Build lookup: sensor_type → reading
            reading_map: Dict[str, SensorReading] = {
                r.sensor_type: r for r in sensor_readings
            }
        else:
            reading_map = {}

        for key, model_v in model_vals.items():
            # Try to find a matching sensor reading
            match_reading = None
            for s_type, reading in reading_map.items():
                if s_type in key:
                    match_reading = reading
                    break

            if match_reading is None:
                fused_vals[key] = model_v
                continue

            if match_reading.status in (
                SensorStatus.FAULT_STUCK, SensorStatus.FAULT_SPIKE, SensorStatus.FAULT_OFFLINE,
            ):
                # Known-bad reading: trust the model fully for fusion, but
                # this is by definition the worst possible disagreement —
                # record it explicitly rather than silently dropping it.
                fused_vals[key] = model_v
                rel_errs.append(1.0)
                sensor_qualities.append(match_reading.quality)
                fault_candidates.append((1.0, key, match_reading, model_v))
                continue

            s_v = match_reading.value
            w_s = match_reading.quality
            w_m = 1.0 - w_s
            fused_vals[key] = w_s * s_v + w_m * model_v
            rel_err = abs(s_v - model_v) / (abs(model_v) + 1e-6)
            rel_errs.append(rel_err)
            sensor_qualities.append(match_reading.quality)
            if match_reading.status == SensorStatus.DEGRADED:
                fault_candidates.append((rel_err, key, match_reading, model_v))

        # Anomaly score: worst-case relative deviation across this node's
        # sensors, scale-free so it is comparable across kW/bar/SOC units.
        if rel_errs:
            worst_rel_err = max(rel_errs)
            anomaly = float(np.clip(worst_rel_err / self._anomaly_threshold, 0.0, 1.0))
            avg_quality = float(np.mean(sensor_qualities))
        else:
            anomaly = 0.0
            avg_quality = 1.0

        health = 1.0 - anomaly

        fault_key = fault_status = None
        fault_sensor_value = fault_model_value = None
        if fault_candidates:
            fault_candidates.sort(key=lambda c: c[0], reverse=True)
            _, fault_key, worst_reading, fault_model_value = fault_candidates[0]
            fault_status = worst_reading.status.name
            fault_sensor_value = None if np.isnan(worst_reading.value) else float(worst_reading.value)

        state = TwinNodeState(
            node_id=self.node_id,
            timestamp=ts,
            model_values=model_vals,
            fused_values=fused_vals,
            anomaly_score=anomaly,
            health_score=health,
            sensor_quality=avg_quality,
            fault_key=fault_key,
            fault_status=fault_status,
            fault_sensor_value=fault_sensor_value,
            fault_model_value=fault_model_value,
        )
        self._current_state = state
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return state

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def state(self) -> Optional[TwinNodeState]:
        return self._current_state

    def history(self, last: Optional[int] = None) -> List[TwinNodeState]:
        return self._history[-last:] if last else list(self._history)

    def observe(self, keys: Optional[List[str]] = None) -> np.ndarray:
        """
        Return a numpy observation vector for RL or forecasting.
        Uses fused values; defaults to all numeric values if keys is None.
        """
        if self._current_state is None:
            return np.zeros(len(keys) if keys else 1, dtype=np.float32)
        vals = self._current_state.fused_values
        if keys:
            return np.array([vals.get(k, 0.0) for k in keys], dtype=np.float32)
        return np.array(list(vals.values()), dtype=np.float32)

    def reset(self) -> None:
        self.model.reset()
        self._current_state = None
        self._history.clear()
