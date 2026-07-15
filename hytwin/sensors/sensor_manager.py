"""
Sensor Manager
==============
Central registry and update coordinator for all virtual sensors.
Reads model outputs, routes values to the correct sensors, and
publishes readings to the event bus each simulation step.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.event_bus import EventBus
from .base_sensor import BaseSensor, SensorReading

logger = logging.getLogger(__name__)


class SensorManager:
    """
    Manages a collection of virtual sensors.

    Usage::

        mgr = SensorManager(bus)
        mgr.register(power_sensor, model_key="wind_turbine.wt1.power_kw")
        # each simulation step:
        readings = mgr.update(state_snapshot, timestamp)
    """

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._bus = event_bus
        # Maps sensor_id → (sensor, model_state_key)
        self._sensors: Dict[str, tuple[BaseSensor, str]] = {}

    def register(self, sensor: BaseSensor, model_key: str) -> None:
        """
        Bind *sensor* to a state-manager key.

        Parameters
        ----------
        sensor : BaseSensor
        model_key : str  — key in the flat state snapshot produced by
                           ``StateManager.snapshot()`` whose value is
                           the true physical measurement.
        """
        self._sensors[sensor.sensor_id] = (sensor, model_key)
        if self._bus is not None and sensor._bus is None:
            sensor._bus = self._bus
        logger.debug(
            "Sensor '%s' bound to model key '%s'", sensor.sensor_id, model_key
        )

    def unregister(self, sensor_id: str) -> None:
        self._sensors.pop(sensor_id, None)

    def update(
        self,
        state_snapshot: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> List[SensorReading]:
        """
        Advance all sensors by one step.

        Parameters
        ----------
        state_snapshot : dict
            Flat dict of model state key → true value
            (as returned by ``StateManager.snapshot()``).
        timestamp : datetime, optional

        Returns
        -------
        list of SensorReading
        """
        ts = timestamp or datetime.utcnow()
        readings: List[SensorReading] = []

        for sensor_id, (sensor, key) in self._sensors.items():
            true_value = state_snapshot.get(key)
            if true_value is None:
                logger.debug("No value for sensor '%s' (key='%s')", sensor_id, key)
                continue
            try:
                reading = sensor.measure(float(true_value), timestamp=ts)
                readings.append(reading)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error in sensor '%s': %s", sensor_id, exc)

        return readings

    def get_sensor(self, sensor_id: str) -> Optional[BaseSensor]:
        entry = self._sensors.get(sensor_id)
        return entry[0] if entry else None

    def all_readings(self) -> List[SensorReading]:
        return [
            s.last_reading
            for s, _ in self._sensors.values()
            if s.last_reading is not None
        ]

    def reset_all(self) -> None:
        for sensor, _ in self._sensors.values():
            sensor.reset()

    def __len__(self) -> int:
        return len(self._sensors)

    def __repr__(self) -> str:
        return f"SensorManager({len(self._sensors)} sensors)"
