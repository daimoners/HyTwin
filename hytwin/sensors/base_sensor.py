"""
Base Virtual Sensor
===================
Abstract base class for all virtual sensors.

A virtual sensor wraps a physical model output and simulates:
  1. Measurement noise (additive Gaussian)
  2. Calibration drift (slow random walk in bias)
  3. Signal delay (FIFO buffer)
  4. Quantisation (ADC resolution)
  5. Sensor failure modes: stuck value, outlier spike, disconnect
  6. Engineering-unit conversion

All sensors report readings through the EventBus in a standardised
SensorReading dataclass so the digital twin can subscribe generically.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Deque, Dict, Optional

import numpy as np

from ..core.event_bus import Event, EventBus
from ..core.rng import resolve_rng

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

class SensorStatus(Enum):
    OK = auto()
    DEGRADED = auto()       # noise/drift above threshold
    FAULT_STUCK = auto()    # values no longer change
    FAULT_SPIKE = auto()    # transient outlier injected
    FAULT_OFFLINE = auto()  # no reading available


@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    status: SensorStatus
    timestamp: datetime
    true_value: Optional[float] = None   # available in simulation only
    quality: float = 1.0                 # 0 (invalid) .. 1 (perfect)


# ------------------------------------------------------------------
# Base sensor
# ------------------------------------------------------------------

class BaseSensor:
    """
    Virtual sensor base.

    Parameters
    ----------
    sensor_id : str
    sensor_type : str         — e.g. 'power', 'pressure', 'temperature'
    unit : str                — engineering unit string, e.g. 'kW'
    noise_std : float         — 1-sigma Gaussian noise (same unit as value)
    drift_rate : float        — drift [unit/step] standard deviation per step
    delay_steps : int         — number of steps measurement is delayed
    quantisation_step : float — ADC resolution; 0 means no quantisation
    fault_probability : float — probability of random fault injection per step
    event_bus : EventBus      — bus to publish readings on
    topic : str               — event topic (default 'sensor.<sensor_id>')
    """

    def __init__(
        self,
        sensor_id: str,
        sensor_type: str,
        unit: str,
        noise_std: float = 0.0,
        drift_rate: float = 0.0,
        delay_steps: int = 0,
        quantisation_step: float = 0.0,
        fault_probability: float = 0.0,
        event_bus: Optional[EventBus] = None,
        topic: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.unit = unit
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        self.delay_steps = delay_steps
        self.quantisation_step = quantisation_step
        self.fault_probability = fault_probability
        self._bus = event_bus
        self.topic = topic or f"sensor.{sensor_id}"
        self._rng = resolve_rng(rng)

        # Internal state
        self._bias: float = 0.0            # accumulated drift bias
        self._buffer: Deque[float] = deque(maxlen=max(1, delay_steps + 1))
        self._last_true: float = 0.0
        self._last_reading: Optional[SensorReading] = None
        self._stuck_value: Optional[float] = None
        self._fault_mode: Optional[SensorStatus] = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    def inject_fault(self, fault: SensorStatus, stuck_value: Optional[float] = None) -> None:
        self._fault_mode = fault
        if fault == SensorStatus.FAULT_STUCK:
            self._stuck_value = stuck_value if stuck_value is not None else self._last_true
        logger.warning("Fault injected on sensor %s: %s", self.sensor_id, fault)

    def clear_fault(self) -> None:
        self._fault_mode = None
        self._stuck_value = None

    # ------------------------------------------------------------------
    # Measurement pipeline
    # ------------------------------------------------------------------

    def _apply_noise(self, value: float) -> float:
        if self.noise_std > 0:
            return value + self._rng.normal(0.0, self.noise_std)
        return value

    def _apply_drift(self, value: float) -> float:
        if self.drift_rate > 0:
            self._bias += self._rng.normal(0.0, self.drift_rate)
        return value + self._bias

    def _apply_quantisation(self, value: float) -> float:
        if self.quantisation_step > 0:
            return round(value / self.quantisation_step) * self.quantisation_step
        return value

    def _apply_delay(self, value: float) -> float:
        self._buffer.append(value)
        if len(self._buffer) <= self.delay_steps:
            return self._buffer[0]
        return self._buffer[0] if self.delay_steps > 0 else value

    def _random_fault_check(self) -> None:
        if self._fault_mode is None and self.fault_probability > 0:
            if self._rng.random() < self.fault_probability:
                self._fault_mode = SensorStatus.FAULT_SPIKE

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def measure(self, true_value: float, timestamp: Optional[datetime] = None) -> SensorReading:
        """
        Apply the full sensor pipeline to *true_value* and return a
        ``SensorReading``.  Optionally publishes to the EventBus.
        """
        ts = timestamp or datetime.utcnow()
        self._last_true = true_value
        self._step_count += 1

        # Random fault injection
        self._random_fault_check()

        # Handle fault modes
        if self._fault_mode == SensorStatus.FAULT_OFFLINE:
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                value=float("nan"),
                unit=self.unit,
                status=SensorStatus.FAULT_OFFLINE,
                timestamp=ts,
                true_value=true_value,
                quality=0.0,
            )
        elif self._fault_mode == SensorStatus.FAULT_STUCK:
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                value=self._stuck_value or 0.0,
                unit=self.unit,
                status=SensorStatus.FAULT_STUCK,
                timestamp=ts,
                true_value=true_value,
                quality=0.1,
            )
        elif self._fault_mode == SensorStatus.FAULT_SPIKE:
            # Transient outlier — 3-10× normal reading
            spike_value = true_value * (3.0 + 7.0 * self._rng.random())
            self._fault_mode = None   # spike clears after one step
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                value=spike_value,
                unit=self.unit,
                status=SensorStatus.FAULT_SPIKE,
                timestamp=ts,
                true_value=true_value,
                quality=0.0,
            )
        else:
            # Normal measurement chain
            v = self._apply_noise(true_value)
            v = self._apply_drift(v)
            v = self._apply_delay(v)
            v = self._apply_quantisation(v)

            abs_err = abs(v - true_value)
            quality = max(0.0, 1.0 - abs_err / (abs(true_value) + 1e-9))
            status = SensorStatus.OK if quality > 0.9 else SensorStatus.DEGRADED

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                value=v,
                unit=self.unit,
                status=status,
                timestamp=ts,
                true_value=true_value,
                quality=quality,
            )

        self._last_reading = reading

        # Publish to event bus
        if self._bus is not None:
            self._bus.publish(Event(
                topic=self.topic,
                payload=reading,
                source=self.sensor_id,
                timestamp=ts,
            ))

        return reading

    @property
    def last_reading(self) -> Optional[SensorReading]:
        return self._last_reading

    def reset(self) -> None:
        self._bias = 0.0
        self._buffer.clear()
        self._last_true = 0.0
        self._last_reading = None
        self._stuck_value = None
        self._fault_mode = None
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.sensor_id!r}, "
            f"noise={self.noise_std}, drift={self.drift_rate})"
        )
