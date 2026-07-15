"""
Specialised Virtual Sensors
============================
Concrete sensor implementations for the H2 grid:
  - PowerSensor        (kW, kWh)
  - VoltageSensor      (V)
  - CurrentSensor      (A)
  - PressureSensor     (bar)
  - TemperatureSensor  (°C)
  - FlowSensor         (kg/h or Nm³/h)
  - HydrogenLevelSensor (kg, SOC fraction)
  - IrradianceSensor   (W/m²)
  - WindSpeedSensor    (m/s)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.event_bus import EventBus
from .base_sensor import BaseSensor


class PowerSensor(BaseSensor):
    """Measures electrical power [kW]."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_kw: float = 0.5,
        drift_rate_kw: float = 0.01,
        delay_steps: int = 0,
        quantisation_step: float = 0.1,
        fault_probability: float = 0.0,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="power",
            unit="kW",
            noise_std=noise_std_kw,
            drift_rate=drift_rate_kw,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            fault_probability=fault_probability,
            event_bus=event_bus,
            rng=rng,
        )


class VoltageSensor(BaseSensor):
    """Measures electrical voltage [V]."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_v: float = 0.5,
        drift_rate_v: float = 0.005,
        delay_steps: int = 0,
        quantisation_step: float = 0.01,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="voltage",
            unit="V",
            noise_std=noise_std_v,
            drift_rate=drift_rate_v,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            event_bus=event_bus,
            rng=rng,
        )


class CurrentSensor(BaseSensor):
    """Measures electrical current [A]."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_a: float = 0.2,
        drift_rate_a: float = 0.002,
        delay_steps: int = 0,
        quantisation_step: float = 0.01,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="current",
            unit="A",
            noise_std=noise_std_a,
            drift_rate=drift_rate_a,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            event_bus=event_bus,
            rng=rng,
        )


class PressureSensor(BaseSensor):
    """Measures gas pressure [bar]."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_bar: float = 0.5,
        drift_rate_bar: float = 0.01,
        delay_steps: int = 0,
        quantisation_step: float = 0.1,
        fault_probability: float = 0.0,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="pressure",
            unit="bar",
            noise_std=noise_std_bar,
            drift_rate=drift_rate_bar,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            fault_probability=fault_probability,
            event_bus=event_bus,
            rng=rng,
        )


class TemperatureSensor(BaseSensor):
    """Measures temperature [°C]."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_c: float = 0.2,
        drift_rate_c: float = 0.005,
        delay_steps: int = 0,
        quantisation_step: float = 0.1,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="temperature",
            unit="°C",
            noise_std=noise_std_c,
            drift_rate=drift_rate_c,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            event_bus=event_bus,
            rng=rng,
        )


class FlowSensor(BaseSensor):
    """Measures mass flow [kg/h] (e.g. H2 pipeline flow)."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_kg_h: float = 0.02,
        drift_rate_kg_h: float = 0.001,
        delay_steps: int = 1,
        quantisation_step: float = 0.001,
        fault_probability: float = 0.0,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="flow",
            unit="kg/h",
            noise_std=noise_std_kg_h,
            drift_rate=drift_rate_kg_h,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            fault_probability=fault_probability,
            event_bus=event_bus,
            rng=rng,
        )


class HydrogenLevelSensor(BaseSensor):
    """Measures H2 storage level [kg]."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_kg: float = 0.5,
        drift_rate_kg: float = 0.01,
        delay_steps: int = 0,
        quantisation_step: float = 0.1,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="h2_level",
            unit="kg",
            noise_std=noise_std_kg,
            drift_rate=drift_rate_kg,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            event_bus=event_bus,
            rng=rng,
        )


class IrradianceSensor(BaseSensor):
    """Measures solar irradiance [W/m²]."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_wm2: float = 5.0,
        drift_rate_wm2: float = 0.1,
        delay_steps: int = 0,
        quantisation_step: float = 1.0,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="irradiance",
            unit="W/m²",
            noise_std=noise_std_wm2,
            drift_rate=drift_rate_wm2,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            event_bus=event_bus,
            rng=rng,
        )


class WindSpeedSensor(BaseSensor):
    """Measures wind speed [m/s] (cup anemometer model)."""

    def __init__(
        self,
        sensor_id: str,
        noise_std_ms: float = 0.2,
        drift_rate_ms: float = 0.005,
        delay_steps: int = 0,
        quantisation_step: float = 0.1,
        event_bus: Optional[EventBus] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            sensor_id=sensor_id,
            sensor_type="wind_speed",
            unit="m/s",
            noise_std=noise_std_ms,
            drift_rate=drift_rate_ms,
            delay_steps=delay_steps,
            quantisation_step=quantisation_step,
            event_bus=event_bus,
            rng=rng,
        )
