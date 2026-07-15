# Virtual Sensors — HyTwin

## 1. Introduction

`hytwin/sensors/` implements a **virtual sensor** layer that mimics real
instrumentation: every sensor applies noise, drift, delay, quantisation, and
can generate controlled or probabilistic faults.

This layer exists to:
- Stress-test the model-sensor fusion algorithm in `TwinNode`
- Train RL agents under realistic, imperfect measurements
- Validate alarm thresholds and fault-detection logic (see the dashboard's
  **Events & Alarms** screen, `07_dashboard.md`)

---

## 2. Sensor pipeline architecture

```
StateManager          SensorManager
    │                      │
    │ reads "node.var"     │ registers / updates
    ▼                      ▼
BaseSensor (virtual pipeline)
    │
    ▼  stage 1  → Gaussian noise           (noise_std)
    ▼  stage 2  → calibration drift        (drift_rate, random-walk bias)
    ▼  stage 3  → measurement delay        (FIFO buffer, delay_steps)
    ▼  stage 4  → quantisation             (quantisation_step / ADC LSB)
    ▼  stage 5  → fault injection          (fault_probability, or manual)
    ▼  stage 6  → quality scoring
    │
    ▼
SensorReading
```

### `BaseSensor.measure()` — pipeline in code

```python
def measure(self, true_value: float, timestamp=None) -> SensorReading:
    self._random_fault_check()          # may set self._fault_mode

    if self._fault_mode == FAULT_OFFLINE:
        return SensorReading(value=nan, status=FAULT_OFFLINE, quality=0.0, ...)
    if self._fault_mode == FAULT_STUCK:
        return SensorReading(value=stuck_value, status=FAULT_STUCK, quality=0.1, ...)
    if self._fault_mode == FAULT_SPIKE:
        spike = true_value * (3 + 7 * U(0,1))   # 3-10x outlier, clears after 1 step
        return SensorReading(value=spike, status=FAULT_SPIKE, quality=0.0, ...)

    v = quantise(delay(drift(noise(true_value))))
    quality = max(0, 1 - |v - true_value| / (|true_value| + 1e-9))
    status = OK if quality > 0.9 else DEGRADED
    return SensorReading(value=v, status=status, quality=quality, ...)
```

---

## 3. SensorReading — reading structure

```python
@dataclass
class SensorReading:
    sensor_id: str          # e.g. "trapani_wt1.power"
    sensor_type: str        # e.g. "power", "pressure", "flow"
    value: float            # measured value (with errors / faults applied)
    unit: str                # e.g. "kW", "bar", "kg/h"
    status: SensorStatus     # OK / DEGRADED / FAULT_STUCK / FAULT_SPIKE / FAULT_OFFLINE
    timestamp: datetime
    true_value: Optional[float]  # available only in simulation, for diagnostics
    quality: float           # [0, 1], 1 = perfect
```

---

## 4. SensorStatus — possible states

| Status | Description | Typical quality |
|--------|-------------|------------------|
| `OK` | Sensor operating normally (deviation ≤ 10%) | > 0.9 |
| `DEGRADED` | Reading has drifted from the model estimate but is still numeric | ≤ 0.9 |
| `FAULT_STUCK` | Value frozen at its last (or a fixed) sample | 0.1 |
| `FAULT_SPIKE` | One-step outlier, 3–10× the true value | 0.0 |
| `FAULT_OFFLINE` | No reading available (`NaN`) | 0.0 |

These five states are exactly what the dashboard's Events & Alarms and
Diagnostics screens surface per component (see `07_dashboard.md`).

---

## 5. Quality scoring

For a normally-operating (non-faulted) sensor, the quality score is a
simple relative-error measure against the true value the sensor is
measuring:

$$q = \max\!\left(0,\ 1 - \frac{|v - v_{\text{true}}|}{|v_{\text{true}}| + \epsilon}\right)$$

`OK` is assigned when $q > 0.9$; otherwise the reading is marked
`DEGRADED`. Faulted readings (`FAULT_STUCK`, `FAULT_SPIKE`, `FAULT_OFFLINE`)
receive a fixed low/zero quality (`0.1`/`0.0`/`0.0`) regardless of the raw
value, so they are effectively excluded from — or minimally weighted in —
the `TwinNode` fusion step (see `03_digital_twin.md` §4.1).

---

## 6. Fault model

Each sensor has a single tunable **`fault_probability`**: the probability,
per simulation step, that a random fault is injected. When triggered, the
sensor enters `FAULT_SPIKE` for that reading (a one-step outlier that
self-clears), unless a fault has been explicitly injected via
`inject_fault()` with a different mode:

```python
sensor.inject_fault(SensorStatus.FAULT_STUCK, stuck_value=312.5)
sensor.clear_fault()
```

`FAULT_STUCK` and `FAULT_OFFLINE` persist until explicitly cleared (or, in
the SensorManager's own duration-based helper, for a configured number of
steps); `FAULT_SPIKE` always self-clears after one reading.

Tuning `fault_probability` (per site, per sensor) is the mechanism for
making a scenario noisier or cleaner — see the Configuration screen in
`07_dashboard.md` and the YAML schema in `08_configuration_reference.md`.

---

## 7. Sensor types

| Class | Quantity | Unit | `sensor_type` |
|-------|----------|------|----------------|
| `PowerSensor` | Active power | kW | `power` |
| `VoltageSensor` | Stack voltage | V | `voltage` |
| `CurrentSensor` | Stack current | A | `current` |
| `PressureSensor` | Tank pressure | bar | `pressure` |
| `TemperatureSensor` | Temperature | °C | `temperature` |
| `FlowSensor` | H₂ flow rate | kg/h | `flow` |
| `HydrogenLevelSensor` | H₂ mass in tank | kg | `h2_level` |
| `IrradianceSensor` | Solar irradiance | W/m² | `irradiance` |
| `WindSpeedSensor` | Wind speed | m/s | `wind_speed` |

Each concrete class only overrides the default noise/unit parameters; the
pipeline itself is entirely inherited from `BaseSensor`.

---

## 8. SensorManager — centralised management

### Registering a sensor

```python
sm = SensorManager(state_manager)
sm.register(PowerSensor(
    sensor_id="wt1.power",
    model_key="wt1.power_kw",
    noise_std=2.5,
    drift_rate=0.001,
    fault_probability=0.001,
))
```

`model_key` is the `StateManager` key the sensor reads the true value from
(e.g. `"wt1.power_kw"`, or `"trapani.trapani_wt1.power_kw"` in a network
deployment).

### Per-step update

```python
readings: List[SensorReading] = sm.update(timestamp)
```

Internally, for every registered sensor:
1. Read `true_value = state_manager.get(sensor.model_key, 0.0)`
2. Call `sensor.measure(true_value, timestamp)`
3. Collect the `SensorReading`s into a list

### Direct sensor access

```python
sm.get("wt1.power")   # → SensorReading or None
```

---

## 9. Fault-injection API for testing

```python
sm.inject_fault(sensor_id="tk1.pressure", fault_type=SensorStatus.FAULT_STUCK, duration_steps=20)
sm.clear_fault("tk1.pressure")
```

`FAULT_SPIKE` lasts 1 step; `FAULT_STUCK` and `FAULT_OFFLINE` last for
`duration_steps` steps and then auto-clear.

---

## 10. Sensor → node mapping

Sensors are associated to nodes via the `sensor_id` prefix:

```
sensor_id = "wt1.power"
              └┘
           prefix = "wt1"  →  TwinNode "wt1"
```

`GridTwin.step()` groups the `SensorReading` list by prefix and delivers to
each node only the readings within its own domain.

### Example: sensor configuration in YAML

```yaml
sensors:
  - id: "wt1.power"
    type: "PowerSensor"
    model_key: "wt1.power_kw"
    noise_std: 2.5
    drift_rate: 0.001
    fault_probability: 0.001

  - id: "tk1.pressure"
    type: "PressureSensor"
    model_key: "tk1.pressure_bar"
    noise_std: 1.0
    quantisation_step: 0.5
    fault_probability: 0.002
```

---

## 11. `model_key` convention

For every sensor, `model_key` must correspond **exactly** to a key written
by `GridTwin` into the `StateManager`. Standard keys follow:

```
{node_id}.{variable}          # single-site
{site_id}.{node_id}.{variable}  # network deployment
```

| Node type | Typical keys |
|-----------|----------------|
| Wind turbine | `power_kw`, `wind_speed_hub_ms`, `air_density_kg_m3` |
| Photovoltaic | `power_kw`, `dc_power_kw`, `cell_temp_c` |
| Electrolyzer | `power_kw`, `h2_flow_kg_h`, `efficiency_hhv`, `cell_voltage_v`, `current_a` |
| Fuel cell | `power_kw`, `h2_consumption_kg_h`, `efficiency_lhv`, `voltage_v`, `current_a` |
| Hydrogen tank | `pressure_bar`, `soc`, `h2_mass_kg`, `temperature_k` |
| Load | `power_kw` |
| Electric line | `power_in_kw`, `power_out_kw`, `loss_kw`, `utilization` |
| H2 pipeline | `flow_kg_h`, `compressor_power_kw`, `linepack_kg`, `delivered_kg_h` |

---

## 12. From raw sensor faults to dashboard alarms

The sensor layer only produces per-reading `SensorReading`s with a status
and a quality score. Turning that into a legible operator alarm is the
digital twin's and dashboard's job, and was specifically redesigned to be
actionable (see `07_dashboard.md` §Events & Alarms):

- `TwinNode` already tracks, per step, the **worst-offending** sensor for
  that component (`fault_key`, `fault_status`, `fault_sensor_value`,
  `fault_model_value` on `TwinNodeState`, see `03_digital_twin.md` §2.3).
- The dashboard renders this as a plain-English message naming the specific
  site + device id + device kind and the fault kind — "outlier spike",
  "stuck reading (not updating)", "sensor offline (no data)", "reading
  drifted from model estimate" — with the measured vs. model-expected value
  where available, instead of a generic "anomaly detected".

---

## 13. Safety and validation notes

- Sensors never **write** to the `StateManager`; reads always flow
  one-way, model → sensor.
- The raw sensor value is never exposed directly to the RL estimator or
  controller: it always passes through `TwinNode` fusion first.
- Faults injected via `inject_fault` are logged at `INFO` level for
  traceability during system tests.
