# Digital Twin Architecture — HyTwin 2.0

## 1. What is a Digital Twin

A **Digital Twin** is a continuously-updated virtual representation of a
real physical asset. In HyTwin 2.0 the digital twin is structured in three
hierarchical levels:

1. **`TwinNode`** — the digital twin of a single component (turbine,
   electrolyzer, tank, …)
2. **`GridTwin`** (also referred to as a **site twin**) — orchestrates all
   the `TwinNode`s of one site and produces that site's aggregated state
   (`GridState`)
3. **`NetworkTwin`** — orchestrates multiple `GridTwin` instances (one per
   geographic site) plus the inter-site links, producing the network-wide
   `NetworkState` (see `05_network_layer.md`)

```
NetworkTwin                              ← multi-site orchestrator
 ├── GridTwin "trapani"
 │     ├── TwinNode "trapani_wt1" ← WindTurbineModel  + sensors
 │     ├── TwinNode "trapani_pv1" ← PhotovoltaicModel + sensors
 │     ├── TwinNode "trapani_el1" ← ElectrolyzerModel + sensors
 │     ├── TwinNode "trapani_tk1" ← HydrogenTankModel + sensors
 │     └── TwinNode "trapani_load"← EnergyLoadModel   + sensors
 ├── GridTwin "napoli"   (same structure, different equipment mix)
 ├── GridTwin "roma"
 └── …
```

A single-site deployment is simply a `NetworkTwin` with one site and no
links — `GridTwin` itself is unmodified from the original single-site
design.

---

## 2. TwinNode — single-component twin

### 2.1 Internal structure

```
TwinNode
 ├── model: BaseModel               ← physical model
 ├── _keys: List[str]               ← state keys of interest
 ├── _anomaly_threshold: float      ← relative deviation threshold
 ├── _current_state: TwinNodeState
 └── _history: List[TwinNodeState]  ← up to 1440 steps
```

### 2.2 Update cycle

Each time step:

```
call: node.update(model_state, sensor_readings)
         │
         ├─ model_vals = model_state.values (dict str→float)
         │
         ├─ [for each key k in model_vals]
         │     find a matching sensor reading
         │       (matching: sensor_type ⊂ key name)
         │
         │   if reading is OK:
         │     fused[k] = w_sensor × sensor_val + (1-w_sensor) × model_val
         │     w_sensor = reading.quality  ∈ [0,1]
         │     rel_err  = |sensor_val - model_val| / (|model_val| + 1e-6)
         │
         │   if reading is FAULT_* or missing:
         │     fused[k] = model_val   (the model is the fallback ground truth)
         │     rel_err  = 1.0         (worst-case, by definition)
         │
         ├─ anomaly_score = clip(max(rel_err) / anomaly_threshold, 0, 1)
         ├─ health_score = 1 - anomaly_score
         │
         └─ returns TwinNodeState (incl. worst-offender diagnostic fields)
```

### 2.3 TwinNodeState

```python
@dataclass
class TwinNodeState:
    node_id: str
    timestamp: datetime
    model_values: Dict[str, float]     # raw model output
    fused_values: Dict[str, float]     # model+sensor fused state
    anomaly_score: float               # 0 = normal, 1 = maximal anomaly
    health_score: float                # 1 - anomaly_score
    sensor_quality: float              # average quality of OK sensors
    # Diagnostic detail for the worst-offending sensor this step (None if
    # nothing flagged) — lets callers (e.g. the dashboard event log) report
    # *which* measurement is at fault and *why*, not just an aggregate score.
    fault_key: Optional[str]
    fault_status: Optional[str]        # SensorStatus name, e.g. "FAULT_SPIKE"
    fault_sensor_value: Optional[float]
    fault_model_value: Optional[float]
```

### 2.4 Observation for RL

`observe(keys=None)` returns a NumPy vector of the fused values:

```python
obs = node.observe(keys=["pressure_bar", "soc"])
# → array([275.3, 0.52], dtype=float32)
```

If `keys=None`, all fused values are returned in insertion order.

### 2.5 Behaviour under sensor faults

| Sensor status | Fusion | Anomaly contribution |
|----------------|--------|-----------------------|
| `OK` | quality-weighted average | `rel_err = |sensor − model| / |model|` |
| `DEGRADED` | quality-weighted average | same, tracked as a fault candidate if numerically off |
| `FAULT_STUCK` | model value only | `rel_err = 1.0` (worst case) |
| `FAULT_OFFLINE` | model value only | `rel_err = 1.0` |
| `FAULT_SPIKE` | model value only (transient) | `rel_err = 1.0` |

> This is intentional: a broken sensor should not silently vanish from the
> anomaly picture — a fault is, by construction, the worst possible
> disagreement between sensor and model, so it always pushes the anomaly
> score toward its cap rather than being excluded from the calculation.

### 2.6 Anomaly scoring — scale-free relative deviation

Unlike a fixed absolute-unit threshold (e.g. "10 kW off is anomalous"), the
anomaly score is a **relative deviation**: the fraction of the model's
expected value by which the sensor disagrees, clipped by
`anomaly_threshold` (default 0.5, i.e. a 50% relative deviation saturates
the score to 1.0):

$$\text{anomaly\_score} = \text{clip}\!\left(\frac{\max_k |y_k - \hat{x}_k| / (|\hat{x}_k| + \epsilon)}{\theta_{\text{anom}}},\ 0,\ 1\right)$$

This makes the same threshold meaningful whether the sensor reports power in
kW, pressure in bar, or SOC as a 0–1 fraction — a component measured in bar
does not need a different threshold from one measured in kW just because
its numbers are larger.

---

## 3. GridTwin — site-level coordination

### 3.1 Construction

```python
twin = GridTwin(config_dict, event_bus, state_manager)
twin.build()   # instantiates TwinNodes from the YAML parameters
```

`build()` walks the configuration sections `wind_turbines`, `pv_arrays`,
`electrolyzers`, `fuel_cells`, `hydrogen_tanks`, `loads`,
`grid_connections`, and creates one `TwinNode` per element.

### 3.2 Step

```python
grid_state = twin.step(
    dt,                    # step duration [s]
    weather,               # dict from WeatherModel
    control_actions,       # {node_id: {param: val}}
    sensor_readings,       # List[SensorReading] (optional)
    timestamp,             # datetime (optional)
)
```

#### Internal `GridTwin.step()` sequence

```
1. Group sensor_readings by node_id prefix
2. For each node (sequential loop):
   a. Build context = weather + control_actions[node_id]
   b. Call model.step(dt, context) → ModelState
   c. Call node.update(model_state, readings) → TwinNodeState
   d. Write to StateManager: "node_id.key" → value
   e. Update aggregate KPIs (wind_kw, pv_kw, el_kw, …)
3. Compute GridState with all aggregated KPIs
4. Publish on EventBus: topic "twin.grid_state"
5. Return GridState
```

> **Architectural note**: the loop is **sequential** (single-pass). H₂ flow
> between electrolyzer and tank is **not** automatically coordinated inside
> `GridTwin.step()` — the caller (`SimulationEngine` for single-site, or the
> network dispatch step for multi-site) handles H₂ storage updates
> separately after the step.

### 3.3 GridState — site-level state

```python
@dataclass
class GridState:
    timestamp: datetime
    wind_power_kw: float          # sum of wind turbines
    pv_power_kw: float            # sum of PV arrays
    electrolyzer_power_kw: float  # sum of electrolyzers
    fuel_cell_power_kw: float     # sum of fuel cells
    load_kw: float                # total load
    grid_exchange_kw: float       # import(+) / export(-) from grid
    h2_production_kg_h: float
    h2_consumption_kg_h: float
    h2_storage_kg: float
    h2_soc: float                 # tank SOC [0-1]
    h2_pressure_bar: float
    renewable_fraction: float     # renewable share [0-1]
    grid_self_sufficiency: float  # off-grid autonomy [0-1]
    system_efficiency: float      # average stack efficiency
    overall_health: float         # average node health score
    anomaly_nodes: List[str]      # nodes with anomaly_score > 0.5
```

#### Performance indices

**Renewable fraction**:
$F_{\text{ren}} = \frac{P_{\text{wind}} + P_{\text{PV}}}{P_{\text{load}} + P_{\text{el}} + \epsilon}$

**Self-sufficiency**:
$F_{\text{self}} = 1 - \frac{\max(0, P_{\text{load}} + P_{\text{el}} - P_{\text{ren}} - P_{\text{FC}})}{P_{\text{load}} + P_{\text{el}} + \epsilon}$

**Grid exchange**: $P_{\text{grid}} = (P_{\text{load}} + P_{\text{el}}) - (P_{\text{ren}} + P_{\text{FC}})$
(positive = import, negative = export)

In a network deployment this per-site residual is only the *local* balance
before inter-site dispatch — the network layer (`05_network_layer.md`)
resolves surplus/deficit across links before falling back to each site's
own grid import/export.

### 3.4 StateManager integration

After every model step, `GridTwin` writes each node's variables to the
`StateManager` with the convention `"node_id.variable"`:

```
"wt1.power_kw"          → 487.3
"wt1.wind_speed_hub_ms" → 11.2
"el1.h2_flow_kg_h"      → 5.67
"tk1.pressure_bar"      → 312.5
"tk1.soc"               → 0.58
…
```

`SensorManager` then reads this snapshot to feed the virtual sensors at the
next step.

---

## 4. Model-sensor fusion — mathematical detail

### 4.1 Weighted estimator

For each variable $k$ with model value $\hat{x}_k$ and sensor reading $y_k$
of quality $q \in [0,1]$:

$$\hat{x}_k^{\text{fused}} = q \cdot y_k + (1-q) \cdot \hat{x}_k$$

The weight $q$ is the sensor's quality score, computed in the sensor layer
from the deviation from a rolling mean and from the fault type (see
`04_virtual_sensors.md`).

### 4.2 Anomaly score

See §2.6 above — the score is the worst-case relative deviation across a
node's sensors, scaled by `anomaly_threshold` and clipped to [0, 1].

### 4.3 Limits of the current estimator

The current estimator is a **fixed-weight complementary filter**. It does
not implement a dynamic state observer (e.g. a Kalman filter). Natural
extensions for future versions include:
- Linear Kalman Filter for variables with additive Gaussian noise
- Unscented Kalman Filter for the polarisation curve's non-linearity
- Recursive sensor-quality estimation (online EM)

---

## 5. H₂ flow within a site

```
ElectrolyzerModel  →  h2_kg_step
                          │
                    [H2 storage update — engine or dispatch layer]
                          │
                    HydrogenTankModel.charge(h2_kg_step)
                          │
                    HydrogenTankModel.discharge(h2_demand)
                          │
                    FuelCellModel  ←  h2_available_kg
```

For a single site, `SimulationEngine` manages H₂ flow between components
**after** each complete `GridTwin` step, collecting production from the
electrolyzer (`gs.h2_production_kg_h`) and consumption from the fuel cell
(`gs.h2_consumption_kg_h`). For a network, the same flow happens per site
inside `NetworkTwin.step()`, in addition to inter-site H₂ pipeline transfer.

---

## 6. Operating modes

### Simulation mode

```
SimulationEngine → WeatherModel → GridTwin → [TwinNode × N]
                                           ↕ StateManager
                                 SensorManager → virtual readings
```

The engine orchestrates everything deterministically at every step.

### Live digital-twin mode (target architecture)

In production, real sensors replace the virtual `SensorManager`. The
physical model keeps running in parallel as a **predictive model**, and
fusion is performed against real data:

```
Physical sensors → SensorReading → TwinNode.update(model_state, readings)
                                        ↑
                           Model.step() at every tick
```

`GridTwin` is designed to support both modes without code changes — the
only difference is the source of the `SensorReading`s.

---

## 7. History and time-travel

Every `TwinNode` keeps a history of `TwinNodeState` (default 1440 entries =
24 h at 1-min steps). The method:

```python
node.history(last=60)   # last 60 steps
```

returns the list of previous states, usable for:
- Trend analysis
- Short-term forecasting
- Debugging and explainability

This is what backs the Diagnostics screen's per-component health/anomaly
history in the Network Control Room dashboard.
