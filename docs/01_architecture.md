# System Architecture — HyTwin

## 1. Overview

HyTwin is structured as a stack of layers, each adding abstraction and
intelligence on top of the previous one. Since the introduction of the
**network layer**, the stack is best read as single-site building blocks
(layers 1–3) topped by a multi-site orchestration layer (layer 4) and a
control/RL layer that can target either a single site or the whole network:

```
┌───────────────────────────────────────────────────────────┐
│  Layer 5 — Control & RL                                    │
│  ClassicalController / RLController        (single site)   │
│  NetworkController / NetworkRLController   (multi-site)     │
│  H2GridEnv / AdvancedH2GridEnv (single) · NetworkRLEnv (N)  │
├───────────────────────────────────────────────────────────┤
│  Layer 4 — Network                                          │
│  NetworkTopology (SiteSpec/LinkSpec) · NetworkTwin           │
│  NetworkState · dispatch (merit-order) · compare.py          │
├───────────────────────────────────────────────────────────┤
│  Layer 3 — Digital Twin                                     │
│  TwinNode (model-sensor fusion) · GridTwin (site aggregation)│
├───────────────────────────────────────────────────────────┤
│  Layer 2 — Virtual Sensors                                  │
│  BaseSensor pipeline · concrete sensor types · SensorManager │
├───────────────────────────────────────────────────────────┤
│  Layer 1 — Physical Models                                  │
│  WindTurbine · PV · Electrolyzer · FuelCell · HydrogenTank   │
│  EnergyLoad · GridConnection · EnergyCost                    │
│  ElectricLine · H2Pipeline  (inter-site links)               │
├───────────────────────────────────────────────────────────┤
│  Core Infrastructure                                        │
│  EventBus · StateManager · SimulationClock                  │
│  WeatherModel / WeatherField · TimeSeriesRecorder            │
└───────────────────────────────────────────────────────────┘
```

A single-site plant is simply a network with one site and zero links — the
`GridTwin` used at each site is unchanged from the original single-site
design, and `NetworkTwin` reuses it verbatim per site.

---

## 2. Package layout

```
hytwin/
├── core/
│   ├── event_bus.py        # Thread-safe publish/subscribe bus
│   ├── state_manager.py    # Key-value store with timestamped history
│   ├── time_engine.py      # SimulationClock (configurable speed)
│   └── registry.py         # Component registry
│
├── models/
│   ├── base_model.py       # BaseModel + ModelState (dataclass)
│   ├── wind_turbine.py     # WindTurbineModel
│   ├── photovoltaic.py     # PhotovoltaicModel
│   ├── electrolyzer.py     # ElectrolyzerModel (PEM)
│   ├── fuel_cell.py        # FuelCellModel (PEM)
│   ├── hydrogen_tank.py    # HydrogenTankModel (Van der Waals)
│   ├── energy_load.py      # EnergyLoadModel
│   ├── grid_connection.py  # GridConnectionModel (import/export, outages)
│   ├── energy_cost.py      # EnergyCostModel (Italian PUN tariff structure)
│   ├── electric_line.py    # ElectricLineModel (inter-site electric link)
│   └── h2_pipeline.py      # H2PipelineModel (inter-site H2 pipeline)
│
├── sensors/
│   ├── base_sensor.py      # BaseSensor, SensorReading, SensorStatus
│   ├── sensors.py          # PowerSensor, PressureSensor, … concrete types
│   └── sensor_manager.py   # SensorManager (registration + per-step update)
│
├── weather/
│   ├── weather_model.py    # WeatherModel: Weibull wind, solar geometry,
│   │                       # AR(1) cloud cover / temperature
│   └── weather_field.py    # WeatherField: couples multiple sites' weather
│                            # (geographic correlation + shared synoptic AR(1))
│
├── digital_twin/
│   ├── twin_node.py        # TwinNode: one component + sensor fusion
│   └── grid_twin.py        # GridTwin: single-site orchestration + GridState
│
├── network/
│   ├── topology.py         # SiteSpec, LinkSpec, NetworkTopology
│   ├── network_twin.py     # NetworkTwin: multi-site orchestrator
│   ├── network_state.py    # NodeState, LinkFlow, NetworkState (dataclasses)
│   ├── dispatch.py         # Inter-site dispatch (L1 greedy merit-order)
│   └── compare.py          # Reproducible controller-comparison harness
│
├── rl/
│   ├── environment.py          # H2GridEnv (single-site, Gymnasium)
│   ├── advanced_environment.py # AdvancedH2GridEnv (adds grid/price/outage)
│   ├── network_environment.py  # NetworkRLEnv + NetworkRewardConfig
│   ├── rewards.py              # RewardConfig + compute_reward() (single-site)
│   ├── trainer.py              # RLTrainer (Stable-Baselines3 wrapper)
│   └── network_trainer.py      # train_network_agent() (network PPO trainer)
│
├── control/
│   ├── classical_controller.py # Cost-aware rule-based dispatch (single site)
│   ├── rl_controller.py        # SB3 policy wrapper (single site)
│   ├── network_controller.py   # NetworkController (network-aware overlay)
│   ├── network_rl_controller.py# NetworkRLController (trained policy wrapper)
│   └── fixed_policy_controller.py
│
├── simulation/
│   ├── engine.py            # SimulationEngine
│   └── scenario.py          # Scenario (loads YAML, single-site or network:)
│
├── data/
│   └── time_series.py       # TimeSeriesRecorder (memory + CSV streaming)
│
├── visualization/
│   └── plotter.py           # Matplotlib dashboard / sensor comparison plots
│
└── dashboard/
    ├── network_app.py       # FastAPI + WebSocket — Network Control Room
    ├── app.py                # Legacy single-site dashboard
    └── static/
        ├── network.html      # Network Control Room UI
        └── index.html        # Legacy single-site UI
```

---

## 3. Data flow in a single network step

```
WeatherField.step(ts)
        │ {site_id: weather_dict}
        ▼
NetworkTwin.step()
   │
   ├─→ [per site]  SiteTwin(=GridTwin).step(dt, weather[site], actions[site])
   │        │
   │        ├─ [per component]  Model.step(dt, context) → ModelState
   │        ├─ TwinNode.update(model_state, sensor_readings) → TwinNodeState
   │        ├─ StateManager.set("site_id.node_id.key", value)
   │        └─→ GridState (site-local KPIs, still a residual balance)
   │
   ├─→ NetworkDispatch.solve(sites, links, actions)
   │        • computes each site's net_balance = gen + FC − load − EL
   │        • routes surplus/deficit across electric & H2 links,
   │          respecting capacity, resistive/compression losses
   │        • uncovered residual → that site's own GridConnectionModel
   │          (import/export) at its own local energy cost / carbon factor
   │
   ├─→ NetworkState (per-node NodeState + per-link LinkFlow + system KPIs)
   │
   ├─→ TimeSeriesRecorder.record(step, network_state, weather)
   │
   └─→ EventBus.publish("network.state", network_state)
```

At single-site scale, this collapses to the original `GridTwin` data flow
(`WeatherModel → GridTwin → [TwinNode × N]`); the network layer only adds the
dispatch step and the outer `NetworkState` aggregation.

---

## 4. Core infrastructure

### 4.1 EventBus

Thread-safe publish/subscribe bus. Components publish events on a *topic*
(e.g. `"sensor.wt1.power"`, `"twin.grid_state"`, `"network.state"`) and other
components subscribe to them.

```python
bus = EventBus()
bus.subscribe("sensor.*", my_callback)
bus.publish(Event(topic="sensor.wt1.power", payload=reading, source="wt1"))
```

### 4.2 StateManager

Key-value store with an unbounded (configurable) time history — default 8640
records = 24 h at 10 s resolution. Keys follow the convention:

```
<node_id>.<variable>          # single-site
<site_id>.<node_id>.<variable> # network (site-qualified)
# examples:
"wt1.power_kw"
"trapani.trapani_wt1.power_kw"
"tk1.pressure_bar"
"el1.h2_flow_kg_h"
```

Key methods:

| Method | Description |
|--------|-------------|
| `set(key, value, ts)` | Update a value with a timestamp |
| `update(mapping, ts)` | Batch update |
| `get(key)` → `(ts, value)` | Timestamped read |
| `snapshot(prefix)` → `dict` | Flat snapshot for sensors |
| `history(key, last=N)` | List of `(ts, value)` |

> **Critical note**: `StateManager.__len__` returns `len(_data)`, so an
> empty instance is falsy in Python. Always use `if obj is not None:`
> rather than `if obj:` when checking whether one exists.

### 4.3 SimulationClock

Manages simulated time:

```python
clock = SimulationClock(start_time=datetime(2024, 6, 15), dt_seconds=600)
ts = clock.tick()   # advances by dt, returns the new timestamp
```

`speed_factor=0.0` runs the simulation at maximum speed (wall-clock is
ignored). Values > 0 slow it down for real-time dashboard playback — this is
the `--speed-factor` flag used by `dashboard.network_app`.

---

## 5. Module dependencies

```
simulation.engine
    └── network.network_twin          (multi-site)
            ├── digital_twin.grid_twin   (one instance per site)
            │       ├── digital_twin.twin_node
            │       │       ├── models.base_model
            │       │       └── sensors.base_sensor
            │       └── models.*
            ├── network.dispatch
            ├── network.topology
            └── core.state_manager

weather.weather_field
    └── weather.weather_model   (one instance per site, correlated)

sensors.sensor_manager
    └── sensors.base_sensor

rl.network_environment
    └── network.network_twin

control.network_controller
    └── control.classical_controller  (per-site policy, reused)

data.time_series       (standalone)
```

---

## 6. SimulationEngine (single-site) and TimeSeriesRecorder

For a single-site scenario, `hytwin/simulation/engine.py`'s
`SimulationEngine` is the orchestrator (for a network scenario, this role
is played by `NetworkTwin`, §3 above). Its `step_once()` cycle:

```
step_once()
├── 1. tick clock: t += dt
├── 2. weather = weather_model.step(t)
├── 3. control_actions = scenario.get_control_actions(step, t, grid_state)
├── 4. sensor_readings = sensor_mgr.update(snapshot, t)     (if sensor_mgr is not None)
├── 5. gs = twin.step(dt, weather, control_actions, sensor_readings, t)
├── 6. _update_h2_storage(gs)   — charges/discharges tanks from EL/FC flow
├── 7. recorder.record(step, gs, weather)                    (if recorder is not None)
├── 8. runs registered step callbacks
└── 9. returns (gs, ctx)
```

> **Guard `is not None`**: `sensor_mgr` and `recorder` are optional
> components. The guard must be `if obj is not None:`, never `if obj:` —
> both classes implement `__len__` and are falsy when empty (0 records,
> 0 sensors), which silently skips their logic if checked the wrong way.

`TimeSeriesRecorder` (`hytwin/data/time_series.py`) records every
`GridState` field to an in-memory list (optionally streamed to CSV), and
exposes `to_dataframe()` (pandas), `to_json(path)`, and `summary()` — the
aggregate KPI dict (`mean_renewable_fraction`, etc.) used for reporting.

`engine.reset(seed=...)` re-seeds the weather model, clears the
`StateManager` and recorder, and resets every `TwinNode`, so identical
`(config, seed)` pairs reproduce identical runs — the same determinism
guarantee that `compare_controllers` relies on at network scale.

---

## 7. Thread-safety

- `StateManager` uses a `threading.RLock` around all read/write operations.
- `EventBus` has an internal lock protecting its subscriber list.
- Physical models are **not** thread-safe (single-thread by design).
- `SimulationEngine` / `NetworkTwin` are single-thread; parallelism happens
  at the RL episode level via SB3's `DummyVecEnv` / `SubprocVecEnv`, and in
  the dashboard via a background thread for training jobs.
