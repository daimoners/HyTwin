# Dashboard — Network Control Room — HyTwin

## 1. Overview

The **Network Control Room** is a **FastAPI + WebSocket** real-time web
dashboard that runs, monitors, and controls the full multi-site network
simulation from a browser. It is the only dashboard entry point in HyTwin.

```bash
python -m dashboard --speed-factor 30
# then open http://localhost:8060
```

CLI flags: `--config <path>` (defaults to `config/italy_network_large.yaml`),
`--port`, `--dt`, `--speed-factor`, `--seed`, `--rl-model`.

The simulation is **built but paused** on launch — press **Start** in the
sidebar to start the clock; the sidebar always shows an explicit
RUNNING / PAUSED state.

---

## 2. Sidebar navigation — 9 screens

In order, exactly as they appear in the sidebar:

1. **Overview**
2. **Network & Map**
3. **Analytics & KPIs**
4. **Control & AI**
5. **Scenario Comparison**
6. **AI Training**
7. **Events & Alarms**
8. **Diagnostics**
9. **Configuration**

---

## 3. Overview

KPI tiles with sparklines: self-sufficiency, reliability, renewable
fraction, average H₂ SOC, cost, CO₂, grid import, and inter-node exchange.
This is the at-a-glance operator view.

---

## 4. Network & Map

An animated map of Italy rendered as a coarse **"big pixel" tile grid**:
square SVG cells masking the true Italy / Sicily / Sardinia coastline via a
point-in-polygon (ray-casting) test over traced coastline coordinates. This
gives a professional heatmap/tile-panel look rather than a literal
coastline drawing. The real network's sites and links — from live lat/lon
topology data (`GET /network/topology` / `GET /api/config`) — are plotted on
top of the tile backdrop. Clicking a node opens a live device drawer showing
that site's components and their current state
(`GET /node/{site_id}`, `GET /api/component/{site_id}/{comp_id}`).

---

## 5. Analytics & KPIs

Time-series charts of the KPI history (`GET /api/history`), for trend
analysis over the running simulation.

---

## 6. Control & AI

Switches the **live active controller**:

| Mode | Meaning |
|------|---------|
| `none` | naive baseline — no dispatch intelligence |
| `classical` | `NetworkClassicalController` — rule-based dispatch |
| `rl` | trained PPO agent, if a compatible model exists |

Set via `POST /control`. This screen also shows a live objective-term
breakdown mirroring the `NetworkRewardConfig` decomposition
(`05_network_layer.md` §7), so an operator can see which reward component is
driving the controller's behaviour in real time.

---

## 7. Scenario Comparison

Runs 2–3 controllers (`none`/`classical`/`rl`) side by side on **identical
conditions** (same seed, same weather/price draw) and compares KPIs, in two
modes:

- **Batch comparison** (`POST /api/compare`) — a one-shot run over a chosen
  horizon, equivalent to calling `compare_controllers` from the CLI
  (`05_network_layer.md` §8).
- **Live lockstep comparison** — steps 2–3 simulations forward together in
  real time, streamed over their own WebSocket channel (`/ws/compare`):
  `POST /api/live_compare/start` (body: `strategies: ["classical","rl"]`,
  `seed`), `.../stop`, `.../resume`, `.../speed`, and
  `GET /api/live_compare/status`. Selecting `rl` requires a model
  compatible with the currently active network (validated the same way as
  §9 below) — otherwise the endpoint returns HTTP 400.

---

## 8. AI Training

Launches a **background PPO training job** (`POST /api/train/start`, body:
`timesteps`, `n_steps`, `seed`, `n_envs`, optional `name`) with live
progress and ETA (`GET /api/train/status`), and a **learning-curve chart**:
the smoothed episode-reward trend sourced from SB3's `Monitor` wrapper's
per-episode `info["episode"]` records (see `06_reinforcement_learning.md`
§7 on why the *trend*, not the absolute value, is the signal to read).
`POST /api/train/stop` aborts a running job.

During training, a `best_model.zip` is saved automatically alongside the
timestamped checkpoint whenever the rolling 10-episode mean reward improves
— this is always the highest-reward checkpoint, not just the most recent
one.

Once a job completes, the freshly trained model can be **activated as the
live `rl` controller without restarting the dashboard**
(`GET /api/models` to list candidates, `POST /api/models/select` to
activate one). Model/topology compatibility is checked before activation:
`probe_model_dims()` reads the candidate model's expected observation
dimensionality and compares it against the currently active network's
`expected_obs_dim()`; a mismatch returns **HTTP 409** with a message naming
how many sites the model was trained for vs. how many the active network
has, preventing a shape-mismatch crash from ever reaching the live
controller.

### Model library

Each row in the model library shows:
- Model name (filename stem) and **number of training steps** (read directly
  from the SB3 zip metadata, so it works even for models trained via the
  CLI without a sidecar JSON file).
- Activation button (disabled for the currently active model or for
  incompatible topologies, with a tooltip explaining why).
- **🗑 Delete button** — opens a confirmation dialog naming the model, then
  calls `POST /api/models/delete` which removes the `.zip` and its `.json`
  sidecar. If the deleted model is currently active, the controller falls
  back to the default path.

Auto-discovery (`--rl-model` auto-mode) picks the model with the **most
training steps** from the model directory, not the most recently modified
file.

For a real (long) training run, prefer the CLI (`06_reinforcement_learning.md`
§4) — a dashboard-launched job is tied to the server process but not to any
particular browser tab, but a CLI run is simpler to leave unattended for
hours.

---

## 9. Events & Alarms

The redesigned event log. Each event names the **specific worst-offending
component** — site + device id + device kind — rather than a generic
"anomaly detected", using the plain-English fault-kind vocabulary from
`04_virtual_sensors.md` §4:

| Internal status | Displayed as |
|-------------------|----------------|
| `FAULT_SPIKE` | "outlier spike" |
| `FAULT_STUCK` | "stuck reading (not updating)" |
| `FAULT_OFFLINE` | "sensor offline (no data)" |
| `DEGRADED` | "reading drifted from model estimate" |

When available, the event also shows the **measured vs. model-expected
value** (from `TwinNodeState.fault_sensor_value` /
`.fault_model_value`, `03_digital_twin.md` §2.3).

**Dead-banded thresholds**: threshold-crossing events (unmet demand, H₂
SOC, price spikes) use a dead-band between their raise and clear
thresholds — e.g. H₂ SOC uses a 5-point dead-band — so a value oscillating
right at a boundary produces **one** event per real episode, not one per
step-crossing. Each of these conditions also emits an explicit
"cleared/recovered" info event (e.g. `soc_clear`: *"H₂ SOC recovered
(NN%)"*), mirroring the grid-outage event's existing raise/clear pair.

The screen shows a Critical / Warning / Info summary plus the full log
(`GET /api/events`).

---

## 10. Diagnostics

A per-component **health / anomaly / sensor-quality heatmap table**, backed
by each `TwinNodeState`'s `health_score`, `anomaly_score`, and
`sensor_quality` (`03_digital_twin.md` §2.3) across every site — the
operator-facing view of the same fusion output that drives the Events &
Alarms log.

---

## 11. Configuration

A scenario YAML editor — this is the mechanism for defining new nodes,
editing existing node/component characteristics, and tuning fault
frequencies, entirely from the UI, with no restart needed.

### REST endpoints (`dashboard/network_app.py`)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/scenarios` | Lists `config/*.yaml`, flags which one is active |
| `GET /api/scenarios/{name}` | Returns the raw YAML text |
| `POST /api/scenarios/{name}` (body `{"yaml": "..."}`) | Validates by round-tripping the text through the real `Scenario.from_yaml(...).topology()` parser (via a temp file) **before** writing to disk; returns **HTTP 400** with the parser's error message if invalid |
| `POST /api/scenarios/{name}/activate` | Marks a scenario as the active config path for the worker, the training worker, and the comparison worker |

Path traversal is blocked: only bare `*.yaml`/`*.yml` filenames directly
inside `config/` are accepted (`_scenario_file()` resolves and checks the
parent directory).

### Activation semantics

Activating a scenario **takes effect on the next** `/sim/reset`, training
run (`/api/train/start`), or comparison run (`/api/live_compare/start`) —
**never** on a run already in progress. This is intentional: it prevents a
mid-simulation topology swap, which would otherwise silently corrupt a
running episode's state.

### UI

A scenario dropdown, a full-height YAML textarea, and **Reload** /
**Validate & Save** / **Activate** buttons, with inline guidance that:

- A **new node** is added by duplicating an existing `- id: ...` site block
  under `network.sites` and giving it a new, unique `id`.
- **Per-sensor fault frequency** is tuned via each site's/component's
  `fault_probability` field (`04_virtual_sensors.md` §6).

See `08_configuration_reference.md` for the full YAML schema and real
examples pulled from `config/italy_network_large.yaml`, and
`09_usage_guide.md` for a worked end-to-end walkthrough of editing and
activating a configuration.

---

## 12. Other endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /sim/start` / `POST /sim/stop` / `POST /sim/reset` | Simulation clock control |
| `POST /control` | Switch active controller (`none`/`classical`/`rl`) |
| `POST /sim/speed` | Change playback speed factor |
| `GET /status` | Current run state (running/paused, step count, controller) |
| `WS /ws` | Live `NetworkState` stream to all connected browsers |
| `WS /ws/compare` | Live lockstep comparison stream (§7) |

---

## 13. Branding

The dashboard uses a near-black background (`#0a0d13`), a teal/turquoise
brand accent (`#54e9c9`), and a pale lime-green accent (`#d4fdaa`) reserved
specifically for AI/RL-related UI — the Training screen, AI-controller
badges, and the learning-curve chart — echoing the HyTwin logo's "AI FOR
SMART H2" tagline, which is rendered in that same lime color. The sidebar
logo is `dashboard/static/logo.png`, a cropped copy of the repository-root
`HyTwin_Logo.png`.
