# HyTwin 2.0 — Technical Documentation

Welcome to the technical documentation for **HyTwin 2.0**, an AI-controlled
digital twin of a multi-node hydrogen (H₂) energy network. The flagship
scenario is a 7-node Italian network — physics-based component models,
virtual sensors with realistic measurement artifacts, an explicit multi-site
dispatch layer, reinforcement-learning-based network control, and a
real-time "Network Control Room" web dashboard.

---

## Table of contents

| Document | Description |
|----------|-------------|
| [01_architecture.md](01_architecture.md) | System architecture — layer stack, package layout, data flow, core infrastructure |
| [02_physical_models.md](02_physical_models.md) | Physical models — equations, rationale and parameters for every component, including the inter-site link models |
| [03_digital_twin.md](03_digital_twin.md) | Digital twin architecture — TwinNode, GridTwin, model-sensor fusion, scale-free anomaly scoring |
| [04_virtual_sensors.md](04_virtual_sensors.md) | Virtual sensor layer — noise/drift/delay/fault pipeline, quality scoring, fault-to-alarm mapping |
| [05_network_layer.md](05_network_layer.md) | Network layer — multi-site topology, explicit dispatch, NetworkState, traditional and RL network control |
| [06_reinforcement_learning.md](06_reinforcement_learning.md) | Reinforcement learning — single-site and network Gymnasium environments, reward design, training |
| [07_dashboard.md](07_dashboard.md) | Dashboard — the 9-screen Network Control Room, REST/WebSocket API, live training and comparison |
| [08_configuration_reference.md](08_configuration_reference.md) | Configuration reference — full YAML schema for single-site and multi-site scenarios, real examples |
| [09_usage_guide.md](09_usage_guide.md) | **How to use HyTwin** — end-to-end CLI and dashboard workflows for simulation, training, comparison, and result interpretation |

If you only read one document beyond this index, read
**`09_usage_guide.md`**.

---

## Quick start

```bash
# Network Control Room dashboard (recommended)
python -m dashboard.network_app --speed-factor 30
# then open http://localhost:8060

# Compare traditional vs. AI control from the CLI
python -c "
from hytwin.simulation.scenario import Scenario
from hytwin.network.compare import compare_controllers
topo = Scenario.from_yaml('config/italy_network_large.yaml').topology()
out = compare_controllers(topo, steps=1008, seed=42,
                           strategies={'none': 'none', 'classical': 'classical'})
for name, kpis in out.items():
    print(name, kpis)
"

# Test suite
pytest tests/ -v
```

See `09_usage_guide.md` for the full walkthrough, including training an AI
agent and running a live Scenario Comparison from the dashboard.

### Validation

| Check | Result |
|-------|--------|
| Unit tests (`pytest tests/`) | **168 / 168 PASS** |

Test coverage spans the digital twin, link models, component models,
network controller, network dashboard, network RL, network topology,
network twin, RL environment, sensors, and weather/cost realism.

### Technology stack

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| NumPy | ≥1.24 |
| SciPy | ≥1.10 |
| Gymnasium | ≥0.29 |
| Stable-Baselines3 | ≥2.1 |
| PyYAML | ≥6.0 |
| Pandas | ≥2.0 |
| Matplotlib | ≥3.7 |
| FastAPI + WebSocket | dashboard runtime |
