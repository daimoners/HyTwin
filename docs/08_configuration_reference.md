# Configuration Reference — HyTwin 2.0

Full YAML schema reference for both single-site and multi-site (`network:`)
scenarios, with real examples drawn from the files under `config/`. Six
scenario files ship with the repository:

| File | Sites | Purpose |
|------|-------|---------|
| `italy_network_large.yaml` | 7 (Trapani, Cagliari, Taranto, Napoli, Roma, Bologna, Milano) | **Flagship scenario** — the default network, producers in the sunny/windy south & islands, consumers in the industrial north |
| `italy_network_pilot.yaml` | 3 | Smaller network, used in the RL unit tests |
| `default_grid.yaml` | 1 (legacy) | Compact single-site pilot |
| `advanced_grid.yaml` | 1 (legacy) | Single-site + grid connection + energy cost |
| `advanced_stress.yaml` | 1 (legacy) | Single-site stress-test configuration |
| `pilot_scenario.yaml` | 1 (legacy) | Single-site scenario used in early tests |

All scenario files can be edited and activated from the dashboard's
**Configuration** screen (`07_dashboard.md` §11) as well as by editing the
file directly and pointing `--config` / `Scenario.from_yaml(...)` at it.

---

## 1. Top-level fields

```yaml
name: "italy_h2_network_large"
start_time: "2024-06-15T00:00:00"   # ISO datetime, simulation start
dt_seconds: 600                      # step duration [s] — 600 = 10 minutes
episode_steps: 144                   # steps per episode — 144 = 24h at dt=600s

grid: { ... }        # single-site schema (mutually exclusive with `network:`)
# — or —
network: { sites: [...], links: [...] }   # multi-site schema
```

`Scenario.from_yaml` is backward-compatible: if the file has a `network:`
block, the multi-site layer is built; otherwise the top-level `grid:` block
is wrapped as a single-site, zero-link `NetworkTopology` automatically — a
legacy single-site file needs no changes to keep working.

---

## 2. Single-site `grid:` schema

```yaml
grid:
  wind_turbines:   [ { id: wt1, params: { ... } }, ... ]
  pv_arrays:       [ { id: pv1, params: { ... } }, ... ]
  electrolyzers:   [ { id: el1, params: { ... } }, ... ]
  fuel_cells:      [ { id: fc1, params: { ... } }, ... ]
  hydrogen_tanks:  [ { id: tk1, params: { ... } }, ... ]
  loads:           [ { id: load1, params: { ... } }, ... ]
  grid_connections:[ { id: grid1, params: { ... } }, ... ]

weather:      { ... }     # WeatherModel parameters — see §5
energy_cost:  { ... }     # EnergyCostModel parameters — see §6
sensors:      [ ... ]     # virtual sensor list — see §7
```

Every component list uses the same `{ id: <str>, params: {...} }` shape —
`id` is the component's key in the `StateManager`/dashboard, `params` holds
the physical-model constructor arguments documented in
`02_physical_models.md`.

### 2.1 Real example — wind turbine (from `config/default_grid.yaml`)

```yaml
wind_turbines:
  - id: wt1
    params:
      rotor_diameter_m: 77.0
      hub_height_m: 80.0
      rated_power_kw: 500.0
      v_cut_in: 3.0
      v_rated: 12.0
      v_cut_out: 25.0
      efficiency_gen: 0.94
      altitude_m: 50.0
      turbulence_intensity: 0.06
```

### 2.2 Real example — PV array

```yaml
pv_arrays:
  - id: pv1
    params:
      n_panels: 600
      panel_area_m2: 1.96
      eta_stc: 0.205
      temp_coeff_pmax: -0.0038
      noct_c: 45.0
      rated_power_kw: 300.0
      soiling_loss: 0.03
      degradation_per_year: 0.005
      tilt_deg: 30.0
      azimuth_deg: 180.0
```

### 2.3 Real example — grid connection (from `config/advanced_grid.yaml`)

```yaml
grid_connections:
  - id: grid1
    params:
      max_import_kw: 800.0
      max_export_kw: 200.0
      base_availability: 0.97
      outage_mean_h: 2.5
      outage_rate_per_day: 0.08
      ramp_rate_kw_s: 80.0
      grid_carbon_factor: 0.38
```

See `02_physical_models.md` for the full parameter tables of every
component type (electrolyzer, fuel cell, hydrogen tank, load).

---

## 3. Multi-site `network:` schema

```yaml
network:
  sites:
    - id: <unique site id>
      location: { name: <str>, lat: <float>, lon: <float>, alt_m: <float> }
      weather: { ... }        # same fields as single-site `weather:` (§5)
      energy_cost: { ... }     # same fields as single-site `energy_cost:` (§6)
      grid: { ... }            # identical schema to the single-site `grid:` block (§2)
      sensors: [ ... ]         # optional, same schema as §7

  links:
    - id: <unique link id>
      type: electric_line | h2_pipeline
      from: <site id>
      to: <site id>
      params: { ... }          # link-type-specific, see §4
```

`location.lat/lon/alt_m` are merged automatically into `weather` (so
`WeatherModel` always has the right geodata, even if `weather:` omits it).
`links[].params.length_km` is optional — if omitted, `NetworkTopology`
computes it from the great-circle distance between the two sites'
`location` coordinates.

### 3.1 Real example — one site (from `config/italy_network_large.yaml`)

```yaml
- id: trapani
  location: { name: "Trapani", lat: 37.98, lon: 12.51, alt_m: 20 }
  weather: { weibull_k: 2.2, weibull_c: 7.2, autocorr_wind: 0.91,
             cloud_cover_mean: 0.20, temp_mean_c: 19.5, temp_amplitude_c: 11 }
  energy_cost: { f1_price: 0.23, f2_price: 0.15, f3_price: 0.08,
                 price_volatility: 0.10, seasonal_amplitude: 0.20,
                 spike_prob_per_day: 0.05, spike_multiplier: 3.0,
                 spike_duration_steps: 4, sell_ratio: 0.30 }
  grid:
    wind_turbines:
      - { id: trapani_wt1, params: { rotor_diameter_m: 82, hub_height_m: 85,
          rated_power_kw: 500, v_cut_in: 3, v_rated: 12, v_cut_out: 25,
          efficiency_gen: 0.94, altitude_m: 20, turbulence_intensity: 0.07 } }
    pv_arrays:
      - { id: trapani_pv1, params: { n_panels: 600, panel_area_m2: 1.96,
          eta_stc: 0.205, temp_coeff_pmax: -0.0038, noct_c: 45,
          rated_power_kw: 300, soiling_loss: 0.03, degradation_per_year: 0.005,
          tilt_deg: 28, azimuth_deg: 180 } }
    electrolyzers:
      - { id: trapani_el1, params: { rated_power_kw: 300, n_cells: 100,
          cell_area_cm2: 400, membrane_resistance_ohm_cm2: 0.16,
          temperature_c: 65, min_load_fraction: 0.05, ramp_rate_kw_s: 5 } }
    hydrogen_tanks:
      - { id: trapani_tk1, params: { volume_m3: 14, max_pressure_bar: 700,
          min_pressure_bar: 10, initial_soc: 0.50, temperature_c: 20,
          max_charge_rate_kg_s: 0.08, max_discharge_rate_kg_s: 0.05,
          boiloff_rate_per_day: 0 } }
    loads:
      - { id: trapani_load, params: { base_load_kw: 150, profile_type: "commercial",
          noise_std_fraction: 0.06, seasonal_amplitude: 0.12,
          demand_response_factor: 0.15 } }
    grid_connections:
      - { id: trapani_grid, params: { max_import_kw: 400, max_export_kw: 300,
          base_availability: 0.98, outage_mean_h: 2, outage_rate_per_day: 0.05,
          ramp_rate_kw_s: 80, grid_carbon_factor: 0.33 } }
```

### 3.2 Real example — links (from `config/italy_network_large.yaml`)

```yaml
links:
  - { id: trapani_napoli_h2,   type: h2_pipeline,   from: trapani, to: napoli,
      params: { diameter_m: 0.4, max_flow_kg_h: 300,
                compressor_spec_kwh_per_kg: 2.2, line_pack_capacity_kg: 1000 } }
  - { id: trapani_napoli_elec, type: electric_line, from: trapani, to: napoli,
      params: { max_power_mw: 40, loss_per_1000km: 0.06, bidirectional: true } }
```

The flagship scenario connects its 7 sites with 12 links (an H₂ pipeline and
an electric line in parallel along each of the 6 backbone corridors:
Trapani↔Napoli, Cagliari↔Roma, Taranto↔Napoli, Taranto↔Bologna,
Napoli↔Roma, Roma↔Bologna, Bologna↔Milano).

### 3.3 How to add a new node

1. Copy an existing `- id: ...` block under `network.sites`.
2. Give it a new, unique `id` and set its `location` coordinates.
3. Adjust its `grid:` equipment lists (wind/PV/electrolyzer/fuel-cell/tank/
   load/grid_connection) to the mix you want at that site — a site can carry
   any subset of component types (e.g. a pure-consumer site has no wind
   turbines, only PV/fuel-cells/loads, as `roma`/`bologna`/`milano` do in
   the flagship scenario).
4. Optionally add `links` entries connecting the new site to its neighbours
   (`length_km` can be omitted — it is computed automatically).

This can be done either by editing the YAML file directly, or through the
dashboard's Configuration screen (`07_dashboard.md` §11), which validates
the edit by round-tripping it through the real parser before saving.

---

## 4. Link parameters

### `electric_line`

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `max_power_mw` | MW | 50 | Thermal transfer capacity (`max_power_kw` also accepted) |
| `length_km` | km | auto | Line length — auto-computed from site coordinates if omitted |
| `loss_per_1000km` | — | 0.06 | Fractional resistive loss per 1000 km |
| `bidirectional` | bool | true | Allow reverse flow |
| `ramp_rate_kw_s` | kW/s | ∞ | Max change in transferred power |
| `outage_rate_per_day`, `outage_mean_h` | — | 0 | Optional stochastic outage model |

### `h2_pipeline`

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `max_flow_kg_h` | kg/h | 300 | Max injection flow |
| `length_km` | km | auto | Auto-computed from site coordinates if omitted |
| `diameter_m` | m | 0.4 | Informative only |
| `compressor_spec_kwh_per_kg` | kWh/kg | 2.0 | Compression energy per kg injected (extra electric load at the source site) |
| `line_pack_capacity_kg` | kg | 800 | Max in-pipe buffered mass |
| `transport_velocity_ms` | m/s | 15 | Effective transport speed (sets delay) |

---

## 5. `weather:` section

```yaml
weather:
  latitude_deg: 40.5
  longitude_deg: 14.8
  altitude_m: 50.0
  weibull_k: 2.0
  weibull_c: 6.5
  autocorr_wind: 0.92
  cloud_cover_mean: 0.35
  temp_mean_c: 15.5
  temp_amplitude_c: 10.0
```

| Parameter | Description |
|-----------|-------------|
| `latitude_deg`, `longitude_deg` | Geographic coordinates (auto-filled from `location` in a network site) |
| `altitude_m` | Altitude — air-density and irradiance correction |
| `weibull_k`, `weibull_c` | Weibull wind-speed shape and scale |
| `autocorr_wind` | Lag-1 wind autocorrelation |
| `cloud_cover_mean` | Mean cloud cover (0–1) |
| `temp_mean_c`, `temp_amplitude_c` | Mean and seasonal amplitude of air temperature |

In a multi-site network, per-site `WeatherModel` instances are coupled by
`WeatherField` — see `02_physical_models.md` §10 for the correlation model
(`correlation_length_km`, `synoptic_tau_hours`).

---

## 6. `energy_cost:` section

```yaml
energy_cost:
  f1_price: 0.28          # peak (Mon-Fri 08:00-19:00)  [€/kWh]
  f2_price: 0.18          # shoulder                     [€/kWh]
  f3_price: 0.09          # off-peak (night / Sunday)    [€/kWh]
  price_volatility: 0.10  # day-ahead market noise
  seasonal_amplitude: 0.22
  spike_prob_per_day: 0.06
  spike_multiplier: 3.5
  spike_duration_steps: 4
  sell_ratio: 0.28         # feed-in tariff = buy price * sell_ratio
```

Models the Italian day-ahead market (PUN) F1/F2/F3 tariff structure,
including the Italian national holiday calendar (movable Easter/Pasquetta
included) and a merit-order renewable-abundance discount — see
`02_physical_models.md` §8. Omitting this section falls back to model
defaults.

---

## 7. `sensors:` section

```yaml
sensors:
  - id: wt1.power
    type: power
    model_key: "wt1.power_kw"
    noise_std_kw: 5.0
    drift_rate_kw: 0.1

  - id: tk1.pressure
    type: pressure
    model_key: "tk1.pressure_bar"
    noise_std: 1.0
    fault_probability: 0.002
```

| Field | Description |
|-------|-------------|
| `id` | Sensor id — must be prefixed with the owning node's id (`04_virtual_sensors.md` §10) |
| `type` | Sensor type (`power`, `voltage`, `current`, `pressure`, `temperature`, `flow`, `h2_level`, `irradiance`, `wind_speed`) |
| `model_key` | `StateManager` key the sensor reads the true value from — must match exactly what `GridTwin`/`NetworkTwin` writes |
| `noise_std` (or `noise_std_<unit>`) | Gaussian noise σ |
| `drift_rate` (or `drift_rate_<unit>`) | Random-walk drift rate per step |
| `delay_steps` | Measurement delay in steps |
| `quantisation_step` | ADC resolution (0 = no quantisation) |
| `fault_probability` | Per-step probability of a random `FAULT_SPIKE` injection — **this is the field to tune per-sensor fault frequency** (`04_virtual_sensors.md` §6) |

The `sensors:` section may be omitted entirely — in that case the
simulation runs without virtual sensors, and the digital twin uses model
values directly (quality = 1.0, no noise).

---

## 8. Validation

The dashboard's `POST /api/scenarios/{name}` endpoint validates a scenario
edit by round-tripping the submitted YAML text through the real
`Scenario.from_yaml(...).topology()` parser via a temporary file, and
returns **HTTP 400** with the parser's own error message if the config is
invalid — nothing is ever written to disk without first passing this check.
The same applies programmatically:

```python
from hytwin.simulation.scenario import Scenario

topo = Scenario.from_yaml("config/my_network.yaml").topology()
```

raises a descriptive exception immediately if the YAML is malformed or a
required field is missing.

---

## 9. Choosing a scenario at runtime

```bash
# Dashboard
python -m dashboard.network_app --config config/italy_network_pilot.yaml --speed-factor 30

# CLI / Python
from hytwin.simulation.scenario import Scenario
topo = Scenario.from_yaml("config/italy_network_pilot.yaml").topology()
```

See `09_usage_guide.md` for the full workflow of editing, validating, and
activating a scenario, including doing so live from the dashboard's
Configuration screen without a restart.
