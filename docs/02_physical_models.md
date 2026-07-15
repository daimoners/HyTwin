# Physical Models — HyTwin 2.0

This document describes the physical models that make up HyTwin's simulation
layer: the six per-site component models, the two inter-site link models, and
the weather model that drives them all, including their physical rationale,
implemented equations, and configuration parameters.

---

## Common interface: `BaseModel`

All models inherit from `BaseModel` and share the same interface:

```python
class BaseModel:
    def step(self, dt: float, context: dict) -> ModelState: ...
    def reset(self) -> None: ...
    @property
    def component_id(self) -> str: ...
    @property
    def params(self) -> dict: ...
```

`ModelState` is a dataclass containing:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `datetime` | Computation instant |
| `component_id` | `str` | Node identifier |
| `values` | `dict[str, float]` | Model output (key → value) |

---

## 1. Wind Turbine — `WindTurbineModel`

### Rationale

A variable-speed turbine model with:
- Analytical power coefficient curve $C_p(\lambda, \beta)$
- Air-density correction for altitude and temperature (ISA)
- Vertical wind-shear profile (power law)
- Additive stochastic turbulence on hub-height wind speed

### Power coefficient model

Analytical approximation from **Heier (2014)**:

$$C_p(\lambda, \beta) = 0.5176 \left(\frac{116}{\lambda_i} - 0.4\beta - 5\right)
e^{-21/\lambda_i} + 0.0068\lambda$$

where:

$$\frac{1}{\lambda_i} = \frac{1}{\lambda + 0.08\beta} - \frac{0.035}{\beta^3 + 1}$$

with the Betz limit constraint: $C_p \leq 0.593$.

### Wind shear

Hub-height wind speed is derived from the reference-height (anemometer)
measurement via the **power law** (shear exponent $\alpha = 0.14$ for an
onshore site):

$$v_{\text{hub}} = v_{\text{ref}} \left(\frac{z_h}{z_r}\right)^\alpha$$

### Electrical power

$$P_{\text{el}} =
\begin{cases}
0 & v < v_{\text{ci}} \text{ or } v > v_{\text{co}} \\
P_{\text{rated}} & v \geq v_{\text{rated}} \\
\frac{1}{2}\rho A v_{\text{hub}}^3 C_p \;\eta_{\text{gen}} \cdot \frac{\rho}{\rho_0}
& v_{\text{ci}} \leq v < v_{\text{rated}}
\end{cases}$$

where the air-density correction uses the **ISA hypsometric formula**:

$$\rho = \frac{P_0}{R_{\text{air}} T_K} \left(1 - \frac{0.0065 \cdot z}{T_0}\right)^{5.2561}$$

### Configuration parameters

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `rotor_diameter_m` | m | — | Rotor diameter |
| `hub_height_m` | m | 80 | Hub height |
| `rated_power_kw` | kW | — | Rated power |
| `v_cut_in` | m/s | 3.0 | Cut-in speed |
| `v_rated` | m/s | 12.0 | Rated speed |
| `v_cut_out` | m/s | 25.0 | Cut-out speed |
| `efficiency_gen` | — | 0.94 | Generator + gearbox efficiency |
| `altitude_m` | m | 0 | Site altitude |
| `wake_loss_factor` | — | 0.0 | Wake loss fraction |
| `turbulence_intensity` | — | 0.05 | σ/μ of turbulence |

### Model output

| Key | Unit | Description |
|-----|------|-------------|
| `power_kw` | kW | Net electrical power |
| `energy_kwh_step` | kWh | Energy produced this step |
| `energy_kwh_total` | kWh | Cumulative energy |
| `wind_speed_hub_ms` | m/s | Computed hub-height speed |
| `wind_speed_ref_ms` | m/s | Input reference speed |
| `available` | 0/1 | Whether the turbine is operating in range |

**References**: Burton et al. (2011) *Wind Energy Handbook*, Wiley; IEC 61400-12-1:2017.

---

## 2. Photovoltaic Array — `PhotovoltaicModel`

### Rationale

The full chain from solar irradiance to AC power delivered to the grid:
1. Plane-of-array (POA) irradiance with beam/diffuse/reflected decomposition
2. Cell temperature via the NOCT model
3. DC power with the $P_{\max}$ thermal coefficient
4. Inverter efficiency via a European efficiency curve

### Plane-of-array (POA) irradiance

$$G_{\text{POA}} = G_{\text{beam}} \cdot \text{IAM}(\theta) + G_{\text{diff}} + G_{\text{refl}}$$

- **Beam** (direct component on the tilted plane): $G_{\text{beam}} = \text{DNI} \cdot \cos\theta_i$
- **Angle of incidence** $\theta_i$: $\cos\theta_i = \cos z \cos\beta + \sin z \sin\beta \cos(\psi_s - \psi_p)$,
  with $z$ = zenith, $\beta$ = panel tilt, $\psi$ = azimuth
- **IAM (ASHRAE)**, correcting for glass-surface reflection:
  $\text{IAM} = 1 - b_0 \left(\frac{1}{\cos\theta_i} - 1\right)$
- **Isotropic diffuse**: $G_{\text{diff}} = G_{\text{DHI}} \frac{1+\cos\beta}{2}$
- **Ground-reflected** (albedo 0.2): $G_{\text{refl}} = G_{\text{GHI}} \cdot 0.2 \frac{1-\cos\beta}{2}$

### Cell temperature (NOCT)

$$T_{\text{cell}} = T_{\text{amb}} + \frac{\text{NOCT} - 20}{800} \cdot G_{\text{POA}} \cdot \frac{9.5}{5.7 + 3.8 v_w}$$

### DC power

$$P_{\text{DC}} = N \cdot A \cdot G_{\text{POA}} \cdot \eta_{\text{STC}} \cdot [1 + \alpha_T(T_{\text{cell}} - 25)] \cdot (1 - \delta_{\text{soil}}) \cdot (1 - \delta_{\text{deg}} \cdot t)$$

where $\alpha_T$ is the $P_{\max}$ thermal coefficient (typically $-0.004$/°C).

### Inverter efficiency

A European efficiency curve approximated by a 4th-order polynomial in the
load ratio $\ell = P_{\text{DC}} / P_{\text{rated}}$:

$$\eta_{\text{inv}}(\ell) = -0.0162\ell^4 + 0.0499\ell^3 - 0.0518\ell^2 + 0.0237\ell + 0.9713$$

Peak ~97% at rated load; falls off quickly at low loads.

### Configuration parameters

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `n_panels` | — | 1 | Number of panels |
| `panel_area_m2` | m² | 1.7 | Area per panel |
| `eta_stc` | — | 0.20 | STC efficiency |
| `temp_coeff_pmax` | 1/°C | −0.004 | $P_{\max}$ thermal coefficient |
| `noct_c` | °C | 47 | NOCT |
| `rated_power_kw` | kW | calc. | Rated DC power |
| `soiling_loss` | — | 0.02 | Soiling loss (2%) |
| `degradation_per_year` | 1/yr | 0.005 | Annual degradation (0.5%) |
| `tilt_deg` | ° | 30 | Tilt |
| `azimuth_deg` | ° | 180 | Azimuth (180 = South) |
| `iam_b0` | — | 0.05 | ASHRAE IAM coefficient |

**References**: De Soto et al. (2006) *Solar Energy*; Duffie & Beckman (2013).

---

## 3. PEM Electrolyzer — `ElectrolyzerModel`

### Rationale

The proton-exchange-membrane (PEM) electrolyzer is the key component
converting surplus renewable energy into green hydrogen. The model
implements the full polarisation curve with three distinct overpotentials.

### Cell voltage

$$V_{\text{cell}} = V_{\text{rev}} + \eta_{\text{act}} + \eta_{\text{ohm}} + \eta_{\text{conc}}$$

**Reversible voltage** (temperature-dependent, linear approximation of the
Nernst potential): $V_{\text{rev}} = 1.229 - 9.0 \times 10^{-4}(T - 298.15)$ [V]

**Activation overpotential** (simplified Butler-Volmer):
$\eta_{\text{act}} = \frac{RT}{\alpha_a F} \ln\left(\frac{i}{i_0}\right)$,
with $\alpha_a = 0.5$, $i_0 = 10^{-3}$ A/cm² (anodic exchange current density).

**Ohmic overpotential** (membrane resistance, Ohm's law):
$\eta_{\text{ohm}} = i \cdot R_{\text{mem}}$, $R_{\text{mem}}$ in Ω·cm² (ASR).

**Concentration overpotential** (mass transport):
$\eta_{\text{conc}} = -\frac{RT}{2F} \ln\left(1 - \frac{i}{i_{\text{lim}}}\right)$,
with $i_{\text{lim}} = 2.0$ A/cm².

### Newton solver

Given the power setpoint $P$ [W], the model iteratively finds the current
density $i$ satisfying $P = N_{\text{cells}} \cdot V_{\text{cell}}(i) \cdot i \cdot A_{\text{cell}}$
via Newton's method with an analytic derivative approximation. Typical
convergence in 5–10 iterations, relative error below $10^{-6}$.

### Hydrogen production (Faraday's law)

$$\dot{m}_{\text{H}_2} = \frac{I \cdot N_{\text{cells}}}{N_e \cdot F} \cdot M_{\text{H}_2} \cdot \eta_F$$

$N_e = 2$ electrons per H₂ molecule, $F = 96485$ C/mol.

**Faradaic efficiency** (parasitic losses at low current):
$\eta_F = \min\!\left(1, \max\!\left(0,\; 1 - e^{-4i + 0.5}\right)\right)$

### Energy efficiency (HHV)

$$\eta_{\text{HHV}} = \frac{\dot{m}_{\text{H}_2} \cdot \text{HHV}_{\text{H}_2}}{P_{\text{input}}}$$

with $\text{HHV}_{\text{H}_2} = 141.86$ MJ/kg. Typical values: 55–70% HHV
(48–55% LHV) at partial load.

### Configuration parameters

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `rated_power_kw` | kW | — | Rated DC power |
| `n_cells` | — | 100 | Cells in series |
| `cell_area_cm2` | cm² | 300 | Active cell area |
| `membrane_resistance_ohm_cm2` | Ω·cm² | 0.17 | Membrane ASR |
| `i_exchange_A_cm2` | A/cm² | 0.001 | Exchange current density |
| `alpha_a` | — | 0.5 | Anodic transfer coefficient |
| `temperature_c` | °C | 60 | Operating temperature |
| `min_load_fraction` | — | 0.05 | Minimum stable load (5%) |
| `ramp_rate_kw_s` | kW/s | ∞ | Ramp limit |
| `degradation_rate` | 1/kh | 1×10⁻⁵ | Degradation per 1000 operating hours |
| `water_flow_l_h_kw` | L/(h·kW) | 0.9 | Water consumption |

**References**: Olivier et al. (2017) *Int. J. Hydrogen Energy*; IRENA (2020).

---

## 4. PEM Fuel Cell — `FuelCellModel`

### Rationale

The PEM fuel cell converts hydrogen into electricity, the reverse reaction
of electrolysis. It reuses the same polarisation-curve structure as the
electrolyzer with losses of different magnitude (the cell operates in the
opposite direction).

### Polarisation curve

$$V_{\text{cell}} = V_{\text{OCV}} - \eta_{\text{act}} - \eta_{\text{ohm}} - \eta_{\text{conc}}$$

**OCV (Open Circuit Voltage)** — Nernst equation at atmospheric pressure:
$V_{\text{OCV}} = 1.229 - 9.0 \times 10^{-4}(T - 298.15)$ [V]

**Activation loss** (Tafel equation, cathode-dominant):
$\eta_{\text{act}} = \frac{RT}{\alpha_c F} \ln\left(\frac{i}{i_0}\right)$, $\alpha_c = 0.4$.

**Ohmic loss**: $\eta_{\text{ohm}} = i \cdot R_{\text{mem}}$ (typically $R = 0.12$ Ω·cm²).

**Concentration loss** (cathode mass-transport limitation):
$\eta_{\text{conc}} = -\frac{RT}{4F} \ln\left(1 - \frac{i}{i_{\text{lim}}}\right)$,
$i_{\text{lim}} = 1.8$ A/cm² (lower than in the electrolyzer).

### Stack sizing

For a rated power $P$ kW at typical cell voltage $V_c \approx 0.6$ V and
current density $i \approx 1$ A/cm²:

$$N_{\text{cells}} \geq \frac{P \times 10^3}{V_c \cdot i \cdot A_{\text{cell}}} = \frac{P \times 10^3}{0.6 \times 1 \times A_{\text{cell}}}$$

**Example**: for $P = 50$ kW with $A = 200$ cm²: $N_{\min} \approx 417$ cells.

### H₂ consumption (Faraday's law)

$$\dot{m}_{\text{H}_2,\text{cons}} = \frac{I \cdot N_{\text{cells}}}{N_e \cdot F} \cdot \frac{M_{\text{H}_2}}{\eta_u}$$

where $\eta_u$ is the H₂ utilisation factor (typical 0.80 — 20% unreacted,
recirculated or vented).

### Efficiency (LHV)

$$\eta_{\text{LHV}} = \frac{P_{\text{el}}}{\dot{m}_{\text{H}_2} \cdot \text{LHV}_{\text{H}_2} / \Delta t}$$

with $\text{LHV}_{\text{H}_2} = 119.96$ MJ/kg. Typical values: 40–60% LHV.

### Configuration parameters

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `rated_power_kw` | kW | — | Rated AC power |
| `n_cells` | — | 80 | Cells in series |
| `cell_area_cm2` | cm² | 300 | Active cell area |
| `membrane_resistance_ohm_cm2` | Ω·cm² | 0.12 | Membrane ASR |
| `i_exchange_A_cm2` | A/cm² | 0.001 | Exchange current density |
| `alpha_c` | — | 0.4 | Cathodic transfer coefficient |
| `temperature_c` | °C | 65 | Operating temperature |
| `h2_utilisation` | — | 0.80 | H₂ utilisation factor |
| `min_load_fraction` | — | 0.10 | Minimum stable load (10%) |
| `ramp_rate_kw_s` | kW/s | ∞ | Ramp limit |
| `degradation_rate` | 1/kh | 2×10⁻⁵ | Degradation per 1000 h |

**References**: Larminie & Dicks (2003) *Fuel Cell Systems Explained*, Wiley; DOE Targets.

---

## 5. Hydrogen Tank — `HydrogenTankModel`

### Rationale

Type-IV storage (composite material, up to 700 bar) is modelled with the
**Van der Waals** equation of state instead of the ideal gas law, since real
deviations become significant above 100 bar (≈15–20% error vs. ideal gas at
700 bar, 20 °C).

### Van der Waals equation of state

$$\left(P + \frac{a}{v_m^2}\right)(v_m - b) = RT$$

where $v_m = V / n$ is the molar volume [m³/mol], and for hydrogen:
$a = 0.2476 \times 10^{-3}$ Pa·m⁶/mol², $b = 26.61 \times 10^{-6}$ m³/mol.

`_vdw_n_from_P(P, V, T)` solves for $n$ given $P$ via Newton's method
(typically < 50 iterations).

### State of charge (SOC)

$$\text{SOC} = \frac{n - n_{\min}}{n_{\max} - n_{\min}}, \qquad m_{\text{H}_2} = n \cdot M_{\text{H}_2}\ [\text{kg}]$$

$n_{\min}$ and $n_{\max}$ correspond to $P_{\min}$ and $P_{\max}$, computed
at construction time.

### Permeation losses (boil-off)

$$\dot{m}_{\text{boil}} = m_{\text{H}_2} \cdot \frac{\delta_{\text{day}}}{86400} \cdot dt$$

$\delta_{\text{day}}$ is the daily loss rate (typically 0 for compressed
ambient-temperature storage; relevant for cryogenic storage).

### Operating limits

Charge/discharge are constrained by: max flow (`max_charge_rate_kg_s`,
`max_discharge_rate_kg_s`), pressure bounds ($P_{\min} \leq P \leq P_{\max}$),
and saturation (cannot charge beyond capacity nor discharge below empty).

### Configuration parameters

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `volume_m3` | m³ | — | Internal geometric volume |
| `max_pressure_bar` | bar | 700 | Maximum design pressure |
| `min_pressure_bar` | bar | 5 | Minimum operating pressure |
| `initial_soc` | — | 0.5 | Initial SOC |
| `temperature_c` | °C | 20 | Gas temperature |
| `max_charge_rate_kg_s` | kg/s | ∞ | Max charge flow |
| `max_discharge_rate_kg_s` | kg/s | ∞ | Max discharge flow |
| `boiloff_rate_per_day` | 1/d | 0.0 | Daily boil-off rate |

**References**: Colozza (2002) NASA/TM-2002-211867; ISO 15869:2009.

---

## 6. Energy Load — `EnergyLoadModel`

### Rationale

Realistic consumption profiles for multiple demand types, with multi-scale
temporal variability including the Italian weekday/weekend/holiday calendar.

### Profile structure

$$P_{\text{load}} = P_{\text{base}} \cdot f_{\text{diurnal}}(h) \cdot f_{\text{seasonal}}(d) \cdot f_{\text{weekday}} \cdot (1 + \xi) \cdot (1 - \text{DR})$$

**Diurnal profile** for `residential` (two Gaussian peaks):

$$f_{\text{diurnal}}(h) = 0.25 + 0.6\,e^{-\frac{(h-7)^2}{4.5}} + 1.0\,e^{-\frac{(h-20)^2}{8}}$$

**Seasonal factor**: a dual-hump curve reflecting both winter heating and
summer air-conditioning demand (not a single annual cosine).

**Weekday/weekend/holiday factor**: reduced load on weekends and on Italian
national holidays (including movable Easter/Pasquetta), plus a distinct
August industrial-shutdown factor for `industrial` profiles.

**Gaussian noise**: $\xi \sim \mathcal{N}(0, \sigma^2_{\text{noise}})$

**Demand response**: reduction of up to `demand_response_factor` (default
15%) of instantaneous load, driven by the control/RL layer.

### Diurnal profiles by type

| Type | Shape | Peaks |
|------|-------|-------|
| `residential` | 2 Gaussians + baseline | Morning ~07:00, evening ~20:00 |
| `industrial` | Flat profile | High 07:00–22:00, night −60% |
| `commercial` | Ramp + plateau | High 09:00–18:00, evening 50%, night 15% |

### Configuration parameters

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `base_load_kw` | kW | — | Average load at peak hour |
| `profile_type` | str | `residential` | Profile type |
| `noise_std_fraction` | — | 0.03 | Noise σ (3% of load) |
| `seasonal_amplitude` | — | 0.10 | Seasonal variation amplitude |
| `demand_response_factor` | — | 0.15 | Max DR reduction (15%) |

**References**: IEEE Std 1459-2010; ENTSO-E consumption profiles.

---

## 7. Grid Connection — `GridConnectionModel`

Each site has its own connection to the national grid, with independent
import/export capacity limits, stochastic outages (mean outage duration and
daily outage rate), ramp-rate limiting, and a carbon intensity factor used
for CO₂ KPIs.

### Configuration parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| `max_import_kw` / `max_export_kw` | kW | Capacity limits |
| `base_availability` | — | Baseline probability the connection is up |
| `outage_mean_h` | h | Mean outage duration |
| `outage_rate_per_day` | 1/d | Expected number of outages per day |
| `ramp_rate_kw_s` | kW/s | Max change in exchanged power |
| `grid_carbon_factor` | kgCO₂/kWh | Carbon intensity of imported energy |

---

## 8. Energy Cost — `EnergyCostModel`

Models the Italian day-ahead market (PUN) tariff structure per site,
including:

- **F1/F2/F3** peak/shoulder/off-peak tariff bands
- The **Italian national holiday calendar**, including the movable Easter
  Monday (Pasquetta), applied to the tariff-band schedule
- A **merit-order renewable-abundance discount**: local price softens when
  the site's own renewable output is high, approximating merit-order effects
- Daily price noise, a seasonal trend, and stochastic price spikes
  (`spike_prob_per_day`, `spike_multiplier`, `spike_duration_steps`)
- `sell_ratio`: fraction of the buy price paid for exported energy

### Configuration parameters

| Parameter | Description |
|-----------|-------------|
| `f1_price`, `f2_price`, `f3_price` | €/kWh tariff bands |
| `price_volatility` | Daily price noise amplitude |
| `seasonal_amplitude` | Seasonal trend amplitude |
| `spike_prob_per_day`, `spike_multiplier`, `spike_duration_steps` | Price-spike model |
| `sell_ratio` | Export price as a fraction of import price |

---

## 9. Inter-site link models

Both link models implement the standard `BaseModel` interface
(`step`/`reset`), so they can carry virtual sensors and participate in the
digital-twin fusion exactly like any other component — the transport
infrastructure is itself part of the digital twin.

### 9.1 `ElectricLineModel`

Models a high-voltage electric interconnection between two sites.

- **Input** (context): `power_request_kw` (signed — sign gives direction).
- **Physics**: resistive loss proportional to `loss_per_1000km · length_km`,
  saturation at `max_power_mw`, ramp-rate limiting, and an optional
  stochastic-outage model reusing the same logic as `GridConnectionModel`.
- **Output**: `power_in_kw`, `power_out_kw`, `loss_kw`, `utilization`, `available`.

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `max_power_mw` | MW | 50 | Thermal transfer capacity (`max_power_kw` also accepted) |
| `length_km` | km | 100 | Line length (auto-filled from site coordinates if omitted) |
| `loss_per_1000km` | — | 0.06 | Fractional resistive loss per 1000 km |
| `bidirectional` | bool | true | Allow reverse flow |
| `ramp_rate_kw_s` | kW/s | ∞ | Max change in transferred power |
| `outage_rate_per_day`, `outage_mean_h` | — | 0 | Optional stochastic line-fault model |

### 9.2 `H2PipelineModel`

Models a hydrogen pipeline between two sites, including compression energy
and transport delay.

- **Input** (context): `flow_request_kg_h` at the sending node.
- **Physics**: saturation at `max_flow_kg_h`, **compression energy**
  (`compressor_spec_kwh_per_kg` → an additional electric load at the source
  site), a FIFO transport delay derived from `length_km / transport_velocity_ms`,
  and a line-pack buffer (`line_pack_capacity_kg`) acting as a small
  in-transit reservoir.
- **Output**: `flow_kg_h`, `compressor_power_kw`, `linepack_kg`, `delivered_kg_h`.

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `max_flow_kg_h` | kg/h | 300 | Max injection flow |
| `length_km` | km | 100 | Pipeline length (auto-filled if omitted) |
| `diameter_m` | m | 0.4 | Informative only in the current flow model |
| `compressor_spec_kwh_per_kg` | kWh/kg | 2.0 | Compression energy per kg injected |
| `line_pack_capacity_kg` | kg | 800 | Max in-pipe buffered mass |
| `transport_velocity_ms` | m/s | 15 | Effective transport speed (sets delay) |
| `bidirectional` | bool | false | Reserved; current model treats the pipe as unidirectional |

---

## 10. Weather — `WeatherModel` and `WeatherField`

### `WeatherModel` (per site)

A standalone stochastic generator producing the environmental variables
consumed by every physical model at one site.

**Wind speed** (Weibull process with autocorrelation) — an AR(1) process in
Weibull-variable space:

$$z_t = \phi z_{t-1} + \sqrt{1 - \phi^2}\,\varepsilon_t, \quad \varepsilon_t \sim |\mathcal{N}(0,1)|, \qquad v_t = c \cdot z_t^{1/k}$$

with $\phi$ the lag-1 autocorrelation (typically 0.92), giving physically
plausible time series without abrupt jumps. Wind climatology has both
seasonal and diurnal modulation.

**Solar irradiance**: solar position (zenith, azimuth) computed exactly for
latitude/longitude using **Spencer (1971)** for declination and
**Kasten & Young** for air mass. Global Horizontal Irradiance:

$$G_{\text{GHI}} = G_{\text{cs}} \cdot (1 - c_f) + G_{\text{cs}} \cdot c_f \cdot r_c$$

where $G_{\text{cs}}$ is clear-sky GHI (simplified Ineichen model), $c_f$ is
cloud cover, and $r_c \approx 0.05$ is cloud transmittance.

**Cloud cover**: seasonally modulated mean with proper AR(1) mean-reversion
to its own configured seasonal mean (not a fixed drift to 0.5), so cloud
cover behaves like a real climatology rather than decaying to an arbitrary
constant.

**Synoptic coupling**: frontal/stormy conditions boost wind speed and cloud
cover together (a shared latent disturbance), instead of treating them as
independent processes.

**Air temperature**: AR(1) process with a sinusoidal seasonal trend:

$$T_t = T_{\text{mean}} + A \sin\!\left(\frac{2\pi (d-172)}{365}\right) + f_{\text{diurnal}}(h) + \rho_T (T_{t-1} - \mu) + \varepsilon_T$$

#### Configuration parameters

| Parameter | Description |
|-----------|-------------|
| `latitude_deg`, `longitude_deg` | Geographic coordinates |
| `altitude_m` | Altitude (air-density and irradiance correction) |
| `weibull_k`, `weibull_c` | Weibull shape and scale |
| `autocorr_wind` | Lag-1 wind autocorrelation |
| `cloud_cover_mean` | Mean cloud cover (0–1) |
| `temp_mean_c`, `temp_amplitude_c` | Mean and seasonal amplitude of T |

### `WeatherField` (multi-site)

Couples the `WeatherModel` instances of every site in a network so that
nearby sites see correlated weather while distant ones remain closer to
independent — matching how real synoptic weather systems move across Italy.

- One `WeatherModel` per site, parameterised from that site's local
  climatology (`location` + `weather` YAML block).
- **Spatial correlation**: a Cholesky-decomposed correlation matrix built
  from the haversine distance between every pair of sites
  (`correlation_length_km`, default 300 km) shapes how strongly sites'
  weather shocks move together.
- **Shared synoptic process**: a common AR(1) disturbance
  (`synoptic_tau_hours`, default 18 h) representing a passing weather front,
  applied with site-dependent weight from the correlation structure — this
  is what makes wind and cloud cover rise together at nearby sites during a
  storm, rather than purely independently-sampled series.
- Exposes `step(ts) -> Dict[site_id, weather_dict]`.
