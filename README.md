# Hydrogen Demand and Critical Materials for Green Steel Electrolysers

This repository contains the full model for my master's thesis:
**"Hydrogen Demand and Critical Material Requirements for Electrolysers in a Decarbonising Global Steel Sector"**

---

## What the model does

I model the transition of global steel production away from coal-based blast furnaces (BF-BOF) toward hydrogen-based direct reduced iron (H-DRI). The core question is: given a plausible range of steel demand trajectories and technology transition speeds, how much hydrogen will the steel sector need, how large does the electrolyser fleet need to be, and how much nickel, platinum, and iridium does that imply?

The approach is a material flow analysis (MFA) built on a Dynamic Stock Model (DSM). I project in-use steel stock forward using population and saturation dynamics, back-calculate annual steel production requirements, split production between BF-BOF, H-DRI, and EAF routes, then trace through to electrolyser capacity and critical material demand.

---

## Model logic — step by step

### Step 1 — Historical steel stock reconstruction
I load historical global steel stock data (Mueller et al. 2011, 1700–2008) and split it into four end-use sectors:
- Vehicles (mean lifetime 20 yr)
- Machinery (30 yr)
- Buildings & Construction (75 yr)
- Appliances (15 yr)

Sector shares are interpolated over time between 1900 values and measured 2008 values using a logistic bridge. Pre-1900 shares are held constant at 1900 values.

### Step 2 — Per-capita stock projection to 2100
I fit a logistic S-curve to historical per-capita steel stock (t/capita) up to 2025 (with a linear bridge 2008–2025 anchored toward a shared 11 t/cap midpoint). Beyond 2025, I switch to a Cubic Hermite Spline that smoothly approaches the scenario saturation level L by 2100 with zero slope at the endpoint. Three demand levels:
- Low: L = 6 t/capita
- Medium: L = 11.3 t/capita
- High: L = 14 t/capita

Population data: OWID historical, UN SSP2 projections (2009–2100).

### Step 3 — Dynamic Stock Model (stock-driven)
For each sector I run a stock-driven DSM (Pauliuk et al., ODYM framework). Given the target stock time series and a normally-distributed lifetime (CV = 0.3), the DSM back-calculates the annual inflow (new steel needed) and outflow (end-of-life scrap). The negative-inflow correction is applied so the model handles periods of declining stock gracefully.

Results cover 1700–2100. Only 2022–2100 is used for the forward analysis.

### Step 4 — Scrap supply and primary steel demand
Outflow from all four sectors is multiplied by a scrap collection efficiency that rises via S-curve from 85% in 2022 to a scenario-specific ceiling (`scrap_end`) by 2050. Collected scrap feeds EAF (electric arc furnace) scrap-based production.

Primary steel demand = total annual inflow − scrap supply (floored at zero).

### Step 5 — H-DRI penetration
H-DRI share of primary steel follows an S-curve from 2% in 2022 to `dri_share_2050` by 2050 (S-curve midpoint fixed at 2038, steepness 0.3). The remainder of primary production stays as BF-BOF.

In the sensitivity analysis I use a different parameterisation (`bfbof_share_2050`) where H-DRI and BF-BOF are both expressed as shares of *total* steel, and EAF takes the remainder.

### Step 6 — CO₂ emissions
- **BF-BOF:** fixed emission factor of 2.3 t CO₂/t steel (all scenarios)
- **H-DRI and EAF:** electricity-based, calculated as:
  `EF = (eaf_efficiency kWh/t) × (grid CO₂ g/kWh) / 1e6 + 0.002 × (44/12)`
  where the second term is graphite electrode combustion.

Grid CO₂ intensity follows one of three IEA trajectories (interpolated annually via PCHIP):
- Current Policies: 446 → 192 g/kWh by 2050 (Baseline, DRI scenarios)
- Stated Policies: 446 → 122 g/kWh by 2050 (Technology scenario)
- Net Zero: 446 → 0 g/kWh by 2050 (Full Transition scenario)

Beyond 2050, grid CO₂ is held constant at the 2050 value.

### Step 7 — Hydrogen demand
```
H₂ demand (Mt/yr) = DRI production (t/yr) × 54 kg H₂/t steel / 1e9
```
Fixed conversion factor across all scenarios. Split by market share: AEC 55%, PEMWE 35%, Other 10%.

### Step 8 — Electrolyser capacity
H₂ demand by technology type is converted to required installed capacity (GW):
```
Capacity (GW) = H₂ demand (Mt/yr) × 1e9 kg/Mt × efficiency (kWh/kg) / available hours (h/yr) / 1e6
```
Available hours = (8760 − 11×24) × (1 − 0.03) = 8241 h/yr, accounting for maintenance downtime and unplanned outages.

A second stock-driven DSM is then run with electrolyser GW as the stock and stack lifetime (in years) as the lifetime distribution. This gives annual capacity additions (GW/yr): the sum of new capacity to meet growing demand and replacement capacity for retired stacks.

Stack lifetimes (hours) converted to years at 8241 h/yr:
- AEC: 90,000 h ≈ 10.9 yr
- PEMWE: 70,000 h ≈ 8.5 yr
- Other: 45,000 h ≈ 5.5 yr

### Step 9 — Critical material demand
Annual capacity additions (GW/yr) drive material demand:
- **Nickel** (AEC): fixed intensity 800 kg/MW → kt/yr
- **Platinum** (PEMWE): time-varying via PCHIP (0.45 g/kW in 2022 → 0.25 g/kW by 2030, held thereafter)
- **Iridium** (PEMWE): time-varying via PCHIP (0.55 g/kW in 2022 → 0.30 g/kW by 2030, held thereafter)

All results are saved to a `.pkl` file per scenario under `electrolysers/data/processed/`.

---

## Scenarios

Twelve scenarios = 3 demand levels × 4 ambition levels:

| Scenario type | H-DRI by 2050 | Scrap ceiling | Grid CO₂ trajectory |
|---|---|---|---|
| Baseline | 15% of primary | 85% | IEA Current Policies |
| DRI Scale-Up | 50% of primary | 90% | IEA Current Policies |
| Technology | 50% of primary | 90% | IEA Stated Policies |
| Full Transition | 100% of primary | 95% | IEA Net Zero |

Each is run at Low (L=6), Medium (L=11.3), and High (L=14) demand.

---

## File structure

```
master/
├── README.md
├── electrolysers/
│   ├── raw_processing/
│   │   ├── dynamic_stock_model.py         ← DSM class (Pauliuk et al., ODYM — not modified)
│   │   ├── model_functions.py             ← all model logic; imported by every scenario notebook
│   │   ├── low_baseline.ipynb             ┐
│   │   ├── low_dri.ipynb                  │
│   │   ├── low_tech.ipynb                 │
│   │   ├── low_full.ipynb                 │
│   │   ├── med_baseline.ipynb             │  12 scenario notebooks
│   │   ├── med_dri.ipynb                  │  (each imports model_functions.py,
│   │   ├── med_tech.ipynb                 │   sets parameters, calls run_scenario(),
│   │   ├── med_full.ipynb                 │   saves result to data/processed/)
│   │   ├── high_baseline.ipynb            │
│   │   ├── high_dri.ipynb                 │
│   │   ├── high_tech.ipynb                │
│   │   ├── high_full.ipynb                ┘
│   │   ├── plots.ipynb                    ← loads all 12 .pkl files, generates thesis figures
│   │   └── sensitivity.ipynb              ← Morris screening sensitivity analysis (3 outputs)
│   ├── gen_appendix.py                    ← generates LaTeX appendix tables from .pkl results
│   └── data/
│       ├── raw/                           ← input data (steel stock, population, REMIND, material intensities)
│       └── processed/                     ← model output .pkl files (one per scenario + sensitivity runs)
└── visualizations/                        ← saved figure .png files
```

---

## How to run

1. **Run scenario notebooks** — open each of the 12 `{low,med,high}_{baseline,dri,tech,full}.ipynb` notebooks in `raw_processing/` and run all cells. Order does not matter. Each notebook saves a `.pkl` file to `data/processed/`.

2. **Generate figures** — open `plots.ipynb` and run all cells. Figures are displayed inline and can be saved by uncommenting the `fig.savefig(...)` lines.

3. **Sensitivity analysis** — open `sensitivity.ipynb` and run all cells. Runs ~700 model evaluations (Morris sampling, N=100, 3 analyses).

4. **Appendix tables** — from the `electrolysers/` directory, run:
   ```
   python gen_appendix.py
   ```
   This writes LaTeX-formatted tables to `data/processed/appendix_tables.tex`.

---

## Data sources

| Data | Source |
|---|---|
| Historical steel stock & sector shares | Mueller et al. (2011); Pauliuk et al. (2013) |
| Historical population | Our World in Data (OWID) |
| Future population | UN World Population Prospects, SSP2 |
| H₂ supply reference scenarios | REMIND IAM model (SSP1, SSP2 variants) |
| Grid CO₂ trajectories | IEA World Energy Outlook 2024 |
| BF-BOF emission factor | worldsteel (2023) |
| EAF electricity consumption | IEA (2023) |
| Electrolyser efficiencies & lifetimes | IRENA (2020); IEA (2022) |
| Nickel intensity (AEC) | Schmidt et al. (2017) |
| Platinum & iridium intensities (PEMWE) | Barber et al. (2021); IRENA (2020) |
| Iridium supply constraint | Johnson Matthey (2024); WPIC (2023) |
| Nickel supply | INSG (2025) |
| Platinum supply | USGS (2025) |
| Historic steel production validation | World Steel Association (WSA), 2022–2025 |
