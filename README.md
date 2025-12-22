# FLEXECOM: Flexible Energy Community Optimisation Model

FLEXECOM is the Julia implementation of the multi-energy optimisation framework developed during Álvaro Manso Burgos's doctoral research on Renewable Energy Communities (RECs). The model minimises annual system costs while co-optimising electricity production, storage, electric mobility, thermal demand and domestic hot water. The underlying formulation is the one documented in Manso Burgos et al. (2025, *Applied Energy*), and this repository contains the operational tool used to generate the results in that work rather than a didactic example derived from it.

## Key capabilities

- **Hourly techno-economic optimisation** over an entire year using JuMP and the Gurobi solver.
- **Configurable asset portfolio** that covers PV, battery energy storage (BESS), electric vehicles (EVs), air-source heat pumps (ASHPs) and domestic hot water (DHW) systems.
- **Scenario-based analysis**: users can explore different investment and tariff assumptions through CSV-based scenario definitions.
- **Financial evaluation** including NPV, IRR, investment shares and operating cost savings for both the community and individual members.
- **Price sensitivity sweeps** that evaluate all combinations of PV, fuel, natural-gas and EV price assumptions defined in `src/constants.jl`.
- **Rich result exports** in CSV, JLD2, Excel and MATLAB formats.
- **Visualization suite** with publication-quality plots and automated HTML reports.

## Repository layout

```
FLEXECOM/
├── src/                  # Core model modules
│   ├── FLEXECOM.jl       # Public API and orchestration logic
│   ├── data_loader.jl    # CSV/XLSX parsers for demand, generation and tariffs
│   ├── optimizer.jl      # JuMP model of the REC operation
│   ├── financial.jl      # Techno-economic post-processing (NPV, IRR, etc.)
│   └── utils.jl          # Result management, summaries and exports
├── scripts/
│   └── run_model.jl      # CLI entry point for batch execution
├── data/
│   ├── inputs/           # Sample input data (hourly series and technology profiles)
│   ├── scenarios/        # Scenario definitions (CSV)
│   └── outputs/          # Folder where results are created
├── test/                 # Unit tests and regression checks
└── Project.toml          # Julia environment definition
```

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Julia     | 1.8 or newer | Use the official binaries from [julialang.org](https://julialang.org/downloads/). |
| Gurobi    | 10.0 or newer | Any edition works as long as the license is active. Configure `GUROBI_HOME` before running the model. |
| Packages  | See `Project.toml` | Install automatically via `Pkg.instantiate()`. |

## Quickstart

1. **Clone and activate the project**
   ```bash
   git clone https://github.com/<your-org>/FLEXECOM.git
   cd FLEXECOM
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

2. **Describe your input datasets** using the catalog mechanism. Review
   `data/inputs/data_catalog.toml` (copy it from `data_catalog.example.toml` if you are starting from
   scratch) and adjust the filenames, sheet names and column ranges so they match your own data exports.
   See [Input data specification](#input-data-specification) for the expected structure of every dataset.

3. **Run the automated test suite** (recommended after any change)
   ```bash
   julia --project test/runtests.jl
   ```

4. **Execute the sample scenario** using the CLI helper. The optional fourth argument controls the verbosity level (0 = silent, 1 = progress, 2 = detailed tracing).
   ```bash
   julia --project scripts/run_model.jl data/scenarios/scenarios_sample.csv data/inputs data/outputs 1
   ```
   Output folders follow the pattern `data/outputs/YYYYMMDD_HHMMSS` and contain:
   - `report.html` – Interactive HTML report with all plots
   - `summary.csv` – Financial metrics for all scenarios
   - `rec_model_results.jld2` – Native Julia serialization
   - `rec_model_results.mat` – MATLAB-compatible export
   - `scenario_N/` – Per-scenario Excel, CSV and plot files
   - `summary_plots/` – Cross-scenario comparison charts

5. **Use the library programmatically** inside Julia:
   ```julia
   using FLEXECOM

   scenario_file = "data/scenarios/scenarios_sample.csv"
   input_dir     = "data/inputs"
   output_dir    = "data/outputs"

   results, financials, summary = FLEXECOM.run_full_model(
       scenario_file,
       input_dir,
       output_dir;
       verbose = 2,
       export_excel = true,
       export_matlab = true,
       generate_plots = true,
   )
   ```
   The `verbose` keyword accepts `0`, `1` or `2` and mirrors the CLI behaviour. Set `generate_plots = true` to produce visualization charts and the HTML report.

## Preparing custom studies

1. **Scenarios** – create CSV files with the columns `COEF, PV, GRID, BESS_price, BESS_cap, EVnum, ASHP, DHW`. Each row represents one optimisation run. Price assumptions are no longer embedded in the scenarios; every execution sweeps all combinations derived from the `PRICE_SENSITIVITY` constants in `src/constants.jl`.
2. **Demand and generation profiles** – provide hourly time series for loads, PV generation, temperatures, EV usage and DHW demand matching the formats in `data/inputs`.
3. **Solver configuration** – adjust `DEFAULT_GUROBI_PARAMS` in `src/constants.jl` if different numerical settings are required.
4. **Financial parameters** – update `FINANCIAL_DEFAULT` (discount rate, CAPEX assumptions, O&M factors) to reflect the regulatory context of your study.

The scripts `scripts/generate_ev_pattern.jl` and `scripts/generate_weekday_pattern.jl` help create EV availability matrices and weekday coefficients consistent with the optimisation inputs.

## Visualization outputs

When `generate_plots = true`, the model produces publication-quality figures (300 DPI) for each scenario:

| Plot | Description |
|------|-------------|
| `daily_profiles.png` | Hourly generation, load and grid exchange for a sample week |
| `monthly_energy.png` | Monthly energy balance breakdown |
| `energy_heatmap.png` | 24-hour × 365-day heatmap of net energy flow |
| `energy_indicators.png` | Self-consumption ratio and self-sufficiency metrics |
| `bess_soc_annual.png` | Annual battery state-of-charge profile (if BESS installed) |
| `ev_soc_weekly.png` | Weekly EV charging patterns (if EVs present) |
| `price_profile.png` | Electricity tariff structure |

Cross-scenario comparison plots in `summary_plots/`:
- NPV and IRR bar charts
- Investment breakdown by technology
- Cost reduction percentages
- NPV vs IRR scatter plot

An HTML report (`report.html`) aggregates all figures with summary statistics for stakeholder presentations.

## Input data specification

All time series are referenced through `data/inputs/data_catalog.toml`. Copy the example file provided in
the repository and update the file paths or spreadsheet ranges to point at your datasets. The table below
summarises the expected format and source of each input:

| Dataset | Default file | Source | Format | Notes |
|---------|--------------|--------|--------|-------|
| Electricity prices | `prices_grid_2021.csv` | Author provided | CSV with `PI_pur`, `PI_sell` | Hourly purchase and compensation price in €/kWh. Delimiter `;`. |
| PV capacity factors | `gen_FV21.csv` | Author provided | CSV column `PV1` | Hourly PV capacity factor in p.u. |
| Electricity demand | `sample_load_profiles.csv` | Synthetic placeholder | CSV with one column per point of consumption | Replace with your smart-meter exports. 8,760 rows × `J` columns, decimal `.`. |
| Thermal demand | `thermal_demand_ninja_run4.xlsx` | Author provided | Excel `Thermal_Load!A1:Y8760` | Positive = heating, negative = cooling, one column per dwelling. |
| DHW profile | `DHW_Residential.csv` | Author provided | CSV single column | 24 hourly litres for a representative day. The model replicates it across the horizon. |
| Temperature | `temperatures_valencia.xlsx` | Author provided | Excel `Hoja3!A1:A8760` | Outdoor temperature used to build COP time series. |
| EV availability | `EV_pattern_50.xlsx` | Generated | Excel sheets `EV_o`, `EV_km`, `EV_lleg`, `EV_h_sal`, `EV_SoC_min` | Built via `scripts/generate_ev_pattern.jl`. Ranges default to `A1:AX8760` for 50 EVs. |
| Weekday profile | `weekday21.xlsx` | Generated | Excel `weekday!A1:C8760` | Built via `scripts/generate_weekday_pattern.jl`. Handles leap years. |

Additional guidance for working with private or generated data lives in `data/inputs/README.md`.

## Testing philosophy

The `test/` folder contains component-level checks for data parsing, financial calculations, utility helpers and energy-balance bookkeeping. The suite builds lightweight synthetic scenarios to ensure reproducible validation without requiring the full 8,760-hour datasets. Run `julia --project test/runtests.jl` regularly to catch regressions.

## Citation and contact

If FLEXECOM supports your research, please cite the underlying methodology and the doctoral work by Álvaro Manso Burgos (UPV Chair in Urban Energy Transition). For questions, collaborations or bug reports, open an issue or contact the author at `almanbur@upv.es`.

## References

Á. Manso Burgos et al., "Flexible energy community optimisation for the Valencian Community," *Applied Energy* (2025). Available at: https://www.sciencedirect.com/science/article/pii/S030626192500902X
