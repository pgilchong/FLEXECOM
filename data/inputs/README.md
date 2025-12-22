# Input data reference

This directory bundles sample datasets and configuration templates that document the
hourly inputs expected by FLEXECOM. The optimisation can only run once every dataset
listed in `data_catalog.toml` is available.

## Workflow overview

1. **Create `data_catalog.toml`** by copying `data_catalog.example.toml` and editing the
   file names, sheet names and ranges so that they match your own data exports. All
   paths are interpreted relative to this folder.
2. **Replace or augment the datasets** described below. The repository only includes
   non-private information and a synthetic load placeholder so that the scripts can be
   executed end-to-end without disclosing smart-meter data.
3. **Keep private data out of version control.** If you generate bespoke CSV/XLSX files,
   add them to your copy of the catalog but do not commit them.

## Dataset catalogue

| File | Status | Format | Description |
|------|--------|--------|-------------|
| `sample_load_profiles.csv` | Synthetic placeholder (replace with private data) | CSV, 8,760 rows × household columns | Hourly electricity demand for each point of consumption. Generated algorithmically for demonstration purposes. |
| `thermal_demand_ninja_run4.xlsx` | Provided (propio, non-private) | Excel `Thermal_Load` sheet | Hourly thermal demand per dwelling. Positive values = heating, negative = cooling. |
| `DHW_Residential.csv` | Provided (propio, non-private) | CSV single column | Typical daily domestic hot water usage in litres. The model repeats the 24 hours across the horizon and converts litres to kWh internally. |
| `EV_pattern_50.xlsx` | Generated | Excel with `EV_o`, `EV_km`, `EV_lleg`, `EV_h_sal`, `EV_SoC_min` sheets | Availability and minimum SoC constraints for up to 50 EVs. Create alternative patterns with `scripts/generate_ev_pattern.jl`. |
| `gen_FV21.csv` | Provided (propio, non-private) | CSV | Hourly PV capacity factors in per unit. |
| `prices_grid_2021.csv` | Provided (propio, non-private) | CSV with `;` delimiter | Hourly purchase (`PI_pur`) and compensation (`PI_sell`) tariffs in €/kWh. |
| `temperatures_valencia.xlsx` | Provided (propio, non-private) | Excel `Hoja3` sheet | Outdoor temperature in °C used to compute the COP of ASHPs. |
| `weekday21.xlsx` | Generated | Excel `weekday` sheet | Seasonal/day-type classifier for variable sharing coefficients. Regenerate via `scripts/generate_weekday_pattern.jl`. |

## Electricity demand (private input)

The repository ships with `sample_load_profiles.csv`, an entirely synthetic 8,760-hour dataset
for five households. This is **not** based on real smart-meter data and only exists to illustrate
the expected format. Replace it with your own measurements:

- Use one column per household/load point.
- Provide 8,760 (non-leap year) or 8,784 (leap year) rows. The model validates that every other
  dataset has the same horizon.
- Decimal separator must match the configuration in `data_catalog.toml` (default `.`).
- Update the `electric_demand` section in the catalog when renaming the file or changing its format.

## EV availability matrices (generated input)

`EV_pattern_50.xlsx` was produced with a stochastic method that mimics evening arrivals and
morning departures. To create your own file, run:

```bash
julia --project scripts/generate_ev_pattern.jl data/inputs 50
```

The second argument sets the number of EVs (columns). Adjust the sheet ranges in the catalog if
you pick a different value.

## Weekday pattern (generated input)

`weekday21.xlsx` encodes the seasonal/day-type cluster for each hour. It can be regenerated for
any year (including leap years) with:

```bash
julia --project scripts/generate_weekday_pattern.jl data/inputs 2024
```

Column 1 enumerates the 8 profiles (winter/spring/summer/autumn × weekday/weekend), column 2 is
the hour-of-day (1–24) and column 3 enumerates the unique coefficient set (1–192).

## Thermal demand and temperature series (provided inputs)

The thermal and temperature series originate from Renewables.ninja simulations and public
weather records. They can remain in the repository. If you replace them with other locations or
years, update the sheet ranges in the catalog accordingly.

## Domestic hot water profile (provided input)

`DHW_Residential.csv` contains 24 hourly water consumption values in litres for a representative
day. FLEXECOM expands this profile over the simulation horizon and internally converts litres to
kWh using the thermodynamic parameters defined in `src/constants.jl`.

## Grid price assumptions (provided input)

`prices_grid_2021.csv` includes:

- `PI_pur`: energy purchase price for the user (€/kWh).
- `PI_sell`: remuneration for surplus energy exported to the grid (€/kWh).

Both series should align with the regulation and tariff structure relevant to your case study.

