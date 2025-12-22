"""
    DataLoader

Input data parsing and validation for the FLEXECOM model.

This module handles loading of scenario definitions and time-series input data
from CSV and Excel files. Data sources are configured through a TOML catalog
file (`data_catalog.toml`) that maps logical dataset names to physical files.

# Public Functions
- `load_scenarios`: Parse scenario CSV files
- `load_input_data`: Load all hourly time series and technology parameters

See `data/inputs/README.md` for detailed format specifications.
"""
module DataLoader

using CSV, XLSX, DataFrames, Dates, TOML
using ..Types
using ..Constants

export load_scenarios, load_input_data

"""
    load_scenarios(scenario_file::String) -> Vector{Scenario}

Load optimization scenarios from a CSV file.

Each row in the CSV defines one scenario with technology deployment parameters.
Supports both multi-column format (one column per parameter) and single-column
semicolon-separated format.

# Arguments
- `scenario_file`: Path to the CSV file

# Returns
- Vector of `Scenario` structs ready for optimization

# Required Columns
`COEF`, `PV`, `GRID`, `BESS_price`, `BESS_cap`, `EVnum`, `ASHP`, `DHW`

# Example
```julia
scenarios = load_scenarios("data/scenarios/scenarios_sample.csv")
```
"""
function load_scenarios(scenario_file)
    # Load scenarios from CSV
    scenarios_df = CSV.read(scenario_file, DataFrame)
    scenarios = Scenario[]
    
    for (i, row) in enumerate(eachrow(scenarios_df))
        # Check if we have a single column with semicolon-separated values
        if ncol(scenarios_df) == 1
            # Parse semicolon-separated values from a single column
            values = split(String(row[1]), ";")
            if length(values) < 8
                error("Not enough values in CSV row: $(row[1])")
            end

            coef = parse(Int, values[1])
            pv = parse(Float64, values[2])
            grid = parse(Float64, values[3])
            bess_price = parse(Float64, values[4])
            bess_cap = parse(Float64, values[5])
            evnum = parse(Int, values[6])
            ashp = parse(Float64, values[7])
            dhw = parse(Int, values[8])
        else
            # Multiple columns, each with its own value
            required_cols = [:COEF, :PV, :GRID, :BESS_price, :BESS_cap, :EVnum, :ASHP, :DHW]
            if !all(col -> col in propertynames(row), required_cols)
                missing_cols = filter(col -> !(col in propertynames(row)), required_cols)
                error("Missing required scenario columns: $(join(string.(missing_cols), ", "))")
            end

            coef = row.COEF
            pv = row.PV
            grid = row.GRID
            bess_price = row.BESS_price
            bess_cap = row.BESS_cap
            evnum = row.EVnum
            ashp = row.ASHP
            dhw = row.DHW
        end

        # Create scenario name
        name = "COEF-$(coef)_PV-$(pv)kWp_BESScap-$(bess_cap)kWh_BESSprice-$(bess_price)_GRID-$(grid)_EVnum-$(evnum)_ASHP-$(ashp)_DHW-$(dhw)"

        push!(scenarios, Scenario(
            i, name, coef, pv, grid, bess_price, bess_cap, evnum, ashp, dhw
        ))
    end

    return scenarios
end

function _catalog_path(input_dir, filename)
    catalog_path = joinpath(input_dir, filename)
    if !isfile(catalog_path)
        error("Input catalog not found at $(catalog_path). Copy data_catalog.example.toml and edit it to describe your datasets.")
    end
    return catalog_path
end

function _as_char(value, default)
    value === nothing && return default
    value isa Char && return value
    value isa AbstractString && return only(value)
    error("Delimiter must be a single character, got $(value)")
end

function _load_csv_matrix(path; delim=',', decimal='.', types=Float64)
    df = CSV.read(path, DataFrame;
        delim=_as_char(delim, ','),
        decimal=_as_char(decimal, '.'),
        types=types,
    )
    return types.(Matrix(df))
end

function _load_table(cfg, input_dir; types=Float64)
    fmt = lowercase(get(cfg, "format", "xlsx"))
    path = joinpath(input_dir, cfg["file"])
    if !isfile(path)
        error("Required input file $(path) not found. See data/inputs/README.md for details.")
    end

    if fmt == "csv"
        delim = get(cfg, "delimiter", ",")
        decimal = get(cfg, "decimal", ".")
        return _load_csv_matrix(path; delim=delim, decimal=decimal, types=types)
    elseif fmt == "xlsx"
        sheet = get(cfg, "sheet", nothing)
        range = get(cfg, "range", nothing)
        sheet === nothing && error("Missing 'sheet' entry for $(cfg["file"]).")
        range === nothing && error("Missing 'range' entry for $(cfg["file"]).")
        return types.(XLSX.readdata(path, "$(sheet)!$(range)"))
    else
        error("Unsupported format '$(fmt)' for $(cfg["file"]). Use 'csv' or 'xlsx'.")
    end
end

"""
    load_input_data(input_dir; catalog_filename="data_catalog.toml") -> Dict

Load all input time series and parameters required for optimization.

Reads the data catalog TOML file and loads each dataset according to its
format specification. Performs validation to ensure all hourly series have
matching lengths (8760 hours for a standard year).

# Arguments
- `input_dir`: Directory containing input files and the data catalog
- `catalog_filename`: Name of the TOML catalog file (default: "data_catalog.toml")

# Returns
Dict with the following keys:
- `:PI_pur`, `:PI_sell`: Grid purchase and sale prices (€/kWh)
- `:PI_cont`: Grid power term price (€/kW/year)
- `:gen_CF`: PV capacity factors (p.u.)
- `:load_DC`: Electricity demand matrix [T × J] (kWh)
- `:load_Thermal`: Thermal demand matrix [T × J] (kWh, positive=heating, negative=cooling)
- `:weekday`: Day-type classification for variable coefficients
- `:EV_*`: Electric vehicle availability patterns
- `:Temp_out`: Outdoor temperatures (°C)
- `:COP_Heating`, `:COP_Cooling`: Heat pump COPs at each hour
- `:DHW_*`: Domestic hot water parameters

# Example
```julia
data = load_input_data("data/inputs")
println("Loaded \$(size(data[:load_DC], 2)) households")
```
"""
function load_input_data(input_dir; catalog_filename="data_catalog.toml")
    # Load all input data (load curves, prices, etc.)
    catalog = TOML.parsefile(_catalog_path(input_dir, catalog_filename))
    data = Dict()

    # Grid prices
    prices_cfg = catalog["grid_prices"]
    prices_matrix = _load_table(prices_cfg, input_dir)
    col_names = get(prices_cfg, "columns", ["PI_pur", "PI_sell"])
    prices_df = DataFrame(prices_matrix, Symbol.(col_names))
    data[:PI_pur] = prices_df[:, Symbol(get(prices_cfg, "purchase_column", "PI_pur"))]
    data[:PI_sell] = prices_df[:, Symbol(get(prices_cfg, "sell_column", "PI_sell"))]
    data[:PI_cont] = GRID_TERM_POWER  # €/kW/year

    # PV generation
    pv_cfg = catalog["pv_generation"]
    pv_matrix = _load_table(pv_cfg, input_dir)
    pv_names = get(pv_cfg, "columns", ["PV1"])
    pv_df = DataFrame(pv_matrix, Symbol.(pv_names))
    data[:gen_CF] = pv_df[:, Symbol(get(pv_cfg, "capacity_factor_column", "PV1"))]

    # Electric demand
    load_cfg = catalog["electric_demand"]
    data[:load_DC] = _load_table(load_cfg, input_dir)

    # Thermal demand
    thermal_cfg = catalog["thermal_demand"]
    data[:load_Thermal] = _load_table(thermal_cfg, input_dir)

    # Weekday pattern (for variable coefficients)
    weekday_cfg = catalog["weekday_pattern"]
    data[:weekday] = Int.(_load_table(weekday_cfg, input_dir; types=Float64))

    # EV patterns
    ev_cfg = catalog["ev_patterns"]
    data[:EV_o] = _load_table(ev_cfg["EV_o"], input_dir)
    data[:EV_km] = _load_table(ev_cfg["EV_km"], input_dir)
    data[:EV_lleg] = _load_table(ev_cfg["EV_lleg"], input_dir)
    data[:EV_h_sal] = _load_table(ev_cfg["EV_h_sal"], input_dir)
    data[:EV_SoC_min] = _load_table(ev_cfg["EV_SoC_min"], input_dir)

    # Temperatures and COP calculation
    temperature_cfg = catalog["temperatures"]
    data[:Temp_out] = vec(_load_table(temperature_cfg, input_dir))

    # Calculate COPs using Vaillant VAIL 1-025 formulas from MATLAB
    T = length(data[:Temp_out])
    data[:COP_Heating] = zeros(T)
    data[:COP_Cooling] = zeros(T)

    for t = 1:T
        temp_celsius = data[:Temp_out][t]  # Already in Celsius
        data[:COP_Heating][t] = cop_from_temp(COP_HEATING_COEFFS, temp_celsius)
        data[:COP_Cooling][t] = cop_from_temp(COP_COOLING_COEFFS, temp_celsius)
    end

    # DHW demand
    dhw_cfg = catalog["dhw_profile"]
    dhw_matrix = _load_table(dhw_cfg, input_dir)
    data[:DHW_flux_base] = vec(dhw_matrix)

    # DHW parameters
    data[:DHW_COP] = DHW_COP
    data[:DHW_max] = DHW_MAX_LITERS  # Liters
    data[:Tw_dhw] = DHW_SUPPLY_TEMPERATURE_K  # K
    data[:Tw_in] = DHW_INLET_TEMPERATURE_K  # K
    data[:CP_w] = WATER_SPECIFIC_HEAT_KJKG  # kJ/K/kg

    # Get number of households from load data
    T_load, J = size(data[:load_DC])

    # Create DHW flux for all households and repeat for all days
    hours_per_day = length(data[:DHW_flux_base])
    days = ceil(Int, T_load / hours_per_day)
    dhw_daily = reshape(data[:DHW_flux_base], hours_per_day, 1)
    dhw_full = repeat(dhw_daily, days, J)
    data[:DHW_flux] = dhw_full[1:T_load, :]

    # Convert from liters to kWh
    energy_per_liter = DHW_ENERGY_PER_LITER
    data[:DHW_flux] = data[:DHW_flux] .* energy_per_liter
    data[:DHW_max] = data[:DHW_max] * energy_per_liter

    # Ensure all hourly vectors match the load horizon
    _validate_series_length(data, T_load)

    return data
end

function _validate_series_length(data, expected_length)
    keys_to_check = [:PI_pur, :PI_sell, :gen_CF, :load_Thermal, :weekday, :EV_o,
                     :EV_km, :EV_lleg, :EV_h_sal, :EV_SoC_min, :Temp_out]
    for key in keys_to_check
        series = data[key]
        length(series) == expected_length || size(series, 1) == expected_length ||
            error("Dataset $(key) has length $(size(series, 1)) but expected $(expected_length). Check your input files.")
    end
end

end # module
