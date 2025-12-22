"""
    Utils

Result management, export, and visualization utilities for FLEXECOM.

This module handles post-optimization tasks including saving results to various
formats (JLD2, CSV, Excel, MATLAB), generating summary reports, and creating
visualization plots.

# Supported Export Formats
- **JLD2**: Native Julia serialization for fast reload
- **CSV**: Per-scenario time series and summary files
- **Excel**: Multi-sheet workbooks matching MATLAB output structure
- **MATLAB (.mat)**: N-dimensional arrays for cross-scenario analysis

# Public Functions
- `save_results`: Export all results in multiple formats
- `load_results`: Reload results from JLD2 file
- `create_summary`: Generate summary DataFrame across scenarios
- `verify_energy_balance`: Check energy conservation
- `export_to_excel`: Export single scenario to Excel
- `export_to_matlab`: Export all scenarios to MATLAB format
- `generate_plots`: Create visualization plots (optional)
"""
module Utils

using JLD2, DataFrames, CSV, XLSX, Dates, MAT, Plots, StatsPlots, Statistics
using ..Types
using ..Constants

export save_results, load_results, create_summary, verify_energy_balance, export_to_excel,
       export_to_matlab, generate_plots, generate_html_report

function _as_excel_table(data; name_prefix="Column")
    if data isa DataFrame
        return data
    elseif data isa AbstractMatrix
        matrix = Float64.(data)
        col_names = Symbol[]
        for idx in 1:size(matrix, 2)
            push!(col_names, Symbol("$(name_prefix)_$idx"))
        end
        return DataFrame(matrix, col_names)
    elseif data isa AbstractVector
        vector = Float64.(data)
        return DataFrame(Symbol(name_prefix) => vector)
    elseif data isa Number
        return DataFrame(Symbol(name_prefix) => [Float64(data)])
    else
        return DataFrame(Symbol(name_prefix) => [data])
    end
end

function _write_numeric_sheet!(xf, sheet_name, data; name_prefix=sheet_name)
    sheet = XLSX.addsheet!(xf, sheet_name)
    table = _as_excel_table(data; name_prefix=name_prefix)
    XLSX.writetable!(sheet, table)
end

"""
    save_results(scenario_results, financial_results, output_dir; kwargs...) -> String

Export optimization and financial results in multiple formats.

Creates a timestamped output directory containing JLD2, CSV, and optionally
Excel, MATLAB, and plot files for all scenarios.

# Arguments
- `scenario_results`: Dict of optimization results keyed by scenario ID
- `financial_results`: Dict of financial analysis results
- `output_dir`: Base directory for outputs

# Keyword Arguments
- `export_excel::Bool=true`: Generate Excel files per scenario
- `export_matlab::Bool=true`: Generate combined MATLAB file
- `generate_plots::Bool=false`: Generate visualization plots

# Returns
Path to the JLD2 file for quick reload.

# Output Structure
```
output_dir/YYYYMMDD_HHMMSS/
├── rec_model_results.jld2
├── rec_model_results.mat (if export_matlab)
├── scenario_1/
│   ├── scenario_1_results.xlsx (if export_excel)
│   ├── scenario_info.csv
│   ├── costs.csv
│   ├── generation_load.csv
│   └── ...
└── scenario_2/
    └── ...
```

# Example
```julia
jld_path = save_results(results, financials, "data/outputs";
                        export_matlab=true, generate_plots=true)
```
"""
function save_results(scenario_results, financial_results, output_dir;
                      export_excel::Bool=true,
                      export_matlab::Bool=true,
                      generate_plots::Bool=false)
    # Save results in JLD2 format
    jld_file = joinpath(output_dir, "rec_model_results.jld2")
    jldopen(jld_file, "w") do file
        file["scenario_results"] = scenario_results
        file["financial_results"] = financial_results
        file["timestamp"] = now()
    end

    # Export to MATLAB format (.mat) with n-dimensional arrays
    if export_matlab
        mat_file = joinpath(output_dir, "rec_model_results.mat")
        export_to_matlab(scenario_results, financial_results, mat_file)
    end

    # Export data to Excel for each scenario
    for (scenario_id, results) in scenario_results
        scenario_dir = joinpath(output_dir, "scenario_$(scenario_id)")
        mkpath(scenario_dir)

        # Export in Excel format (optional)
        if export_excel
            excel_file = joinpath(scenario_dir, "scenario_$(scenario_id)_results.xlsx")
            export_to_excel(results, financial_results[scenario_id], excel_file)
        end

        # Always export CSV versions for easier access
        export_scenario_results(results, financial_results[scenario_id], scenario_dir)

        # Generate plots (optional)
        if generate_plots
            plots_dir = joinpath(scenario_dir, "plots")
            mkpath(plots_dir)
            generate_scenario_plots(results, financial_results[scenario_id], plots_dir)
        end
    end

    # Generate summary plots across all scenarios
    if generate_plots && length(scenario_results) > 1
        plots_dir = joinpath(output_dir, "summary_plots")
        mkpath(plots_dir)
        generate_summary_plots(scenario_results, financial_results, plots_dir)

        # Generate HTML report
        generate_html_report(scenario_results, financial_results, output_dir)
    end

    return jld_file
end

function export_to_excel(results, financial_results, filename)
    # Create Excel file with sheets matching MATLAB output format
    XLSX.openxlsx(filename, mode="w") do xf
        # Demands sheet
        _write_numeric_sheet!(xf, "Demands", results[:load_DC]; name_prefix="Demand")

        # Generators sheet
        _write_numeric_sheet!(xf, "Generators", results[:gen_pow]; name_prefix="Generation")
        
        # ESCENARIOS sheet (scenario parameters)
        scenario = results[:scenario]
        prices = get(financial_results, :baseline_prices, PRICE_SENSITIVITY_MEANS)
        pv_price = prices.PV_price
        fuel_price = prices.Fuel_price
        ng_price = prices.NG_price
        ev_price = prices.EV_price
        esc_sheet = XLSX.addsheet!(xf, "ESCENARIOS")
        esc_data = DataFrame(
            Parameter = ["COEF", "PV", "GRID", "BESS_price", "BESS_cap", "EVnum",
                        "ASHP", "DHW", "PV_price", "Fuel_price", "NG_price", "EV_price"],
            Value = [scenario.COEF, scenario.PV, scenario.GRID, scenario.BESS_price,
                    scenario.BESS_cap, scenario.EVnum, scenario.ASHP, scenario.DHW,
                    pv_price, fuel_price, ng_price, ev_price]
        )
        XLSX.writetable!(esc_sheet, esc_data)
        
        # Cost sheets
        _write_numeric_sheet!(xf, "Cost_REC_j", vec(results[:Cost_REC_j]); name_prefix="Cost")

        _write_numeric_sheet!(xf, "Cost_REC_j_LCOS", vec(results[:Cost_REC_j_LCOS]); name_prefix="Cost_LCOS")
        
        # Coefficient sheets
        _write_numeric_sheet!(xf, "Coeff_allo", results[:Coef_allo]; name_prefix="Coeff_allo")

        _write_numeric_sheet!(xf, "Coeff_surplus", results[:Coef_surp]; name_prefix="Coeff_surplus")
        
        # Power flow sheets
        _write_numeric_sheet!(xf, "P_pur", results[:P_pur])

        _write_numeric_sheet!(xf, "P_sold", results[:P_sold])

        _write_numeric_sheet!(xf, "P_gift", results[:P_gift])

        _write_numeric_sheet!(xf, "P_sc", results[:P_sc])

        _write_numeric_sheet!(xf, "P_cont", results[:P_cont])
        
        # BESS sheets
        _write_numeric_sheet!(xf, "BESS_SoC", results[:BESS_SoC])

        _write_numeric_sheet!(xf, "BESS_cha_pur", results[:BESS_cha_pur])

        _write_numeric_sheet!(xf, "BESS_cha_sc", results[:BESS_cha_sc])

        _write_numeric_sheet!(xf, "BESS_dis", results[:BESS_dis])
        
        # BESS parameters
        bess_param_sheet = XLSX.addsheet!(xf, "BESS_param")
        bess_param_data = DataFrame(
            Parameter = ["BESS_cap_uni", "BESS_cap", "BESS_num", "BESS_DoD", "BESS_Eff",
                        "BESS_V", "BESS_Pmax", "BESS_InvUni", "BESS_Cycles"],
            Value = [results[:BESS_COM].cap, results[:BESS_COM].cap * results[:BESS_num],
                    results[:BESS_num], results[:BESS_COM].DoD, results[:BESS_COM].Eff,
                    results[:BESS_COM].V, results[:BESS_COM].Pmax, results[:BESS_COM].Inv,
                    results[:BESS_COM].Cycles]
        )
        XLSX.writetable!(bess_param_sheet, bess_param_data)
        
        # EV sheets if applicable
        if results[:EVbool] == 1
            _write_numeric_sheet!(xf, "EV_Cost", results[:Cost_EV])

            _write_numeric_sheet!(xf, "EV_SoC", results[:EV_SoC])

            _write_numeric_sheet!(xf, "EV_km", results[:EV_km])

            _write_numeric_sheet!(xf, "P_pur_EVk", results[:P_pur_EVk])

            _write_numeric_sheet!(xf, "P_sc_EVk", results[:P_sc_EVk])

            _write_numeric_sheet!(xf, "P_sc_EVj", results[:P_sc_EVj])
        end
        
        # EV parameters
        ev_param_sheet = XLSX.addsheet!(xf, "EV_param")
        ev_param_data = DataFrame(
            Parameter = ["EV_cap", "EV_cons_km", "EV_DoD", "EV_Eff", "EV_Pch"],
            Value = [results[:EV].cap, results[:EV].cons_km, 0.2, 
                    results[:EV].Eff, results[:EV].Pch]
        )
        XLSX.writetable!(ev_param_sheet, ev_param_data)
        
        # ASHP sheets
        _write_numeric_sheet!(xf, "W_pur_heating", results[:W_pur_heating])

        _write_numeric_sheet!(xf, "W_pur_cooling", results[:W_pur_cooling])

        _write_numeric_sheet!(xf, "W_sc_heating", results[:W_sc_heating])

        _write_numeric_sheet!(xf, "W_sc_cooling", results[:W_sc_cooling])

        _write_numeric_sheet!(xf, "ASHP_capheating", results[:W_pur_heating] .+ results[:W_sc_heating])

        _write_numeric_sheet!(xf, "ASHP_capcooling", results[:W_pur_cooling] .+ results[:W_sc_cooling])
        
        # DHW sheets
        _write_numeric_sheet!(xf, "DHW_flux", results[:DHW_flux])

        _write_numeric_sheet!(xf, "DHW_Soc", results[:DHW_SoC])

        _write_numeric_sheet!(xf, "DHW_pur", results[:DHW_pur])

        _write_numeric_sheet!(xf, "DHW_sc", results[:DHW_sc])
    end
end

"""
    verify_energy_balance(results::Dict) -> Vector{Float64}

Verify energy conservation at each timestep.

Identical to `Optimizer.calculate_energy_balance` but accessible from the
Utils module for post-processing validation.

# Arguments
- `results`: Dict from `optimize_scenario`

# Returns
Vector of length T with energy balance residuals. Values should be
approximately zero for valid solutions.

See also: [`Optimizer.calculate_energy_balance`](@ref)
"""
function verify_energy_balance(results)
    # Verify energy balance at each timestep
    T = size(results[:gen_pow], 1)
    balance = zeros(T)
    
    BESS_eff = results[:BESS] == 1 ? results[:BESS_COM].Eff : 1.0
    
    for t = 1:T
        # Energy balance equation from MATLAB
        balance[t] = results[:gen_pow][t] + sum(results[:P_pur][t, :]) -
                     sum(results[:load_DC][t, :]) - sum(results[:P_sold][t, :]) - 
                     sum(results[:P_gift][t, :]) -
                     sum(results[:BESS_cha_pur][t, :] + results[:BESS_cha_sc][t, :]) / 
                     sqrt(BESS_eff) + sum(results[:BESS_dis][t, :]) * sqrt(BESS_eff) -
                     sum(results[:P_sc_EVj][t, :]) -
                     sum(results[:W_pur_heating][t, :] + results[:W_pur_cooling][t, :] + 
                         results[:W_sc_heating][t, :] + results[:W_sc_cooling][t, :]) -
                     sum(results[:DHW_pur][t, :] + results[:DHW_sc][t, :])
    end
    
    return balance
end

function export_scenario_results(results, financial_results, output_dir)
    # Create DataFrames for key results and export to CSV

    # General information
    prices = get(financial_results, :baseline_prices, PRICE_SENSITIVITY_MEANS)
    pv_price = prices.PV_price
    fuel_price = prices.Fuel_price
    ng_price = prices.NG_price
    ev_price = prices.EV_price

    scenario_info = DataFrame(
        Parameter = ["COEF", "PV", "GRID", "BESS_price", "BESS_cap", "EVnum", "ASHP", "DHW",
                    "PV_price", "Fuel_price", "NG_price", "EV_price"],
        Value = [
            results[:scenario].COEF,
            results[:scenario].PV,
            results[:scenario].GRID,
            results[:scenario].BESS_price,
            results[:scenario].BESS_cap,
            results[:scenario].EVnum,
            results[:scenario].ASHP,
            results[:scenario].DHW,
            pv_price,
            fuel_price,
            ng_price,
            ev_price
        ]
    )
    CSV.write(joinpath(output_dir, "scenario_info.csv"), scenario_info)
    
    # Cost results
    cost_results = DataFrame(
        Household = 1:size(results[:Cost_REC_j], 2),
        Cost = vec(results[:Cost_REC_j]),
        Cost_LCOS = vec(results[:Cost_REC_j_LCOS])
    )
    CSV.write(joinpath(output_dir, "costs.csv"), cost_results)

    # Time series data
    T = size(results[:gen_pow], 1)
    J = size(results[:load_DC], 2)
    
    # Generation and load
    gen_load = DataFrame(
        Hour = 1:T,
        Generation = results[:gen_pow],
        Load = [sum(results[:load_DC][t, :]) for t in 1:T]
    )
    CSV.write(joinpath(output_dir, "generation_load.csv"), gen_load)
    
    # Grid exchange
    grid_exchange = DataFrame(
        Hour = 1:T,
        Purchase = [sum(results[:P_pur][t, :]) for t in 1:T],
        Sale = [sum(results[:P_sold][t, :]) for t in 1:T]
    )
    CSV.write(joinpath(output_dir, "grid_exchange.csv"), grid_exchange)
    
    # BESS operation
    bess_operation = DataFrame(
        Hour = 1:T,
        SoC = results[:BESS_SoC],
        Charge_Grid = [sum(results[:BESS_cha_pur][t, :]) for t in 1:T],
        Charge_PV = [sum(results[:BESS_cha_sc][t, :]) for t in 1:T],
        Discharge = [sum(results[:BESS_dis][t, :]) for t in 1:T]
    )
    CSV.write(joinpath(output_dir, "bess_operation.csv"), bess_operation)
    
    # EV operation if applicable
    if results[:scenario].EVnum > 0
        ev_operation = DataFrame(
            Hour = 1:T,
            SoC = [results[:EV_SoC][t, 1] for t in 1:T],
            Charge_Grid = [sum(results[:P_pur_EVk][t, :]) for t in 1:T],
            Charge_PV = [sum(results[:P_sc_EVk][t, :]) for t in 1:T]
        )
        CSV.write(joinpath(output_dir, "ev_operation.csv"), ev_operation)
    end
    
    # ASHP operation if applicable
    if results[:scenario].ASHP > 0
        ashp_operation = DataFrame(
            Hour = 1:T,
            Heating_Grid = [sum(results[:W_pur_heating][t, :]) for t in 1:T],
            Heating_PV = [sum(results[:W_sc_heating][t, :]) for t in 1:T],
            Cooling_Grid = [sum(results[:W_pur_cooling][t, :]) for t in 1:T],
            Cooling_PV = [sum(results[:W_sc_cooling][t, :]) for t in 1:T]
        )
        CSV.write(joinpath(output_dir, "ashp_operation.csv"), ashp_operation)
    end
    
    # DHW operation if applicable
    if results[:scenario].DHW > 0
        dhw_operation = DataFrame(
            Hour = 1:T,
            SoC = [sum(results[:DHW_SoC][t, :]) for t in 1:T],
            Heating_Grid = [sum(results[:DHW_pur][t, :]) for t in 1:T],
            Heating_PV = [sum(results[:DHW_sc][t, :]) for t in 1:T]
        )
        CSV.write(joinpath(output_dir, "dhw_operation.csv"), dhw_operation)
    end
    
    # Allocation coefficients
    if results[:scenario].COEF == 1
        alloc_coeff = DataFrame(
            Household = 1:J,
            Coefficient = vec(results[:Coef_allo])
        )
    else
        # For time-varying coefficients, export a few sample days
        sample_days = DEFAULT_SAMPLE_DAYS  # Days to sample
        sample_hours = [(d-1)*24+1:d*24 for d in sample_days]
        
        alloc_coeff = DataFrame(Hour = Int[])
        for j in 1:J
            alloc_coeff[!, Symbol("Household_$j")] = Float64[]
        end

        for hours in sample_hours
            for t in hours
                row = Dict{Symbol, Any}(:Hour => t)
                for j in 1:J
                    value = if results[:scenario].COEF == 2
                        results[:Coef_allo][results[:weekday][t, 3], j]
                    else  # COEF == 3
                        results[:Coef_allo][t, j]
                    end
                    row[Symbol("Household_$j")] = value
                end
                push!(alloc_coeff, row)
            end
        end
    end
    CSV.write(joinpath(output_dir, "allocation_coefficients.csv"), alloc_coeff)
end

"""
    load_results(jld_file::String) -> Tuple

Reload optimization results from a JLD2 file.

# Arguments
- `jld_file`: Path to the JLD2 file created by `save_results`

# Returns
Tuple of (scenario_results, financial_results, timestamp)

# Example
```julia
results, financials, ts = load_results("data/outputs/20250101_120000/rec_model_results.jld2")
println("Results generated at: \$ts")
```
"""
function load_results(jld_file)
    # Load results from JLD2 file
    data = jldopen(jld_file, "r") do file
        Dict(
            "scenario_results" => file["scenario_results"],
            "financial_results" => file["financial_results"],
            "timestamp" => file["timestamp"]
        )
    end
    
    return data["scenario_results"], data["financial_results"], data["timestamp"]
end

"""
    create_summary(scenario_results, financial_results) -> DataFrame

Generate a summary DataFrame with key metrics for all scenarios.

Useful for comparing scenarios at a glance or exporting to a report.

# Arguments
- `scenario_results`: Dict of optimization results
- `financial_results`: Dict of financial analysis results

# Returns
DataFrame with columns including:
- Scenario identification (ID, name)
- Technology deployment (PV, BESS, EV, ASHP, DHW)
- Price assumptions
- Financial metrics (NPV, IRR, cost reduction %)
- Investment breakdown by technology

# Example
```julia
summary = create_summary(results, financials)
CSV.write("summary.csv", summary)
```
"""
function create_summary(scenario_results, financial_results)
    # Create summary DataFrame with key results
    summary = DataFrame(
        Scenario_ID = Int[],
        Scenario_Name = String[],
        COEF_Type = Int[],
        PV_Power = Float64[],
        Grid_Price = Float64[],
        BESS_Capacity = Float64[],
        BESS_Price = Float64[],
        EV_Number = Int[],
        ASHP_Power = Float64[],
        DHW = Int[],
        PV_Price = Float64[],
        Fuel_Price = Float64[],
        NG_Price = Float64[],
        EV_Price = Float64[],
        Total_Cost = Float64[],
        Initial_Cost = Float64[],
        Cost_Reduction_Pct = Float64[],
        Total_Investment = Float64[],
        NPV = Float64[],
        IRR = Float64[],
        PV_Investment_Share = Float64[],
        BESS_Investment_Share = Float64[],
        ASHP_Investment_Share = Float64[],
        DHW_Investment_Share = Float64[],
        EV_Investment_Share = Float64[]
    )
    
    for (scenario_id, fin_results) in financial_results
        scenario = scenario_results[scenario_id][:scenario]

        prices = get(fin_results, :baseline_prices, PRICE_SENSITIVITY_MEANS)
        pv_price = prices.PV_price
        fuel_price = prices.Fuel_price
        ng_price = prices.NG_price
        ev_price = prices.EV_price

        # Calculate total cost and reduction
        total_cost = sum(scenario_results[scenario_id][:Cost_REC_j])
        initial_cost = fin_results[:initial_costs][:total]
        
        # Calculate cost reduction percentage
        cost_reduction = (initial_cost - total_cost) / initial_cost * 100
        
        # Get NPV and IRR
        npv = fin_results[:npv][1]  # REC NPV
        irr = fin_results[:irr][1]  # REC IRR
        
        # Get investment shares
        inv_shares = fin_results[:investment_shares]
        
        push!(summary, (
            scenario_id,
            scenario.name,
            scenario.COEF,
            scenario.PV,
            scenario.GRID,
            scenario.BESS_cap,
            scenario.BESS_price,
            scenario.EVnum,
            scenario.ASHP,
            scenario.DHW,
            pv_price,
            fuel_price,
            ng_price,
            ev_price,
            total_cost,
            initial_cost,
            cost_reduction,
            fin_results[:investments][:total],
            npv,
            irr,
            inv_shares[:pv],
            inv_shares[:bess],
            inv_shares[:ashp],
            inv_shares[:dhw],
            inv_shares[:ev]
        ))
    end
    
    return summary
end

"""
    export_to_matlab(scenario_results, financial_results, filename)

Export results to MATLAB .mat format with n-dimensional arrays for easy comparison
across scenarios. Structure:
- P_pur[T, J, num_scenarios]: Grid purchase power
- P_sold[T, J, num_scenarios]: Grid sale power
- BESS_SoC[T, num_scenarios]: Battery state of charge
- EV_SoC[T, K, num_scenarios]: EV state of charge
- financial_summary: DataFrame with NPV, IRR per scenario
- price_sensitivity[num_scenarios]: Price sensitivity results
"""
function export_to_matlab(scenario_results, financial_results, filename)
    # Get dimensions from first scenario
    scenario_ids = sort(collect(keys(scenario_results)))
    num_scenarios = length(scenario_ids)

    first_results = scenario_results[scenario_ids[1]]
    T = size(first_results[:gen_pow], 1)
    J = size(first_results[:load_DC], 2)

    # Initialize n-dimensional arrays
    P_pur = zeros(T, J, num_scenarios)
    P_sold = zeros(T, J, num_scenarios)
    P_sc = zeros(T, J, num_scenarios)
    gen_pow = zeros(T, num_scenarios)
    load_DC = zeros(T, J, num_scenarios)
    BESS_SoC = zeros(T, num_scenarios)
    BESS_cha = zeros(T, J, num_scenarios)
    BESS_dis = zeros(T, J, num_scenarios)

    # Thermal arrays
    W_heating = zeros(T, J, num_scenarios)
    W_cooling = zeros(T, J, num_scenarios)
    load_Thermal = zeros(T, J, num_scenarios)

    # Financial summary arrays
    npv_values = zeros(num_scenarios)
    irr_values = zeros(num_scenarios)
    investment_total = zeros(num_scenarios)
    savings_total = zeros(num_scenarios)

    # Scenario parameters
    scenario_params = zeros(num_scenarios, 8)  # COEF, PV, GRID, BESS_price, BESS_cap, EVnum, ASHP, DHW

    # Fill arrays
    for (idx, scenario_id) in enumerate(scenario_ids)
        results = scenario_results[scenario_id]
        fin_results = financial_results[scenario_id]
        scenario = results[:scenario]

        # Power flows
        P_pur[:, :, idx] = results[:P_pur]
        P_sold[:, :, idx] = results[:P_sold]
        P_sc[:, :, idx] = results[:P_sc]
        gen_pow[:, idx] = results[:gen_pow]
        load_DC[:, :, idx] = results[:load_DC]

        # BESS
        BESS_SoC[:, idx] = results[:BESS_SoC]
        BESS_cha[:, :, idx] = results[:BESS_cha_pur] .+ results[:BESS_cha_sc]
        BESS_dis[:, :, idx] = results[:BESS_dis]

        # Thermal (only first J columns if input has more)
        W_heating[:, :, idx] = results[:W_pur_heating] .+ results[:W_sc_heating]
        W_cooling[:, :, idx] = results[:W_pur_cooling] .+ results[:W_sc_cooling]
        thermal_data = results[:load_Thermal]
        thermal_cols = min(size(thermal_data, 2), J)
        load_Thermal[:, 1:thermal_cols, idx] = thermal_data[:, 1:thermal_cols]

        # Financial
        npv_values[idx] = fin_results[:npv][1]
        irr_values[idx] = fin_results[:irr][1]
        investment_total[idx] = fin_results[:investments][:total]
        savings_total[idx] = fin_results[:savings][:total]

        # Scenario parameters
        scenario_params[idx, :] = [scenario.COEF, scenario.PV, scenario.GRID,
                                   scenario.BESS_price, scenario.BESS_cap,
                                   scenario.EVnum, scenario.ASHP, scenario.DHW]
    end

    # Write to .mat file
    matwrite(filename, Dict(
        "P_pur" => P_pur,
        "P_sold" => P_sold,
        "P_sc" => P_sc,
        "gen_pow" => gen_pow,
        "load_DC" => load_DC,
        "BESS_SoC" => BESS_SoC,
        "BESS_cha" => BESS_cha,
        "BESS_dis" => BESS_dis,
        "W_heating" => W_heating,
        "W_cooling" => W_cooling,
        "load_Thermal" => load_Thermal,
        "NPV" => npv_values,
        "IRR" => irr_values,
        "Investment_total" => investment_total,
        "Savings_total" => savings_total,
        "scenario_params" => scenario_params,
        "scenario_ids" => collect(scenario_ids),
        "T" => T,
        "J" => J,
        "num_scenarios" => num_scenarios
    ))

    return filename
end

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Professional color palette
const COLORS = (
    pv = RGB(1.0, 0.6, 0.0),        # Orange
    load = RGB(0.2, 0.4, 0.8),       # Blue
    grid_buy = RGB(0.8, 0.2, 0.2),   # Red
    grid_sell = RGB(0.2, 0.7, 0.3),  # Green
    bess = RGB(0.6, 0.3, 0.7),       # Purple
    ev = RGB(0.3, 0.7, 0.9),         # Cyan
    ashp = RGB(0.9, 0.4, 0.4),       # Coral
    dhw = RGB(0.5, 0.8, 0.5),        # Light green
)

"""
    _setup_plot_defaults()

Configure plot defaults for publication-quality figures.
"""
function _setup_plot_defaults()
    default(
        fontfamily = "Helvetica",
        titlefontsize = 14,
        guidefontsize = 12,
        tickfontsize = 10,
        legendfontsize = 10,
        dpi = 300,
        size = (1000, 600),
        grid = true,
        gridalpha = 0.3,
        framestyle = :box,
        margin = 5Plots.mm,
        bottom_margin = 8Plots.mm,
        left_margin = 5Plots.mm,
    )
end

"""
    generate_scenario_plots(results, financial_results, output_dir)

Generate visualization plots for a single scenario. Includes:
- Daily profile comparison (4 representative days)
- BESS state of charge over the year
- Monthly energy summary
- Thermal balance (if ASHP enabled)
- Energy indicators
- Hourly heatmap
- EV operation (if EVs enabled)
- Price curves
"""
function generate_scenario_plots(results, financial_results, output_dir)
    T = size(results[:gen_pow], 1)
    J = size(results[:load_DC], 2)
    scenario = results[:scenario]

    # Set plot defaults for publication quality
    _setup_plot_defaults()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_ranges = MONTHLY_HOUR_RANGES

    # 1. Daily profile comparison (4 representative days)
    sample_days = [15, 100, 200, 300]
    day_labels = ["Winter (Jan 15)", "Spring (Apr 10)", "Summer (Jul 19)", "Autumn (Oct 27)"]

    p1 = plot(layout=(2, 2), size=(1200, 800),
              plot_title="Daily Energy Profiles - Representative Days",
              plot_titlefontsize=16)

    for (i, day) in enumerate(sample_days)
        hours = (day-1)*24+1:day*24
        hour_labels = 0:23

        total_load = vec(sum(results[:load_DC][hours, :], dims=2))
        gen = results[:gen_pow][hours]
        purchase = vec(sum(results[:P_pur][hours, :], dims=2))
        sale = vec(sum(results[:P_sold][hours, :], dims=2))

        plot!(p1[i], hour_labels, gen,
              label="PV Generation", lw=2.5, color=COLORS.pv, fillalpha=0.3, fillrange=0)
        plot!(p1[i], hour_labels, total_load,
              label="Load", lw=2.5, color=COLORS.load)
        plot!(p1[i], hour_labels, purchase,
              label="Grid Purchase", lw=2, ls=:dash, color=COLORS.grid_buy)
        plot!(p1[i], hour_labels, sale,
              label="Grid Sale", lw=2, ls=:dash, color=COLORS.grid_sell)

        xlabel!(p1[i], "Hour of Day")
        ylabel!(p1[i], "Power (kW)")
        title!(p1[i], day_labels[i])
        xlims!(p1[i], (0, 23))
    end
    savefig(p1, joinpath(output_dir, "daily_profiles.png"))

    # 2. BESS State of Charge over the year
    if scenario.BESS_cap > 0
        p2 = plot(1:T, results[:BESS_SoC],
                  label="State of Charge",
                  xlabel="Hour of Year", ylabel="State of Charge (kWh)",
                  title="Battery Storage - Annual Profile",
                  lw=1.5, color=COLORS.bess,
                  fillalpha=0.2, fillrange=0,
                  size=(1200, 500),
                  legend=:topright)

        # Add capacity line
        hline!(p2, [scenario.BESS_cap],
               label="Max Capacity", color=:gray, ls=:dash, lw=1.5)
        hline!(p2, [scenario.BESS_cap * BESS_MIN_SOC],
               label="Min SoC", color=:red, ls=:dot, lw=1.5)

        # Add month labels
        month_starts = [1, 745, 1417, 2161, 2881, 3625, 4345, 5089, 5833, 6553, 7297, 8017]
        for (ms, mn) in zip(month_starts, month_names)
            vline!(p2, [ms], color=:gray, alpha=0.2, label="")
        end
        xticks!(p2, month_starts, month_names)

        savefig(p2, joinpath(output_dir, "bess_soc_annual.png"))
    end

    # 3. Monthly energy summary (grouped bar chart)
    monthly_gen = [sum(results[:gen_pow][r]) / 1000 for r in month_ranges]  # MWh
    monthly_load = [sum(results[:load_DC][r, :]) / 1000 for r in month_ranges]
    monthly_purchase = [sum(results[:P_pur][r, :]) / 1000 for r in month_ranges]
    monthly_sale = [sum(results[:P_sold][r, :]) / 1000 for r in month_ranges]

    p3 = groupedbar([monthly_gen monthly_load monthly_purchase monthly_sale],
                    bar_position=:dodge,
                    label=["PV Generation" "Total Load" "Grid Purchase" "Grid Sale"],
                    xlabel="Month", ylabel="Energy (MWh)",
                    title="Monthly Energy Balance",
                    xticks=(1:12, month_names),
                    color=[COLORS.pv COLORS.load COLORS.grid_buy COLORS.grid_sell],
                    legend=:topright,
                    bar_width=0.8,
                    size=(1200, 600))
    savefig(p3, joinpath(output_dir, "monthly_energy.png"))

    # 4. Thermal demand vs ASHP operation (if applicable)
    if scenario.ASHP > 0
        monthly_heating = [sum(max.(results[:load_Thermal][r, :], 0)) / 1000 for r in month_ranges]
        monthly_cooling = [sum(abs.(min.(results[:load_Thermal][r, :], 0))) / 1000 for r in month_ranges]
        monthly_heat_op = [sum(results[:W_pur_heating][r, :] .+ results[:W_sc_heating][r, :]) / 1000 for r in month_ranges]
        monthly_cool_op = [sum(results[:W_pur_cooling][r, :] .+ results[:W_sc_cooling][r, :]) / 1000 for r in month_ranges]

        p4 = groupedbar([monthly_heating monthly_cooling monthly_heat_op monthly_cool_op],
                        bar_position=:dodge,
                        label=["Heating Demand" "Cooling Demand" "ASHP Heating" "ASHP Cooling"],
                        xlabel="Month", ylabel="Energy (MWh)",
                        title="Thermal Demand vs ASHP Operation",
                        xticks=(1:12, month_names),
                        color=[RGB(0.9,0.3,0.2) RGB(0.2,0.5,0.9) RGB(1.0,0.6,0.3) RGB(0.3,0.8,0.9)],
                        legend=:topright,
                        size=(1200, 600))
        savefig(p4, joinpath(output_dir, "thermal_balance.png"))
    end

    # 5. Energy independence indicators (pie-style bar)
    total_gen = sum(results[:gen_pow])
    total_load = sum(results[:load_DC])
    total_sc = sum(results[:P_sc])
    total_purchase = sum(results[:P_pur])

    self_consumption = total_gen > 0 ? total_sc / total_gen * 100 : 0
    self_sufficiency = total_load > 0 ? (total_load - total_purchase) / total_load * 100 : 0

    p5 = bar(["Self-Consumption\n(% of PV used locally)", "Self-Sufficiency\n(% of load from local)"],
             [self_consumption, self_sufficiency],
             ylabel="Percentage (%)",
             title="Energy Independence Indicators",
             color=[COLORS.pv COLORS.load],
             legend=false,
             ylims=(0, 100),
             bar_width=0.6,
             size=(800, 500),
             annotations=[(1, self_consumption + 3, text("$(round(self_consumption, digits=1))%", 12)),
                         (2, self_sufficiency + 3, text("$(round(self_sufficiency, digits=1))%", 12))])
    savefig(p5, joinpath(output_dir, "energy_indicators.png"))

    # 6. NEW: Hourly energy heatmap (24h × 365d)
    daily_net = zeros(24, 365)
    for d in 1:365
        for h in 1:24
            t = (d-1)*24 + h
            if t <= T
                daily_net[h, d] = results[:gen_pow][t] - sum(results[:load_DC][t, :])
            end
        end
    end

    p6 = heatmap(1:365, 0:23, daily_net,
                 xlabel="Day of Year", ylabel="Hour of Day",
                 title="Net Energy Balance Heatmap (Generation - Load)",
                 color=:RdYlGn,
                 clims=(-maximum(abs.(daily_net)), maximum(abs.(daily_net))),
                 size=(1400, 500),
                 colorbar_title="kW")
    # Add month markers
    month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    vline!(p6, month_days, color=:white, alpha=0.5, label="", lw=0.5)
    savefig(p6, joinpath(output_dir, "energy_heatmap.png"))

    # 7. NEW: EV State of Charge (if EVs enabled)
    if scenario.EVnum > 0 && haskey(results, :EV_SoC)
        # Weekly sample (first week of each season)
        weeks = [(1, 168), (2161, 2328), (4345, 4512), (6553, 6720)]
        week_names = ["Winter (Week 1)", "Spring (Week 13)", "Summer (Week 26)", "Autumn (Week 39)"]

        p7 = plot(layout=(2, 2), size=(1200, 800),
                  plot_title="EV Fleet State of Charge - Seasonal Samples",
                  plot_titlefontsize=16)

        for (i, (start_h, end_h)) in enumerate(weeks)
            hours = start_h:min(end_h, T)
            ev_soc_total = vec(sum(results[:EV_SoC][hours, :], dims=2))

            plot!(p7[i], 1:length(hours), ev_soc_total,
                  label="Total EV SoC", lw=2, color=COLORS.ev,
                  fillalpha=0.3, fillrange=0,
                  xlabel="Hour of Week", ylabel="SoC (kWh)",
                  title=week_names[i])
        end
        savefig(p7, joinpath(output_dir, "ev_soc_weekly.png"))
    end

    # 8. NEW: Electricity price profile
    if haskey(results, :PI_pur)
        # Daily average prices
        daily_buy = [mean(results[:PI_pur][(d-1)*24+1:d*24]) for d in 1:365]
        daily_sell = [mean(results[:PI_sell][(d-1)*24+1:d*24]) for d in 1:365]

        p8 = plot(1:365, daily_buy * 1000,  # €/kWh to €/MWh
                  label="Purchase Price", lw=1.5, color=COLORS.grid_buy,
                  xlabel="Day of Year", ylabel="Price (€/MWh)",
                  title="Daily Average Electricity Prices",
                  size=(1200, 500),
                  legend=:topright)
        plot!(p8, 1:365, daily_sell * 1000,
              label="Sell Price", lw=1.5, color=COLORS.grid_sell)

        # Add month markers
        vline!(p8, month_days, color=:gray, alpha=0.3, label="", lw=0.5)
        xticks!(p8, month_days, month_names)

        savefig(p8, joinpath(output_dir, "price_profile.png"))
    end

    return nothing
end

"""
    generate_summary_plots(scenario_results, financial_results, output_dir)

Generate comparison plots across all scenarios. Includes:
- NPV comparison bar chart
- IRR comparison
- Investment breakdown
- Cost reduction comparison
- NPV vs IRR scatter
"""
function generate_summary_plots(scenario_results, financial_results, output_dir)
    scenario_ids = sort(collect(keys(scenario_results)))
    num_scenarios = length(scenario_ids)

    _setup_plot_defaults()

    # Extract data
    scenario_names = String[]
    short_names = String[]
    npv_values = Float64[]
    irr_values = Float64[]
    inv_pv = Float64[]
    inv_bess = Float64[]
    inv_ev = Float64[]
    inv_ashp = Float64[]
    cost_reduction = Float64[]

    for scenario_id in scenario_ids
        results = scenario_results[scenario_id]
        fin_results = financial_results[scenario_id]
        scenario = results[:scenario]

        push!(scenario_names, scenario.name)
        # Create shorter names for display
        short = "S$(scenario_id)"
        push!(short_names, short)

        push!(npv_values, fin_results[:npv][1] / 1000)  # k€
        push!(irr_values, fin_results[:irr][1] * 100)

        inv_shares = fin_results[:investment_shares]
        total_inv = fin_results[:investments][:total] / 1000  # k€
        push!(inv_pv, inv_shares[:pv] * total_inv)
        push!(inv_bess, inv_shares[:bess] * total_inv)
        push!(inv_ev, inv_shares[:ev] * total_inv)
        push!(inv_ashp, inv_shares[:ashp] * total_inv)

        initial = fin_results[:initial_costs][:total]
        final = sum(results[:Cost_REC_j])
        push!(cost_reduction, initial > 0 ? (initial - final) / initial * 100 : 0)
    end

    # Color by positive/negative NPV
    npv_colors = [v >= 0 ? COLORS.grid_sell : COLORS.grid_buy for v in npv_values]

    # 1. NPV Comparison
    p1 = bar(short_names, npv_values,
             xlabel="Scenario", ylabel="NPV (k€)",
             title="Net Present Value by Scenario",
             color=npv_colors, legend=false,
             xrotation=45,
             size=(max(800, num_scenarios * 50), 600))
    hline!(p1, [0], color=:black, lw=1, label="")
    savefig(p1, joinpath(output_dir, "npv_comparison.png"))

    # 2. IRR Comparison
    irr_colors = [v >= 5 ? COLORS.grid_sell : (v >= 0 ? COLORS.pv : COLORS.grid_buy) for v in irr_values]
    p2 = bar(short_names, irr_values,
             xlabel="Scenario", ylabel="IRR (%)",
             title="Internal Rate of Return by Scenario",
             color=irr_colors, legend=false,
             xrotation=45,
             size=(max(800, num_scenarios * 50), 600))
    hline!(p2, [5], color=:gray, ls=:dash, lw=1.5, label="")  # 5% reference
    savefig(p2, joinpath(output_dir, "irr_comparison.png"))

    # 3. Investment Breakdown (stacked bar)
    inv_matrix = hcat(inv_pv, inv_bess, inv_ev, inv_ashp)
    p3 = groupedbar(inv_matrix,
                    bar_position=:stack,
                    label=["PV+Inverter" "BESS" "EV Chargers" "ASHP+DHW"],
                    xlabel="Scenario", ylabel="Investment (k€)",
                    title="Investment Breakdown by Scenario",
                    xticks=(1:num_scenarios, short_names),
                    color=[COLORS.pv COLORS.bess COLORS.ev COLORS.ashp],
                    xrotation=45,
                    legend=:outertopright,
                    size=(max(900, num_scenarios * 50), 600))
    savefig(p3, joinpath(output_dir, "investment_breakdown.png"))

    # 4. Cost Reduction Comparison
    cr_colors = [v >= 0 ? COLORS.grid_sell : COLORS.grid_buy for v in cost_reduction]
    p4 = bar(short_names, cost_reduction,
             xlabel="Scenario", ylabel="Cost Reduction (%)",
             title="Annual Cost Reduction vs Baseline",
             color=cr_colors, legend=false,
             xrotation=45,
             size=(max(800, num_scenarios * 50), 600))
    hline!(p4, [0], color=:black, lw=1, label="")
    savefig(p4, joinpath(output_dir, "cost_reduction.png"))

    # 5. NPV vs IRR scatter plot
    p5 = scatter(irr_values, npv_values,
                 xlabel="IRR (%)", ylabel="NPV (k€)",
                 title="Financial Performance: NPV vs IRR",
                 markersize=10, color=COLORS.load,
                 markerstrokewidth=2,
                 legend=false,
                 size=(900, 700))
    # Add quadrant lines
    vline!(p5, [5], color=:gray, ls=:dash, alpha=0.5, label="")
    hline!(p5, [0], color=:gray, ls=:dash, alpha=0.5, label="")
    # Add labels
    for (i, (x, y)) in enumerate(zip(irr_values, npv_values))
        annotate!(p5, x, y + maximum(abs.(npv_values)) * 0.03,
                  text(short_names[i], 9, :center))
    end
    savefig(p5, joinpath(output_dir, "npv_vs_irr.png"))

    return nothing
end

"""
    generate_html_report(scenario_results, financial_results, output_dir)

Generate an HTML report containing all plots and summary statistics.
"""
function generate_html_report(scenario_results, financial_results, output_dir)
    scenario_ids = sort(collect(keys(scenario_results)))

    # Build HTML content
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FLEXECOM Results Report</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }
            h2 { color: #34495e; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 15px; }
            h3 { color: #7f8c8d; }
            .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
            .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
            .stat-card.positive { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
            .stat-card.negative { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
            .stat-value { font-size: 2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; }
            img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
            .plot-section { margin: 30px 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #3498db; color: white; }
            tr:hover { background: #f5f5f5; }
            .scenario-section { background: #fafafa; padding: 20px; margin: 20px 0; border-radius: 10px; }
            .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔋 FLEXECOM Optimization Results</h1>
            <p><strong>Generated:</strong> $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))</p>
            <p><strong>Scenarios analyzed:</strong> $(length(scenario_ids))</p>

            <h2>📊 Executive Summary</h2>
    """

    # Calculate summary statistics
    all_npv = [financial_results[id][:npv][1] for id in scenario_ids]
    all_irr = [financial_results[id][:irr][1] * 100 for id in scenario_ids]
    total_investment = sum(financial_results[id][:investments][:total] for id in scenario_ids)

    best_npv_idx = argmax(all_npv)
    best_irr_idx = argmax(all_irr)

    html *= """
            <div class="summary-grid">
                <div class="stat-card $(mean(all_npv) >= 0 ? "positive" : "negative")">
                    <div class="stat-value">$(round(mean(all_npv)/1000, digits=1)) k€</div>
                    <div class="stat-label">Average NPV</div>
                </div>
                <div class="stat-card $(mean(all_irr) >= 5 ? "positive" : "")">
                    <div class="stat-value">$(round(mean(all_irr), digits=1))%</div>
                    <div class="stat-label">Average IRR</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">$(round(total_investment/1000, digits=0)) k€</div>
                    <div class="stat-label">Total Investment</div>
                </div>
                <div class="stat-card positive">
                    <div class="stat-value">S$(scenario_ids[best_npv_idx])</div>
                    <div class="stat-label">Best NPV Scenario</div>
                </div>
            </div>

            <h2>📈 Comparative Analysis</h2>
            <div class="plot-section">
                <h3>Financial Performance</h3>
                <img src="summary_plots/npv_comparison.png" alt="NPV Comparison">
                <img src="summary_plots/irr_comparison.png" alt="IRR Comparison">
                <img src="summary_plots/npv_vs_irr.png" alt="NPV vs IRR">
            </div>

            <div class="plot-section">
                <h3>Investment Analysis</h3>
                <img src="summary_plots/investment_breakdown.png" alt="Investment Breakdown">
                <img src="summary_plots/cost_reduction.png" alt="Cost Reduction">
            </div>

            <h2>📋 Scenario Details</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>COEF</th>
                    <th>PV (kWp)</th>
                    <th>BESS (kWh)</th>
                    <th>EVs</th>
                    <th>ASHP (kW)</th>
                    <th>NPV (€)</th>
                    <th>IRR (%)</th>
                </tr>
    """

    for id in scenario_ids
        s = scenario_results[id][:scenario]
        f = financial_results[id]
        html *= """
                <tr>
                    <td>S$(id)</td>
                    <td>$(s.COEF)</td>
                    <td>$(s.PV)</td>
                    <td>$(s.BESS_cap)</td>
                    <td>$(s.EVnum)</td>
                    <td>$(s.ASHP)</td>
                    <td>$(round(Int, f[:npv][1]))</td>
                    <td>$(round(f[:irr][1] * 100, digits=1))</td>
                </tr>
        """
    end

    html *= """
            </table>

            <h2>🔍 Individual Scenario Plots</h2>
    """

    for id in scenario_ids
        s = scenario_results[id][:scenario]
        html *= """
            <div class="scenario-section">
                <h3>Scenario $(id): $(s.name)</h3>
                <img src="scenario_$(id)/plots/daily_profiles.png" alt="Daily Profiles">
                <img src="scenario_$(id)/plots/monthly_energy.png" alt="Monthly Energy">
                <img src="scenario_$(id)/plots/energy_heatmap.png" alt="Energy Heatmap">
                <img src="scenario_$(id)/plots/energy_indicators.png" alt="Energy Indicators">
        """
        if s.BESS_cap > 0
            html *= """<img src="scenario_$(id)/plots/bess_soc_annual.png" alt="BESS SoC">"""
        end
        if s.ASHP > 0
            html *= """<img src="scenario_$(id)/plots/thermal_balance.png" alt="Thermal Balance">"""
        end
        html *= """
            </div>
        """
    end

    html *= """
            <div class="footer">
                <p>Generated by FLEXECOM v1.0.0 - Flexible Energy Community Optimisation Model</p>
                <p>Reference: Á. Manso Burgos et al., "Flexible energy community optimisation for the Valencian Community," Applied Energy (2025)</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Write HTML file
    report_path = joinpath(output_dir, "report.html")
    open(report_path, "w") do f
        write(f, html)
    end

    return report_path
end

end # module