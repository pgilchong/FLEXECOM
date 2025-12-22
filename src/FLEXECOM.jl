module FLEXECOM

using JuMP
using Gurobi
using DataFrames
using CSV
using XLSX
using Dates
using JLD2
using LinearAlgebra
using Statistics
using MAT
using Plots
using StatsPlots

include("types.jl")
using .Types

include("constants.jl")
using .Constants

include("data_loader.jl")
using .DataLoader

include("optimizer.jl")
using .Optimizer

include("financial.jl")
using .Financial

include("utils.jl")
using .Utils

# Reexport key types and functions
export
    # Main function
    run_full_model,
    
    # From types.jl
    Scenario, BESS_specs, EV_specs, ASHP_specs, Financial_params,
    
    # From data_loader.jl
    load_scenarios, load_input_data,
    
    # From optimizer.jl
    optimize_scenario, calculate_energy_balance,
    
    # From financial.jl
    calculate_financial_metrics, calculate_npv, calculate_irr, calculate_initial_costs,
    price_sensitivity_values,
    
    # From utils.jl
    save_results, load_results, create_summary, verify_energy_balance, export_to_excel,
    export_to_matlab, generate_plots, generate_html_report

function run_full_model(scenario_file, input_dir, output_dir;
                        gurobi_env=nothing,
                        verbose=1,
                        export_excel::Bool=true,
                        export_matlab::Bool=true,
                        generate_plots::Bool=false)
    0 <= verbose <= 2 || throw(ArgumentError("verbose must be 0, 1, or 2"))

    log(level, message) = verbose >= level && println(message)

    log(2, "Preparing optimisation environment…")

    # Create Gurobi environment if not provided
    if isnothing(gurobi_env)
        env = Gurobi.Env()
        log(2, "Created new Gurobi environment")
    else
        env = gurobi_env
        log(2, "Using provided Gurobi environment")
    end

    # Load scenarios and input data
    log(1, "Loading scenarios from $(scenario_file)")
    scenarios = DataLoader.load_scenarios(scenario_file)
    log(1, "Loading input data from $(input_dir)")
    input_data = DataLoader.load_input_data(input_dir)
    log(1, "Loaded $(length(scenarios)) scenario(s)")

    # Create output directory with timestamp
    timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    result_dir = joinpath(output_dir, timestamp)
    mkpath(result_dir)
    log(1, "Results will be stored in $(result_dir)")

    # Run optimization for each scenario
    results = Dict()

    for scenario in scenarios
        log(1, "Running scenario $(scenario.id): $(scenario.name)")
        scenario_results = Optimizer.optimize_scenario(scenario, input_data, env)

        # Add scenario info to results
        scenario_results[:scenario] = scenario
        results[scenario.id] = scenario_results

        # Check energy balance
        energy_balance = Optimizer.calculate_energy_balance(scenario_results)
        log(2, "Energy balance residual for scenario $(scenario.id): $(sum(energy_balance))")
    end

    # Calculate financial metrics
    financial_params = FINANCIAL_DEFAULT

    log(1, "Calculating financial metrics")
    financial_results = Financial.calculate_financial_metrics(results, financial_params, input_data)

    # Save results
    log(1, "Saving detailed results")
    Utils.save_results(results, financial_results, result_dir;
                       export_excel=export_excel,
                       export_matlab=export_matlab,
                       generate_plots=generate_plots)

    # Create summary
    summary = Utils.create_summary(results, financial_results)
    CSV.write(joinpath(result_dir, "summary.csv"), summary)

    log(1, "Run completed successfully")

    return results, financial_results, summary
end

end # module
