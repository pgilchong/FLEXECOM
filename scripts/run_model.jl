#!/usr/bin/env julia

# Add the current directory to the load path
push!(LOAD_PATH, "@.")

using Pkg
Pkg.activate(".")

using FLEXECOM
using Gurobi

"""
Run the FLEXECOM model with specified parameters.

Usage:
    julia run_model.jl [SCENARIO_FILE] [INPUT_DIR] [OUTPUT_DIR] [OPTIONS]

Arguments:
    SCENARIO_FILE    Path to the scenario CSV file (default: "data/scenarios/scenarios_sample.csv")
    INPUT_DIR        Path to the input data directory (default: "data/inputs")
    OUTPUT_DIR       Path to the output directory (default: "data/outputs")

Options:
    --verbose=N      Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1)
    --no-excel       Disable Excel export (enabled by default)
    --no-matlab      Disable MATLAB export (enabled by default)
    --plots          Enable plot generation (disabled by default)

Examples:
    julia --project scripts/run_model.jl
    julia --project scripts/run_model.jl data/scenarios/my_scenarios.csv data/inputs data/outputs --verbose=2
    julia --project scripts/run_model.jl data/scenarios/my_scenarios.csv data/inputs data/outputs --plots --no-excel
"""

function parse_args(args)
    # Default values
    scenario_file = "data/scenarios/scenarios_sample.csv"
    input_dir = "data/inputs"
    output_dir = "data/outputs"
    verbose_level = 1
    export_excel = true
    export_matlab = true
    generate_plots = false

    positional_idx = 0

    for arg in args
        if startswith(arg, "--")
            # Parse option
            if startswith(arg, "--verbose=")
                verbose_level = parse(Int, split(arg, "=")[2])
            elseif arg == "--no-excel"
                export_excel = false
            elseif arg == "--no-matlab"
                export_matlab = false
            elseif arg == "--plots"
                generate_plots = true
            elseif arg == "--help" || arg == "-h"
                println(@doc main)
                exit(0)
            else
                println("Unknown option: $arg")
                println("Use --help for usage information")
                exit(1)
            end
        else
            # Positional argument
            positional_idx += 1
            if positional_idx == 1
                scenario_file = arg
            elseif positional_idx == 2
                input_dir = arg
            elseif positional_idx == 3
                output_dir = arg
            end
        end
    end

    return (
        scenario_file=scenario_file,
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=verbose_level,
        export_excel=export_excel,
        export_matlab=export_matlab,
        generate_plots=generate_plots
    )
end

function main()
    # Parse command line arguments
    config = parse_args(ARGS)

    if config.verbose < 0 || config.verbose > 2
        error("Verbose level must be 0, 1, or 2")
    end

    log(level, message) = config.verbose >= level && println(message)

    # Check if files and directories exist
    if !isfile(config.scenario_file)
        error("Scenario file not found: $(config.scenario_file)")
    end

    if !isdir(config.input_dir)
        error("Input directory not found: $(config.input_dir)")
    end

    if !isdir(config.output_dir)
        log(1, "Creating output directory: $(config.output_dir)")
        mkpath(config.output_dir)
    end

    # Create Gurobi environment
    gurobi_env = Gurobi.Env()

    log(1, "Running FLEXECOM model with the following parameters:")
    log(1, "  Scenario file: $(config.scenario_file)")
    log(1, "  Input directory: $(config.input_dir)")
    log(1, "  Output directory: $(config.output_dir)")
    log(1, "  Verbose level: $(config.verbose)")
    log(1, "  Export Excel: $(config.export_excel)")
    log(1, "  Export MATLAB: $(config.export_matlab)")
    log(1, "  Generate plots: $(config.generate_plots)")

    # Run the model
    start_time = time()
    results, financial_results, summary = FLEXECOM.run_full_model(
        config.scenario_file,
        config.input_dir,
        config.output_dir;
        gurobi_env=gurobi_env,
        verbose=config.verbose,
        export_excel=config.export_excel,
        export_matlab=config.export_matlab,
        generate_plots=config.generate_plots,
    )
    elapsed_time = time() - start_time

    log(1, "Model run completed in $(round(elapsed_time, digits=2)) seconds.")
    log(1, "Results saved to: $(config.output_dir)")

    # Print summary statistics
    log(1, "\nSummary of results:")
    log(1, "  Number of scenarios: $(length(results))")

    # Get the scenario with the best financial results (highest NPV)
    best_scenario = argmax(summary.NPV)
    log(1, "  Best scenario (highest NPV): $(summary.Scenario_ID[best_scenario])")
    log(1, "    - Scenario Name: $(summary.Scenario_Name[best_scenario])")
    log(1, "    - COEF Type: $(summary.COEF_Type[best_scenario])")
    log(1, "    - PV Power: $(summary.PV_Power[best_scenario]) kWp/household")
    log(1, "    - BESS Capacity: $(summary.BESS_Capacity[best_scenario]) kWh")
    log(1, "    - EV Number: $(summary.EV_Number[best_scenario])")
    log(1, "    - ASHP Power: $(summary.ASHP_Power[best_scenario]) kW")
    log(1, "    - DHW: $(summary.DHW[best_scenario])")
    log(1, "    - Cost Reduction: $(round(summary.Cost_Reduction_Pct[best_scenario], digits=2))%")
    log(1, "    - NPV: $(round(summary.NPV[best_scenario], digits=2)) €")
    log(1, "    - IRR: $(round(summary.IRR[best_scenario] * 100, digits=2))%")
    log(1, "    - Investment Shares:")
    log(1, "      * PV: $(round(summary.PV_Investment_Share[best_scenario] * 100, digits=2))%")
    log(1, "      * BESS: $(round(summary.BESS_Investment_Share[best_scenario] * 100, digits=2))%")
    log(1, "      * ASHP: $(round(summary.ASHP_Investment_Share[best_scenario] * 100, digits=2))%")
    log(1, "      * DHW: $(round(summary.DHW_Investment_Share[best_scenario] * 100, digits=2))%")
    log(1, "      * EV: $(round(summary.EV_Investment_Share[best_scenario] * 100, digits=2))%")

    return 0
end

# Run the main function
exit(main())
