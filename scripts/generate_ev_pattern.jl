#!/usr/bin/env julia

# Add the current directory to the load path
push!(LOAD_PATH, "@.")

using Pkg
Pkg.activate(".")

using XLSX, DataFrames, Random

"""
Generate an EV availability workbook compatible with FLEXECOM.

The script reproduces the stochastic method used in the publication and writes the
result to `EV_pattern_<NUM_EVS>.xlsx`. Update the `ev_patterns` section in
`data_catalog.toml` so that the optimiser loads the correct file and sheet ranges.

Usage:
    julia scripts/generate_ev_pattern.jl [OUTPUT_DIR] [NUM_EVS]

Arguments:
    OUTPUT_DIR    Path to output directory (default: "data/inputs")
    NUM_EVS       Number of EVs to generate patterns for (default: 50)
"""

function main()
    # Parse command line arguments
    args = ARGS
    
    # Set defaults
    output_dir = length(args) >= 1 ? args[1] : "data/inputs"
    K = length(args) >= 2 ? parse(Int, args[2]) : 50
    
    println("Generating EV patterns for $K vehicles...")
    
    # Parameters
    T = 8760  # Hours in a year
    EV_sal_obj = 8  # Target departure hour
    EV_SoC_max = 0.8  # Maximum SoC target

    # EV specifications
    EV_cap = 50.0  # kWh
    EV_cons_km = 0.159  # kWh/km
    EV_Eff = 0.90
    EV_Pch = 7.4  # kW

    # Initialize arrays
    EV_km = zeros(T, K)
    EV_SoC_obj = zeros(K)
    EV_o = ones(T, K)
    EV_SoC_min = zeros(T, K)
    EV_lleg = zeros(T, K)
    EV_lleg = Int.(EV_lleg)
    EV_h_sal = ones(T, K)
    EV_h_sal = Int.(EV_h_sal)

    # Probability distribution parameters
    # Arrival
    mi_l = 17.6
    sigma_l = 2

    # Departure
    mi_s = 8.92
    sigma_s = 2

    # Calculate probability mass functions
    PMF_EV = zeros(24, 2)  # arrival and departure probabilities
    h = 1
    while h < 25
        if h <= mi_l - 12  # Arrival probability
            PMF_EV[h, 1] = ((2π)^0.5 * sigma_l)^-1 * exp(-((h + 24 - mi_l)^2 / (2 * sigma_l^2)))
        else
            PMF_EV[h, 1] = ((2π)^0.5 * sigma_l)^-1 * exp(-((h - mi_l)^2 / (2 * sigma_l^2)))
        end
        
        if h <= mi_s + 12  # Departure probability
            PMF_EV[h, 2] = ((2π)^0.5 * sigma_s)^-1 * exp(-((h - mi_s)^2 / (2 * sigma_s^2)))
        else
            PMF_EV[h, 2] = ((2π)^0.5 * sigma_s)^-1 * exp(-((h - 24 - mi_s)^2 / (2 * sigma_s^2)))
        end
        h = h + 1
    end

    # Convert to cumulative distribution
    h = 2
    while h < 25
        PMF_EV[h, :] = PMF_EV[h, :] + PMF_EV[h-1, :]
        h = h + 1
    end

    # Generate patterns for each EV
    dias = div(T, 24)
    Random.seed!(1234)  # For reproducibility

    alea_s = rand(dias, K)  # Random values for departure
    alea_l = rand(dias, K)  # Random values for arrival

    for v in 1:K  # v is equivalent to k
        dia = 0
        sal_obj = EV_sal_obj
        EV_SoC_obj[v] = EV_SoC_max * EV_cap
        EV_SoC_min[1, v] = max(EV_SoC_obj[v] - (sal_obj - 1) * EV_Pch, 0)
        h_sal = 0
        
        for te in 2:T
            EV_o[te, v] = EV_o[te - 1, v]
            
            if PMF_EV[te - 24 * dia, 2] > alea_s[dia + 1, v]  # Does it leave?
                EV_o[te, v] = max(EV_o[te - 1, v] - 1, 0)
            end
            
            if PMF_EV[te - 24 * dia, 1] > alea_l[dia + 1, v]  # Does it park?
                EV_o[te, v] = min(EV_o[te - 1, v] + 1, 1)
            end
            
            if te <= sal_obj && EV_o[te, v] == 1
                EV_SoC_min[te, v] = max(EV_SoC_obj[v] - (sal_obj - te) * EV_Pch, 0)
                h_sal = te  # departure hour
            end
            
            if EV_o[te - 1, v] < EV_o[te, v]  # If EV arrives, calculate consumption
                EV_h_sal[te, v] = h_sal  # save departure hour at arrival cell
                EV_lleg[te, v] = 1
                
                # Distance probability (Weibull distribution)
                PMF_KM = 0
                alfa = 3
                lambda = 28.9  # Average distance according to MITECO
                lambda = 1 / lambda
                km = 1
                alea_km = rand()
                
                while PMF_KM < alea_km
                    PMF_KM = 1 - exp(-(lambda * km)^alfa)
                    km += 1
                end
                
                EV_km[te, v] = km
            end
            
            if rem(te, 24) == 0
                dia += 1
                sal_obj += 24
            end
        end
    end

    # Create DataFrames
    R_EV_o = DataFrame(EV_o, :auto)
    R_EV_km = DataFrame(EV_km, :auto)
    R_EV_lleg = DataFrame(EV_lleg, :auto)
    R_EV_h_sal = DataFrame(EV_h_sal, :auto)
    R_EV_SoC_min = DataFrame(EV_SoC_min, :auto)

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Write to Excel file
    nombre = joinpath(output_dir, "EV_pattern_$(K).xlsx")
    XLSX.openxlsx(nombre, mode="w") do xf
        # EV_o sheet
        sheet = XLSX.addsheet!(xf, "EV_o")
        XLSX.writetable!(sheet, R_EV_o, anchor_cell=XLSX.CellRef("A1"))
        
        # EV_km sheet
        sheet = XLSX.addsheet!(xf, "EV_km")
        XLSX.writetable!(sheet, R_EV_km, anchor_cell=XLSX.CellRef("A1"))
        
        # EV_lleg sheet
        sheet = XLSX.addsheet!(xf, "EV_lleg")
        XLSX.writetable!(sheet, R_EV_lleg, anchor_cell=XLSX.CellRef("A1"))
        
        # EV_h_sal sheet
        sheet = XLSX.addsheet!(xf, "EV_h_sal")
        XLSX.writetable!(sheet, R_EV_h_sal, anchor_cell=XLSX.CellRef("A1"))
        
        # EV_SoC_min sheet
        sheet = XLSX.addsheet!(xf, "EV_SoC_min")
        XLSX.writetable!(sheet, R_EV_SoC_min, anchor_cell=XLSX.CellRef("A1"))
    end

    println("EV pattern file generated: $nombre")
    println("Total EVs: $K")
    println("Total hours: $T")
    
    return 0
end

# Run the main function
exit(main())