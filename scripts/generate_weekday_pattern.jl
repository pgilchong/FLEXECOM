#!/usr/bin/env julia

# Add the current directory to the load path
push!(LOAD_PATH, "@.")

using Pkg
Pkg.activate(".")

using XLSX, DataFrames

"""
Generate the weekday/season classifier required for variable coefficient allocation.

The script writes `weekday<YY>.xlsx` (e.g. `weekday21.xlsx`) with 8 profiles combining
four seasons and weekday/weekend behaviour. Leap years are handled automatically.
Update the `weekday_pattern` section in `data_catalog.toml` if you change the output
file or simulate a leap year horizon (8,784 hours).

Usage:
    julia scripts/generate_weekday_pattern.jl [OUTPUT_DIR] [YEAR]

Arguments:
    OUTPUT_DIR    Path to output directory (default: "data/inputs")
    YEAR          Year to generate patterns for (default: 2021)
"""

function get_first_day_of_year(year)
    # Get the day of week for January 1st of the given year
    # 1=Monday, 2=Tuesday, ..., 7=Sunday
    first_day = Dates.dayofweek(Date(year, 1, 1))
    return first_day
end

function main()
    # Parse command line arguments
    args = ARGS
    
    # Set defaults
    output_dir = length(args) >= 1 ? args[1] : "data/inputs"
    year = length(args) >= 2 ? parse(Int, args[2]) : 2021
    
    println("Generating weekday patterns for year $year...")
    
    # Parameters
    d0 = get_first_day_of_year(year)  # First day of the year
    T = Dates.isleapyear(year) ? 8784 : 8760  # Hours in the year
    
    println("First day of $year is: $(Dates.dayname(Date(year, 1, 1))) (d0=$d0)")
    println("Total hours in $year: $T")

    # Initialize weekday array
    weekday = zeros(Int, T, 3)

    # Determine initial days
    if d0 <= 5
        endini = (6 - d0) * 24
        weekday[1:endini, 1] .= 1
        endweek = endini + 2 * 24
        weekday[endini+1:endweek, 1] .= 2
    elseif d0 <= 7
        endweek = (8 - d0) * 24
        weekday[1:endweek, 1] .= 2
    end

    # Fill the rest of the year
    while endweek < T
        monday = min(T, endweek + 1)
        friday = min(T, endweek + 24 * 5)
        endweek = min(T, endweek + 24 * 7)
        
        if endweek <= 24 * 79  # Winter until March 20th
            weekday[monday:friday, 1] .= 1
            weekday[(friday + 1):endweek, 1] .= 2
        elseif endweek <= 24 * 172  # Spring until June 21st
            weekday[monday:friday, 1] .= 3
            weekday[(friday + 1):endweek, 1] .= 4
        elseif endweek <= 24 * 266  # Summer until September 23rd
            weekday[monday:friday, 1] .= 5
            weekday[(friday + 1):endweek, 1] .= 6
        elseif endweek <= 24 * 355  # Fall until December 21st
            weekday[monday:friday, 1] .= 7
            weekday[(friday + 1):endweek, 1] .= 8
        else  # Winter until end of year
            weekday[monday:friday, 1] .= 1
            weekday[(friday + 1):endweek, 1] .= 2
        end
    end

    # Fill hour of day (column 2)
    count = 1
    while count <= T
        for c = 1:24
            if count + c - 1 <= T
                weekday[count + c - 1, 2] = c
            end
        end
        count += 24
    end

    # Fill unique coefficient set (column 3)
    weekday[:, 3] = weekday[:, 2] .+ 24 .* (weekday[:, 1] .- 1)

    # Verify
    coef_sets = maximum(weekday[:, 1]) * maximum(weekday[:, 2])
    println("Total coefficient sets: $coef_sets")

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Write to Excel file
    filename = joinpath(output_dir, "weekday$(year % 100).xlsx")
    XLSX.openxlsx(filename, mode="w") do xf
        sheet = xf[1]
        XLSX.rename!(sheet, "weekday")
        sheet["A1:C$T"] = weekday[1:T, :]
    end

    println("Weekday pattern file generated: $filename")
    println("Profile mapping:")
    println("  1-2: Winter (weekday/weekend)")
    println("  3-4: Spring (weekday/weekend)")
    println("  5-6: Summer (weekday/weekend)")
    println("  7-8: Fall (weekday/weekend)")
    
    return 0
end

# Run the main function
exit(main())