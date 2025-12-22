"""
Diagnostic test for baseline price detection in price sensitivity analysis.

This test verifies that the baseline price combination (mean values) is correctly
detected and that results are consistent.

Run with:
    julia --project test/test_baseline_detection.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using FLEXECOM

println("=" ^ 60)
println("BASELINE PRICE DETECTION DIAGNOSTIC")
println("=" ^ 60)
println()

# Test 1: Verify mean values are exactly in the price grid
println("Test 1: Checking if mean values are exactly in price grids")
println("-" ^ 60)

price_values = FLEXECOM.Financial.price_sensitivity_values()
baseline = FLEXECOM.Constants.PRICE_SENSITIVITY_MEANS

all_exact = true

for key in [:PV_price, :Fuel_price, :NG_price, :EV_price]
    grid = getfield(price_values, key)
    mean_val = getfield(baseline, key)

    # Find closest value in grid
    idx = argmin(abs.(grid .- mean_val))
    closest = grid[idx]
    diff = abs(closest - mean_val)

    is_exact = closest == mean_val
    is_in_grid = mean_val in grid

    println("$key:")
    println("  Mean expected:      $mean_val")
    println("  Closest in grid:    $closest")
    println("  Difference:         $diff")
    println("  Exactly equal?:     $is_exact")
    println("  In grid (∈)?:       $is_in_grid")

    if !is_exact
        global all_exact = false
        println("  ⚠️  WARNING: Mean value not exactly in grid!")
        println("      This may cause baseline detection to fail.")
    end
    println()
end

# Test 2: Check _piecewise_range function
println("Test 2: Verifying _piecewise_range generates exact mean values")
println("-" ^ 60)

for key in FLEXECOM.Constants.PRICE_SENSITIVITY_KEYS
    params = getfield(FLEXECOM.Constants.PRICE_SENSITIVITY, key)
    grid = FLEXECOM.Financial._piecewise_range(params)
    mean_val = params.mean

    # The mean should be exactly at the center point
    center_idx = div(params.count + 1, 2)
    center_val = grid[center_idx]

    println("$key:")
    println("  Grid size: $(length(grid))")
    println("  Mean value: $mean_val")
    println("  Center index: $center_idx")
    println("  Value at center: $center_val")
    println("  Match: $(center_val == mean_val)")
    println()
end

# Test 3: Simulate baseline detection logic
println("Test 3: Simulating baseline detection in financial calculations")
println("-" ^ 60)

baseline_prices = FLEXECOM.Constants.PRICE_SENSITIVITY_MEANS
found_baseline = false
iteration_count = 0

for (pv_price, fuel_price, ng_price, ev_price) in Iterators.product(
    price_values.PV_price,
    price_values.Fuel_price,
    price_values.NG_price,
    price_values.EV_price,
)
    global iteration_count += 1
    prices = (PV_price = pv_price, Fuel_price = fuel_price,
              NG_price = ng_price, EV_price = ev_price)

    if prices == baseline_prices
        global found_baseline = true
        println("✅ Baseline found at iteration $iteration_count")
        println("   Prices: $prices")
        break
    end
end

if !found_baseline
    println("❌ ERROR: Baseline NOT found in $(iteration_count) iterations!")
    println()
    println("This confirms the bug: the baseline price combination")
    println("is not being detected due to floating-point comparison issues.")
    println()
    println("RECOMMENDATION: Use approximate comparison with isapprox()")
end

println()
println("=" ^ 60)
println("DIAGNOSTIC COMPLETE")
println("=" ^ 60)

# Return test result
@testset "Baseline Detection" begin
    @test all_exact  # Mean values should be exactly in price grids
    @test found_baseline  # Baseline should be found in price sensitivity loop
end
