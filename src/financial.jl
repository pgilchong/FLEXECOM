"""
    Financial

Techno-economic analysis for Renewable Energy Community investments.

This module calculates financial metrics (NPV, IRR) for REC investments,
including baseline cost estimation, investment calculations, operating cost
projections, and multi-dimensional price sensitivity analysis.

# Methodology
The analysis compares the REC scenario against a baseline where:
- Electricity is purchased from the grid at reference prices
- Vehicles are ICEVs consuming fossil fuel
- Heating uses natural gas boilers
- Cooling uses inefficient standalone AC units

# Public Functions
- `calculate_financial_metrics`: Complete financial analysis with price sensitivity
- `calculate_npv`: Net Present Value calculation
- `calculate_irr`: Internal Rate of Return calculation
- `calculate_initial_costs`: Baseline cost estimation
- `price_sensitivity_values`: Generate price variation grids
"""
module Financial

using DataFrames, Statistics, Dates
using ..Types
using ..Constants

export calculate_financial_metrics, calculate_npv, calculate_irr, calculate_initial_costs,
       price_sensitivity_values

function _piecewise_range(params)
    min_val = params.min
    mean_val = params.mean
    max_val = params.max
    count = params.count

    count >= 3 || throw(ArgumentError("price sensitivity grids require at least 3 points"))
    isodd(count) || throw(ArgumentError("price sensitivity grids must have an odd number of points"))
    min_val <= mean_val <= max_val || throw(ArgumentError("expected min ≤ mean ≤ max for price sensitivity configuration"))

    lower_steps = ceil(Int, (count - 1) / 2)
    upper_steps = (count - 1) - lower_steps

    lower = collect(range(min_val, stop=mean_val, length=lower_steps + 1))
    upper = collect(range(mean_val, stop=max_val, length=upper_steps + 1))

    return vcat(lower[1:end-1], upper)
end

"""
    price_sensitivity_values() -> NamedTuple

Generate price variation grids for sensitivity analysis.

Returns grids for PV, fuel, natural gas, and EV prices based on the
`PRICE_SENSITIVITY` configuration in `constants.jl`. Each grid has an odd
number of points with the mean value exactly at the center.

# Returns
NamedTuple with fields `:PV_price`, `:Fuel_price`, `:NG_price`, `:EV_price`,
each containing a Vector of price values to evaluate.

# Example
```julia
prices = price_sensitivity_values()
println("PV prices: \$(prices.PV_price)")  # e.g., [0.4, 0.5, ..., 1.25]
```
"""
function price_sensitivity_values()
    (
        PV_price = _piecewise_range(PRICE_SENSITIVITY.PV_price),
        Fuel_price = _piecewise_range(PRICE_SENSITIVITY.Fuel_price),
        NG_price = _piecewise_range(PRICE_SENSITIVITY.NG_price),
        EV_price = _piecewise_range(PRICE_SENSITIVITY.EV_price),
    )
end

"""
    calculate_financial_metrics(scenario_results, financial_params, input_data) -> Dict

Perform complete financial analysis for all scenarios with price sensitivity.

Iterates over all combinations of price assumptions and calculates NPV, IRR,
investment costs, operating costs, and savings for each scenario. Results at
baseline prices are stored separately for quick access.

# Arguments
- `scenario_results`: Dict of optimization results keyed by scenario ID
- `financial_params`: `Financial_params` struct with analysis parameters
- `input_data`: Dict from `load_input_data`

# Returns
Dict keyed by scenario ID, each containing:
- `:initial_costs`: Baseline annual costs before REC
- `:investments`: Technology investment breakdown
- `:operation_costs`: Annual O&M costs
- `:savings`: Annual cost reductions
- `:npv`: (REC NPV, household NPVs, EV NPV)
- `:irr`: (REC IRR, household IRRs, EV IRR)
- `:investment_shares`: Fractional investment by technology
- `:energy_shares`: Energy allocation per household
- `:price_sensitivity`: DataFrame with all price combinations
- `:baseline_prices`: Reference prices used for main results

# Example
```julia
fin_results = calculate_financial_metrics(results, FINANCIAL_DEFAULT, input_data)
println("NPV: \$(fin_results[1][:npv][1]) €")
```
"""
function calculate_financial_metrics(scenario_results, financial_params, input_data)
    financial_results = Dict()

    price_values = price_sensitivity_values()
    baseline_prices = PRICE_SENSITIVITY_MEANS

    for (scenario_id, results) in scenario_results
        scenario = results[:scenario]
        financial_results[scenario_id] = Dict()

        # Calculate household energy shares
        energy_shares = calculate_energy_shares(results)
        scenario_df = DataFrame(
            PV_price = Float64[],
            Fuel_price = Float64[],
            NG_price = Float64[],
            EV_price = Float64[],
            initial_cost_total = Float64[],
            investment_total = Float64[],
            operation_cost_total = Float64[],
            savings_total = Float64[],
            npv_rec = Float64[],
            irr_rec = Float64[],
        )

        baseline_data = nothing

        for (pv_price, fuel_price, ng_price, ev_price) in Iterators.product(
            price_values.PV_price,
            price_values.Fuel_price,
            price_values.NG_price,
            price_values.EV_price,
        )
            prices = (PV_price = pv_price, Fuel_price = fuel_price,
                      NG_price = ng_price, EV_price = ev_price)

            initial_costs = calculate_initial_costs(results, input_data, scenario, financial_params, prices)
            investments = calculate_investments(results, scenario, financial_params, prices)
            operation_costs = calculate_operation_costs(results, investments, scenario, financial_params)
            savings = calculate_savings(results, initial_costs, scenario, input_data)

            npv_rec, npv_j, npv_ev = calculate_npv_detailed(
                investments, operation_costs, savings, energy_shares, financial_params)
            irr_rec, irr_j, irr_ev = calculate_irr_detailed(
                investments, operation_costs, savings, energy_shares, financial_params)

            push!(scenario_df, (
                pv_price,
                fuel_price,
                ng_price,
                ev_price,
                initial_costs[:total],
                investments[:total],
                operation_costs[:total],
                savings[:total],
                npv_rec,
                irr_rec,
            ))

            if prices == baseline_prices
                baseline_data = Dict(
                    :initial_costs => initial_costs,
                    :investments => investments,
                    :operation_costs => operation_costs,
                    :savings => savings,
                    :npv => (npv_rec, npv_j, npv_ev),
                    :irr => (irr_rec, irr_j, irr_ev),
                    :investment_shares => calculate_investment_shares(investments),
                )
            end
        end

        baseline_data === nothing && error("baseline price combination was not evaluated")

        for (key, value) in baseline_data
            financial_results[scenario_id][key] = value
        end
        financial_results[scenario_id][:energy_shares] = energy_shares
        financial_results[scenario_id][:price_sensitivity] = scenario_df
        financial_results[scenario_id][:baseline_prices] = baseline_prices
    end

    return financial_results
end

"""
    calculate_initial_costs(results, input_data, scenario, params, prices) -> Dict

Estimate baseline annual costs without the REC investment.

Calculates what the community would pay annually without PV, BESS, EVs, ASHPs,
or DHW systems. This baseline is used to compute savings from the REC.

# Arguments
- `results`: Optimization results Dict
- `input_data`: Dict from `load_input_data`
- `scenario`: Scenario struct
- `params`: Financial_params struct
- `prices`: NamedTuple with current price assumptions

# Returns
Dict with keys:
- `:elec`: Electricity costs (variable + power term)
- `:icev`: Fuel costs for equivalent ICEVs
- `:heat`: Natural gas costs for heating
- `:cool`: Electricity costs for inefficient AC cooling
- `:dhw`: Natural gas costs for water heating
- `:total`: Sum of all components
"""
function calculate_initial_costs(results, input_data, scenario, params, prices)
    initial_costs = Dict()

    # Get scenario parameters
    J = size(results[:load_DC], 2)
    K = results[:K]
    ASHP = results[:ASHP]
    DHW = results[:DHW]
    
    # Reference prices
    PI_ref = input_data[:PI_pur]
    
    # Calculate electricity costs
    Cost_Ini_elec_var = sum(results[:load_DC] .* PI_ref, dims=1)
    Cost_Ini_elec_pow = maximum(results[:load_DC], dims=1) .* params.PI_power
    Cost_Ini_elec = Cost_Ini_elec_var .+ Cost_Ini_elec_pow
    
    # Calculate ICEV costs
    Cost_Ini_ICEV = zeros(1, K)
    if K > 0
        Cost_Ini_ICEV = DEFAULT_EV.cons_fuel * prices.Fuel_price *
                        sum(results[:EV_km], dims=1)
    end

    # Calculate thermal demand costs
    thermal_demand = results[:load_Thermal]
    
    # Heating demand (positive values)
    Demand_Heat = zeros(size(thermal_demand))
    Demand_Heat[thermal_demand .> 0] = thermal_demand[thermal_demand .> 0]
    
    # Cooling demand (negative values)
    Demand_Cool = zeros(size(thermal_demand))
    Demand_Cool[thermal_demand .< 0] = -thermal_demand[thermal_demand .< 0]
    
    # Calculate costs with natural gas for heating and inefficient AC for cooling
    Cost_Ini_Heat = sum(Demand_Heat, dims=1) .* prices.NG_price
    
    # Assume inefficient AC has COP of actual COP - 1
    ASHP_ACcop_ini = results[:COP_Cooling] .- 1
    Cost_Ini_Cool = sum(Demand_Cool ./ ASHP_ACcop_ini, dims=1) .* PI_ref
    
    # DHW costs with natural gas
    Cost_Ini_DHW = sum(results[:DHW_flux], dims=1) .* prices.NG_price
    
    # Store all initial costs
    initial_costs[:elec] = Cost_Ini_elec
    initial_costs[:icev] = Cost_Ini_ICEV
    initial_costs[:heat] = Cost_Ini_Heat
    initial_costs[:cool] = Cost_Ini_Cool
    initial_costs[:dhw] = Cost_Ini_DHW
    initial_costs[:total] = sum(Cost_Ini_elec) + sum(Cost_Ini_ICEV) + sum(Cost_Ini_Heat) + 
                            sum(sum(Cost_Ini_Cool)) + sum(sum(Cost_Ini_DHW))
    
    return initial_costs
end

function calculate_investments(results, scenario, params, prices)
    investments = Dict()
    
    # Get scenario parameters
    J = size(results[:load_DC], 2)
    K = results[:K]
    ASHP = results[:ASHP]
    DHW = results[:DHW]
    
    # PV investment
    inv_pv = params.PV_CAPEX * scenario.PV * J * prices.PV_price
    investments[:pv] = inv_pv
    
    # Inverter investment (R² = 0.9777)
    inv_inv = INVERTER_COST_SLOPE * scenario.PV * J + INVERTER_COST_INTERCEPT
    investments[:inv] = inv_inv
    
    # BESS investment
    if scenario.BESS_cap > 0
        inv_bess = results[:BESS_COM].Inv * results[:BESS_num] * scenario.BESS_price
    else
        inv_bess = 0
    end
    investments[:bess] = inv_bess
    
    # EV investment (additional cost compared to ICEV)
    inv_ev = 0
    if K > 0
        inv_ev = K * (prices.EV_price + RECHARGE_POINT_COST)
    end
    investments[:ev] = inv_ev
    
    # ASHP investment
    inv_ashp = 0
    if ASHP == 1
        inv_ashp = ASHP_BASE_PRICE * J
    end
    investments[:ashp] = inv_ashp
    
    # DHW investment
    inv_dhw = 0
    if DHW == 1
        inv_dhw = params.DHW_price * J
    end
    investments[:dhw] = inv_dhw
    
    # Total investment
    investments[:total] = inv_pv + inv_inv + inv_bess + inv_ev + inv_ashp + inv_dhw
    
    return investments
end

function calculate_operation_costs(results, investments, scenario, params)
    operation_costs = Dict()
    
    # Get scenario parameters
    J = size(results[:load_DC], 2)
    
    # PV O&M (€/kWp/year)
    om_pv = scenario.PV * J * params.OM_PV
    
    # BESS O&M
    om_bess = 0
    if scenario.BESS_cap > 0
        om_bess = params.OM_BESS * investments[:bess]
    end
    
    # EV O&M (recharge point)
    om_ev = 0
    if results[:K] > 0
        om_ev = params.OM_RP * RECHARGE_POINT_COST * results[:K]
    end
    
    # ASHP O&M
    om_ashp = 0
    if results[:ASHP] == 1
        om_ashp = params.OM_ASHP * investments[:ashp]
    end
    
    # DHW O&M
    om_dhw = 0
    if results[:DHW] == 1
        om_dhw = params.OM_DHW * investments[:dhw]
    end
    
    # Total O&M
    operation_costs[:pv] = om_pv
    operation_costs[:bess] = om_bess
    operation_costs[:ev] = om_ev
    operation_costs[:ashp] = om_ashp
    operation_costs[:dhw] = om_dhw
    operation_costs[:total] = om_pv + om_bess + om_ev + om_ashp + om_dhw
    
    return operation_costs
end

function calculate_energy_shares(results)
    shares = Dict()
    
    # Get scenario parameters
    scenario = results[:scenario]
    J = size(results[:load_DC], 2)
    
    # Calculate energy share per household
    if scenario.PV > 0
        total_allocated = sum(results[:P_allo])
        if isapprox(total_allocated, 0.0; atol=1e-9)
            # Avoid division by zero when there is no allocated energy
            shares[:j] = ones(1, J) / J
        elseif scenario.COEF == 1 || scenario.COEF == 2
            # For static or variable coefficients use the actually allocated energy
            shares[:j] = sum(results[:P_allo], dims=1) ./ total_allocated
        elseif scenario.COEF == 3
            # For dynamic coefficients the allocated energy already accounts for the surplus
            shares[:j] = sum(results[:P_allo], dims=1) ./ total_allocated
        end
    elseif scenario.BESS_cap > 0
        # If no PV but BESS exists, share based on BESS charging
        bess_cha = results[:BESS_cha_pur] + results[:BESS_cha_sc]
        total_bess = sum(bess_cha)
        if isapprox(total_bess, 0.0; atol=1e-9)
            shares[:j] = ones(1, J) / J
        else
            shares[:j] = sum(bess_cha, dims=1) / total_bess
        end
    else
        # Default equal shares
        shares[:j] = ones(1, J) / J
    end
    
    # Verify total shares sum to 1
    if abs(sum(shares[:j]) - 1.0) > 1e-6
        @warn "Energy shares don't sum to 1: $(sum(shares[:j]))"
        # Normalize to ensure sum = 1
        shares[:j] = shares[:j] / sum(shares[:j])
    end
    
    return shares
end

function calculate_savings(results, initial_costs, scenario, input_data)
    savings = Dict()
    
    # Get scenario parameters
    J = size(results[:load_DC], 2)
    K = results[:K]
    ASHP = results[:ASHP]
    DHW = results[:DHW]
    
    # Grid prices with multiplier
    PI_pur = input_data[:PI_pur] .* scenario.GRID
    PI_sell = input_data[:PI_sell] .* scenario.GRID
    
    # REC electricity savings (difference between initial and optimized costs)
    elec_costs_rec = sum(results[:P_pur] .* PI_pur, dims=1) + 
                     maximum(results[:P_pur], dims=1) .* input_data[:PI_cont]
    savings_elec = initial_costs[:elec] - elec_costs_rec
    
    # EV savings (difference between ICEV fuel costs and EV electricity costs)
    savings_ev = zeros(1, K)
    if K > 0
        ev_opex = sum(results[:P_pur_EVk] .* PI_pur, dims=1)
        savings_ev = initial_costs[:icev] - ev_opex
    end
    
    # ASHP savings
    savings_ashp = 0
    if ASHP == 1
        ashp_opex = sum(sum((results[:W_pur_heating] + results[:W_pur_cooling]) .* 
                          PI_pur, dims=1))
        savings_ashp = sum(initial_costs[:heat]) + sum(sum(initial_costs[:cool])) - ashp_opex
    end
    
    # DHW savings
    savings_dhw = 0
    if DHW == 1
        dhw_opex = sum(sum(results[:DHW_pur] .* PI_pur))
        savings_dhw = sum(sum(initial_costs[:dhw])) - dhw_opex
    end
    
    # Calculate PV savings
    if scenario.COEF == 1 || scenario.COEF == 2
        # For static or variable coefficients
        savings_pv = sum(sum(results[:P_sc], dims=2) .* PI_pur + 
                       sum(max.(results[:P_sold] - results[:BESS_dis], 0), dims=2) .* 
                       PI_sell)
    elseif scenario.COEF == 3
        # For dynamic coefficients
        savings_pv = sum(sum(results[:gen_pow] .* (1 .- results[:Coef_surp]) .* 
                          PI_pur))
    else
        savings_pv = 0
    end
    
    # Calculate BESS savings (total - other components)
    savings_bess = sum(savings_elec) + sum(savings_ev) + savings_ashp + savings_dhw - 
                  savings_pv
    
    # Store all savings
    savings[:elec] = savings_elec
    savings[:ev] = savings_ev
    savings[:ashp] = savings_ashp
    savings[:dhw] = savings_dhw
    savings[:pv] = savings_pv
    savings[:bess] = savings_bess
    savings[:total] = sum(savings_elec) + sum(savings_ev) + savings_ashp + savings_dhw
    
    return savings
end

function calculate_investment_shares(investments)
    shares = Dict()
    
    total_inv = investments[:total]
    
    if total_inv > 0
        shares[:pv] = (investments[:pv] + investments[:inv]) / total_inv
        shares[:bess] = investments[:bess] / total_inv
        shares[:ashp] = investments[:ashp] / total_inv
        shares[:dhw] = investments[:dhw] / total_inv
        shares[:ev] = investments[:ev] / total_inv
    else
        shares[:pv] = 0
        shares[:bess] = 0
        shares[:ashp] = 0
        shares[:dhw] = 0
        shares[:ev] = 0
    end
    
    return shares
end

function calculate_npv_detailed(investments, operation_costs, savings, energy_shares, params)
    # Initialize variables
    N = params.N  # Time horizon
    d = params.d  # Discount rate
    shares_j = vec(energy_shares[:j])
    J = length(shares_j)
    savings_elec = vec(savings[:elec])
    
    # Calculate BESS lifetime and replacements
    bess_cycles_yr = 0
    bess_life = N
    if investments[:bess] > 0
        # Simplified calculation: assume BESS lasts 10 years
        bess_life = 10
    end
    
    # Initialize NPV arrays
    npv_inv_rec = zeros(N)
    npv_inv_j = zeros(N, J)
    npv_inv_ev = zeros(N)
    
    npv_om = zeros(N)
    npv_om_j = zeros(N, J)
    npv_om_ev = zeros(N)
    
    npv_sav = zeros(N)
    npv_sav_j = zeros(N, J)
    npv_sav_ev = zeros(N)
    
    # Year 1: Initial investments
    npv_inv_rec[1] = investments[:total]
    npv_inv_j[1, :] = (investments[:pv] + investments[:inv] + investments[:bess]) .*
                      shares_j .+ investments[:ashp] / J .+
                      investments[:dhw] / J
    npv_inv_ev[1] = investments[:ev]
    
    # Year 1: O&M and savings
    npv_om[1] = operation_costs[:total] / d * (1 - 1/(1+d))
    npv_om_j[1, :] = (operation_costs[:pv] + operation_costs[:bess]) .*
                     shares_j ./ d .* (1 .- 1 ./ (1 .+ d)) .+
                     operation_costs[:ashp] / J / d * (1 - 1/(1+d)) .+
                     operation_costs[:dhw] / J / d * (1 - 1/(1+d))
    npv_om_ev[1] = operation_costs[:ev] / d * (1 - 1/(1+d))
    
    npv_sav[1] = savings[:total] / d * (1 - 1/(1+d))
    npv_sav_j[1, :] = savings_elec ./ d .* (1 .- 1 ./ (1 .+ d))
    if size(savings[:ev], 2) > 0
        npv_sav_ev[1] = sum(savings[:ev]) / d * (1 - 1/(1+d))
    end
    
    # Years 2 to N: Account for replacements, O&M, and savings
    for n = 2:N
        # Investment replacements
        if mod(n-1, bess_life) == 0 && mod(n-1, params.INV_life) == 0
            # Replace both BESS and inverter
            npv_inv_rec[n] = npv_inv_rec[n-1] + (investments[:bess] + investments[:inv]) / 
                             (1+d)^n
            npv_inv_j[n, :] = npv_inv_j[n-1, :] + (investments[:bess] + investments[:inv]) ./
                              (1+d)^n .* shares_j
        elseif mod(n-1, bess_life) == 0
            # Replace only BESS
            npv_inv_rec[n] = npv_inv_rec[n-1] + investments[:bess] / (1+d)^n
            npv_inv_j[n, :] = npv_inv_j[n-1, :] + investments[:bess] ./ (1+d)^n .*
                              shares_j
        elseif mod(n-1, params.INV_life) == 0
            # Replace only inverter
            npv_inv_rec[n] = npv_inv_rec[n-1] + investments[:inv] / (1+d)^n
            npv_inv_j[n, :] = npv_inv_j[n-1, :] + investments[:inv] ./ (1+d)^n .*
                              shares_j
        else
            # No replacements
            npv_inv_rec[n] = npv_inv_rec[n-1]
            npv_inv_j[n, :] = npv_inv_j[n-1, :]
        end
        
        npv_inv_ev[n] = npv_inv_ev[n-1]
        
        # O&M and savings NPV
        npv_om[n] = npv_om[n-1] + operation_costs[:total] / d * (1 - 1/(1+d)^n)
        npv_om_j[n, :] = npv_om_j[n-1, :] + (operation_costs[:pv] +
                         operation_costs[:bess]) .* shares_j ./ d .*
                         (1 .- 1 ./ (1 .+ d).^n) .+
                         operation_costs[:ashp] / J / d *
                         (1 - 1/(1+d)^n) .+
                         operation_costs[:dhw] / J / d *
                         (1 - 1/(1+d)^n)
        npv_om_ev[n] = npv_om_ev[n-1] + operation_costs[:ev] / d * (1 - 1/(1+d)^n)
        
        npv_sav[n] = npv_sav[n-1] + savings[:total] / d * (1 - 1/(1+d)^n)
        npv_sav_j[n, :] = npv_sav_j[n-1, :] + savings_elec ./ d .* (1 .- 1 ./ (1 .+ d).^n)
        
        if size(savings[:ev], 2) > 0
            npv_sav_ev[n] = npv_sav_ev[n-1] + sum(savings[:ev]) / d * (1 - 1/(1+d)^n)
        end
    end
    
    # Calculate BESS residual value at end of project
    residual_bess = 0
    if investments[:bess] > 0
        # Simplified residual value calculation
        replace_bess = floor(N/bess_life) * bess_life
        if replace_bess < N
            residual_bess = investments[:bess] / (1+d)^replace_bess * 
                           (1 - (N - replace_bess) / bess_life)
        end
    end
    
    # Final NPV calculation
    npv_rec = npv_sav[N] - npv_om[N] - npv_inv_rec[N] + residual_bess
    npv_j = npv_sav_j[N, :] - npv_om_j[N, :] - npv_inv_j[N, :] +
            residual_bess .* shares_j
    npv_ev = npv_sav_ev[N] - npv_om_ev[N] - npv_inv_ev[N]
    
    return npv_rec, npv_j, npv_ev
end

"""
    calculate_npv(cashflows::Vector, discount_rate::Real) -> Float64

Calculate Net Present Value of a cash flow series.

Uses the standard NPV formula where year 0 is not discounted.

# Arguments
- `cashflows`: Vector of annual cash flows (year 0, 1, 2, ...)
- `discount_rate`: Annual discount rate (e.g., 0.02 for 2%)

# Returns
Net Present Value in the same currency as the cash flows.

# Formula
```
NPV = Σ(CF_t / (1 + r)^t) for t = 0, 1, 2, ...
```

# Example
```julia
cf = [-10000, 2000, 2500, 3000, 3500]  # Initial investment + returns
npv = calculate_npv(cf, 0.05)
```
"""
function calculate_npv(cashflows, discount_rate)
    npv = 0.0
    for (t, cf) in enumerate(cashflows)
        npv += cf / (1 + discount_rate)^(t-1)
    end
    return npv
end

function calculate_irr_detailed(investments, operation_costs, savings, energy_shares, params)
    # Calculate cash flows for REC, households, and EVs
    N = params.N
    shares_j = vec(energy_shares[:j])
    J = length(shares_j)
    savings_elec = vec(savings[:elec])

    # Initialize cash flow arrays
    cf_rec = zeros(N+1)
    cf_j = zeros(N+1, J)
    cf_ev = zeros(N+1)
    
    # Initial investment (year 0)
    cf_rec[1] = -investments[:total]
    cf_j[1, :] = -(investments[:pv] + investments[:inv] + investments[:bess]) .*
                 shares_j .- investments[:ashp] / J .- investments[:dhw] / J
    cf_ev[1] = -investments[:ev]
    
    # BESS lifetime calculation
    bess_life = N
    if investments[:bess] > 0
        # Simplified calculation: assume BESS lasts 10 years
        bess_life = 10
    end
    
    # Years 1 to N: Annual cash flows
    for n = 1:N
        # Year n cash flow is at index n+1
        if mod(n, bess_life) == 0 && mod(n, params.INV_life) == 0
            # Replace both BESS and inverter
            cf_rec[n+1] = savings[:total] - operation_costs[:total] - 
                         (investments[:bess] + investments[:inv])
            cf_j[n+1, :] = savings_elec -
                          (operation_costs[:pv] + operation_costs[:bess]) .* shares_j .-
                          operation_costs[:ashp] / J .- operation_costs[:dhw] / J .-
                          (investments[:bess] + investments[:inv]) .* shares_j
        elseif mod(n, bess_life) == 0
            # Replace only BESS
            cf_rec[n+1] = savings[:total] - operation_costs[:total] - investments[:bess]
            cf_j[n+1, :] = savings_elec -
                          (operation_costs[:pv] + operation_costs[:bess]) .* shares_j .-
                          operation_costs[:ashp] / J .- operation_costs[:dhw] / J .-
                          investments[:bess] .* shares_j
        elseif mod(n, params.INV_life) == 0
            # Replace only inverter
            cf_rec[n+1] = savings[:total] - operation_costs[:total] - investments[:inv]
            cf_j[n+1, :] = savings_elec -
                          (operation_costs[:pv] + operation_costs[:bess]) .* shares_j .-
                          operation_costs[:ashp] / J .- operation_costs[:dhw] / J .-
                          investments[:inv] .* shares_j
        else
            # No replacements
            cf_rec[n+1] = savings[:total] - operation_costs[:total]
            cf_j[n+1, :] = savings_elec -
                          (operation_costs[:pv] + operation_costs[:bess]) .* shares_j .-
                          operation_costs[:ashp] / J .- operation_costs[:dhw] / J
        end
        
        if size(savings[:ev], 2) > 0
            cf_ev[n+1] = sum(savings[:ev]) - operation_costs[:ev]
        end
    end
    
    # Calculate BESS residual value at end of project
    residual_bess = 0
    if investments[:bess] > 0
        # Simplified residual value calculation
        replace_bess = floor(N/bess_life) * bess_life
        if replace_bess < N
            residual_bess = investments[:bess] * (1 - (N - replace_bess) / bess_life)
        end
    end
    
    # Add residual value to final year cash flow
    cf_rec[N+1] += residual_bess
    cf_j[N+1, :] += residual_bess .* shares_j
    
    # Calculate IRR
    irr_rec = calculate_irr(cf_rec)
    irr_j = [calculate_irr(cf_j[:, j]) for j in 1:J]
    irr_ev = calculate_irr(cf_ev)
    
    return irr_rec, irr_j, irr_ev
end

"""
    calculate_irr(cashflows::Vector) -> Float64

Calculate Internal Rate of Return using the bisection method.

Finds the discount rate at which NPV equals zero. Handles edge cases
where all cash flows have the same sign.

# Arguments
- `cashflows`: Vector of annual cash flows (year 0, 1, 2, ...)

# Returns
IRR as a decimal (e.g., 0.15 for 15%). Returns -1.0 if no positive
cash flows, or 1.0 if no negative cash flows.

# Algorithm
Uses bisection search with automatic range expansion if the solution
lies outside the initial [-0.99, 1.0] range.

# Example
```julia
cf = [-10000, 2000, 3000, 4000, 5000]
irr = calculate_irr(cf)
println("IRR: \$(round(irr * 100, digits=2))%")
```
"""
function calculate_irr(cashflows)
    # Simple IRR calculation using bisection method

    # Special cases
    if all(cashflows .<= 0)
        return -1.0  # No positive cash flows, return -100%
    elseif all(cashflows .>= 0)
        return 1.0   # No negative cash flows, return 100%
    end

    # Bisection method to find IRR
    function npv_at_rate(rate)
        return sum(cashflows ./ (1 + rate).^(0:length(cashflows)-1))
    end

    # Initial guesses
    rate_lower = -0.99  # Lowest possible rate: -99%
    rate_upper = 1.0    # Highest rate to try: 100%

    # Check if solution exists in range and expand if necessary
    npv_lower = npv_at_rate(rate_lower)
    npv_upper = npv_at_rate(rate_upper)

    expansion_steps = 0
    while sign(npv_lower) == sign(npv_upper) && expansion_steps < 20
        rate_upper += 1.0
        npv_upper = npv_at_rate(rate_upper)
        expansion_steps += 1
    end

    if sign(npv_lower) == sign(npv_upper)
        return abs(npv_lower) < abs(npv_upper) ? rate_lower : rate_upper
    end

    # Bisection iterations
    max_iter = 200
    npv_tolerance = 1e-9

    if abs(npv_lower) < abs(npv_upper)
        best_rate = rate_lower
        best_npv = abs(npv_lower)
    else
        best_rate = rate_upper
        best_npv = abs(npv_upper)
    end

    for i in 1:max_iter
        rate_mid = (rate_lower + rate_upper) / 2
        npv_mid = npv_at_rate(rate_mid)

        if abs(npv_mid) < best_npv
            best_rate = rate_mid
            best_npv = abs(npv_mid)
        end

        if abs(npv_mid) < npv_tolerance
            return rate_mid
        end

        if npv_mid * npv_lower > 0
            rate_lower = rate_mid
            npv_lower = npv_mid
        else
            rate_upper = rate_mid
            npv_upper = npv_mid
        end
    end

    # Return the estimate that gives the smallest residual NPV
    return best_rate
end

end # module
