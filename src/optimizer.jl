"""
    Optimizer

JuMP-based optimization model for Renewable Energy Communities.

This module implements the core hourly optimization that minimizes annual energy
costs for a community while coordinating PV generation, battery storage, electric
vehicles, heat pumps, and domestic hot water systems.

The optimization formulation follows the methodology described in:
Manso Burgos et al., "Flexible energy community optimisation for the Valencian
Community," Applied Energy (2025).

# Key Features
- Three allocation coefficient modes: static, variable (season/day), dynamic (hourly)
- Monthly net billing constraints for grid exchange
- Technology-specific constraints for BESS, EV, ASHP, and DHW

# Public Functions
- `optimize_scenario`: Run the full optimization for one scenario
- `calculate_energy_balance`: Verify energy conservation at each timestep
"""
module Optimizer

using JuMP, Gurobi
using DataFrames, LinearAlgebra
using ..Types
using ..Constants

export optimize_scenario, calculate_energy_balance

materialize_array(x) = x
materialize_array(x::JuMP.Containers.DenseAxisArray) = Array(x)

value_array(var_container) = materialize_array(value.(var_container))

"""
    optimize_scenario(scenario::Scenario, data::Dict, gurobi_env) -> Dict

Solve the hourly energy optimization problem for a given scenario.

Constructs and solves a JuMP model that minimizes annual energy costs for the
community. The objective function includes grid purchase costs, grid sale revenues,
power term costs, and battery degradation costs (LCOS).

# Arguments
- `scenario`: Scenario struct with technology deployment parameters
- `data`: Dict from `load_input_data` with all time series
- `gurobi_env`: Gurobi environment for solver configuration

# Returns
Dict containing:
- Scenario and technology parameters (`:scenario`, `:BESS_COM`, `:EV`, etc.)
- Decision variable values (`:P_pur`, `:P_sold`, `:BESS_SoC`, etc.)
- Calculated metrics (`:Cost_REC_j`, `:P_allo`, etc.)
- Original input data for post-processing

Returns `Dict(:status => termination_status)` if optimization fails.

# Allocation Coefficient Modes
- `COEF=1`: Static coefficients (one set for entire year)
- `COEF=2`: Variable coefficients (8 sets: 4 seasons × 2 day types)
- `COEF=3`: Dynamic coefficients (optimized hourly, includes surplus)

# Example
```julia
env = Gurobi.Env()
results = optimize_scenario(scenarios[1], input_data, env)
if haskey(results, :P_pur)
    println("Optimization successful")
end
```
"""
function optimize_scenario(scenario::Scenario, data, gurobi_env)
    # Create optimization model
    model = Model(() -> Gurobi.Optimizer(gurobi_env))
    for (attr, value) in DEFAULT_GUROBI_PARAMS
        set_optimizer_attribute(model, attr, value)
    end
    
    # Define parameters
    T = HOURS_PER_YEAR  # Number of hours in a year
    J = size(data[:load_DC], 2)  # Number of households
    
    # Configure PV generation
    sc_Pnom = scenario.PV * J
    gen_pow = data[:gen_CF] .* sc_Pnom
    
    # Configure grid prices
    PI_pur = data[:PI_pur] .* scenario.GRID
    PI_sell = data[:PI_sell] .* scenario.GRID
    PI_cont = data[:PI_cont]
    
    # Configure BESS
    BESS_COM = DEFAULT_BESS
    BESS_minsoc = BESS_MIN_SOC
    
    if scenario.BESS_cap == 0
        BESS = 0
    else
        BESS = 1
    end
    
    BESS_capmin = scenario.BESS_cap / BESS_COM.DoD / BESS_COM.V / BESS_COM.Eff
    b_s = 1
    b_p = ceil(BESS_capmin * 1000 / BESS_COM.capAh)
    BESS_num = b_s * b_p
    
    BESS_LCOS = scenario.BESS_price * BESS_COM.Inv / (BESS_COM.cap * 2 * BESS_COM.Cycles)
    
    # Configure EV
    sc_EV = scenario.EVnum
    
    if sc_EV == 0
        EVbool = 0
    else
        EVbool = 1
    end
    
    K = sc_EV
    EV = DEFAULT_EV
    EV_SoC_max = EV_SOC_MAX
    
    # Configure ASHP
    sc_ASHP = scenario.ASHP
    
    if sc_ASHP == 0
        ASHP = 0
    else
        ASHP = 1
    end
    
    ASHP_capheating = fill(sc_ASHP, T, J)
    ASHP_capcooling = fill(sc_ASHP, T, J)
    
    # Adjust capacities based on demand
    for p in 1:T
        for hh in 1:J
            if data[:load_Thermal][p, hh] <= 0  # Disable heating
                ASHP_capheating[p, hh] = 0
            end
            
            if data[:load_Thermal][p, hh] >= 0  # Disable cooling
                ASHP_capcooling[p, hh] = 0
            end
        end
    end
    
    # Configure DHW
    sc_DHW = scenario.DHW
    
    if sc_DHW == 0
        DHW = 0
    else
        DHW = 1
    end
    
    # DEFINE MODEL VARIABLES
    
    # Allocation coefficients based on type
    if scenario.COEF == 1 
        coef_sets = 1
        @variable(model, 0 <= Coef_allo[[1], j=1:J] <= 1)
    elseif scenario.COEF == 2
        coef_sets = maximum(data[:weekday][:,1]) * maximum(data[:weekday][:,2])
        @variable(model, 0 <= Coef_allo[1:coef_sets, j = 1:J] <= 1)
    elseif scenario.COEF == 3
        coef_sets = T
        @variable(model, 0 <= Coef_allo[t = 1:T, j = 1:J] <= 1)
    end
    
    @variable(model, 0 <= Coef_surp[t = 1:T, [1]] <= 1)
    
    # Grid power
    @variable(model, 0 <= P_pur[t=1:T, j=1:J])
    @variable(model, 0 <= P_sold[t=1:T, j=1:J])
    @variable(model, 0 <= P_cont[[1], j=1:J])
    @variable(model, 0 <= P_sc[t=1:T, j=1:J])
    @variable(model, 0 <= P_gift[t=1:T, j=1:J])
    
    # BESS
    @variable(model, BESS_minsoc * BESS_COM.cap * BESS_num <= BESS_SoC[t=1:T] <= BESS_COM.cap * BESS_num)
    @variable(model, 0 <= BESS_cha_pur[t=1:T, j=1:J])
    @variable(model, 0 <= BESS_cha_sc[t=1:T, j=1:J])
    @variable(model, 0 <= BESS_dis[t=1:T, j=1:J])
    
    # EV
    if K > 0
        @variable(model, data[:EV_SoC_min][t, k] <= EV_SoC[t=1:T, k=1:K] <= EV_SoC_max * EV.cap)
        @variable(model, 0 <= P_sc_EVj[t=1:T, j=1:J])
        @variable(model, 0 <= P_sc_EVk[t=1:T, k=1:K])
        @variable(model, 0 <= P_pur_EVk[t=1:T, k=1:K])
    else
        # Define dummy EV variables when no EVs are present
        @variable(model, EV_SoC[t=1:T, k=1:1])
        @constraint(model, [t=1:T, k=1:1], EV_SoC[t,k] == 0)
        @variable(model, 0 <= P_sc_EVj[t=1:T, j=1:J])
        @constraint(model, [t=1:T, j=1:J], P_sc_EVj[t,j] == 0)
        @variable(model, 0 <= P_sc_EVk[t=1:T, k=1:1])
        @constraint(model, [t=1:T, k=1:1], P_sc_EVk[t,k] == 0)
        @variable(model, 0 <= P_pur_EVk[t=1:T, k=1:1])
        @constraint(model, [t=1:T, k=1:1], P_pur_EVk[t,k] == 0)
    end
    
    # ASHP
    @variable(model, 0 <= W_pur_heating[t=1:T, j=1:J])
    @variable(model, 0 <= W_sc_heating[t=1:T, j=1:J])
    @variable(model, 0 <= W_pur_cooling[t=1:T, j=1:J])
    @variable(model, 0 <= W_sc_cooling[t=1:T, j=1:J])
    
    # DHW
    @variable(model, 0 <= DHW_SoC[t=1:T, j=1:J] <= data[:DHW_max])
    @variable(model, 0 <= DHW_pur[t=1:T, j=1:J])
    @variable(model, 0 <= DHW_sc[t=1:T, j=1:J])
    
    # DEFINE CONSTRAINTS
    
    # Constraints based on coefficient type
    if scenario.COEF == 1
        @constraint(model, energy_balance[t=1:T, j=1:J],
            P_sc[t, j] 
            + P_pur[t, j]
            + BESS_dis[t, j] * sqrt(BESS_COM.Eff) * BESS
            ==
            data[:load_DC][t, j]
            + BESS_cha_pur[t, j] / sqrt(BESS_COM.Eff) * BESS
            + (W_pur_heating[t, j] + W_pur_cooling[t, j]) * ASHP
            + DHW_pur[t,j] * DHW
        )

        @constraint(model, selling_balance[t=1:T, j=1:J], 
            P_sc[t, j] 
            + P_sold[t, j] 
            + P_gift[t, j]
            + BESS_cha_sc[t, j] / sqrt(BESS_COM.Eff) * BESS
            + P_sc_EVj[t, j] * EVbool
            + (W_sc_heating[t, j] + W_sc_cooling[t, j]) * ASHP
            + DHW_sc[t,j] * DHW
            ==
            Coef_allo[1, j] * gen_pow[t]
        )

        @constraint(model, nulLCOSfsell[t = 1:T], Coef_surp[t, 1] == 0)

        @constraint(model, sum_coefs, sum(Coef_allo) == 1)

    elseif scenario.COEF == 2
        @constraint(model, energy_balance[t=1:T, j=1:J],
            P_sc[t, j] 
            + P_pur[t, j]
            + BESS_dis[t, j] * sqrt(BESS_COM.Eff) * BESS
            ==
            data[:load_DC][t, j]
            + BESS_cha_pur[t, j] / sqrt(BESS_COM.Eff) * BESS
            + (W_pur_heating[t, j] + W_pur_cooling[t, j]) * ASHP
            + DHW_pur[t,j] * DHW
        )
        
        @constraint(model, selling_balance[t=1:T, j=1:J], 
            P_sc[t, j] 
            + P_sold[t, j] 
            + P_gift[t, j]
            + BESS_cha_sc[t, j] / sqrt(BESS_COM.Eff) * BESS
            + P_sc_EVj[t, j] * EVbool
            + (W_sc_heating[t, j] + W_sc_cooling[t, j]) * ASHP
            + DHW_sc[t,j] * DHW
            ==
            Coef_allo[data[:weekday][t, 3], j] * gen_pow[t]
        )

        @constraint(model, nulLCOSfsell[t = 1:T], Coef_surp[t, 1] == 0)

        @constraint(model, sum_coefs[c = 1:coef_sets], sum(Coef_allo[c,:]) == 1)

    elseif scenario.COEF == 3
        @constraint(model, energy_balance[t = 1:T, j = 1:J], 
                Coef_allo[t,j] * gen_pow[t] 
                + P_pur[t, j]                            				
                + BESS_dis[t,j] * sqrt(BESS_COM.Eff) * BESS
                == 
                data[:load_DC][t, j]
                + BESS_cha_pur[t, j] / sqrt(BESS_COM.Eff) * BESS
                + (W_pur_heating[t, j] + W_pur_cooling[t, j]) * ASHP
                + DHW_pur[t,j] * DHW
        )

        @constraint(model, selling_balance[t = 1:T, j = 1:J],
                P_sold[t, j]
                + P_gift[t, j]
                + BESS_cha_sc[t, j] / sqrt(BESS_COM.Eff) * BESS
                + P_sc_EVj[t, j] * EVbool
                + (W_sc_heating[t, j] + W_sc_cooling[t, j]) * ASHP
                + DHW_sc[t,j] * DHW
                ==
                Coef_surp[t, 1] * gen_pow[t] / J
        )

        @constraint(model, sum_coefs[t = 1:T], sum(Coef_allo[t,:]) + Coef_surp[t, 1] == 1)
    end
    
    # Net balance constraints (monthly periods)
    months = MONTHLY_HOUR_RANGES
    
    @variable(model, bill[1:length(months), j=1:J] >= 0)

    for (m, period) in enumerate(months)
        @constraint(model, [j=1:J], bill[m, j] == sum(P_pur[t, j] * PI_pur[t] for t in period))
        @constraint(model, sum(bill[m, j] for j=1:J) >=
            sum(P_sold[t, j] * PI_sell[t] for t in period, j=1:J))
    end
    
    # Maximum power constraint
    @constraint(model, max_power[t=1:T, j=1:J], P_pur[t, j] <= P_cont[1, j])
    
    # BESS constraints
    if BESS == 1
        @constraint(model, bess_net, BESS_SoC[1] == BESS_SoC[T])
        @constraint(model, bess_SoC[t=2:T], BESS_SoC[t] == BESS_SoC[t-1] + 
            sum(BESS_cha_pur[t-1, :] + BESS_cha_sc[t-1, :]) * sqrt(BESS_COM.Eff) - 
            sum(BESS_dis[t-1, :]) / sqrt(BESS_COM.Eff))
        @constraint(model, besslimitcha[t=1:T], 
            sum(BESS_cha_pur[t, :] + BESS_cha_sc[t, :]) <= BESS_COM.Pmax * BESS_num)
        @constraint(model, besslimitdis[t=1:T], 
            sum(BESS_dis[t, :]) <= BESS_COM.Pmax * BESS_num)
    end
    
    # EV constraints
    if EVbool == 1
        @constraint(model, ev_net[k=1:K], EV_SoC[1, k] == EV_SoC[T, k])
        @constraint(model, EVlimitcha[t=1:T, k=1:K], 
            P_pur_EVk[t, k] + P_sc_EVk[t, k] <= EV.Pch * data[:EV_o][t, k])
            @constraint(model, ev_soc[t=2:T, k=1:K], EV_SoC[t, k] == 
            (EV_SoC[t-1, k] + (P_pur_EVk[t-1, k] + P_sc_EVk[t-1, k]) * sqrt(EV.Eff)) * 
            data[:EV_o][t-1, k] + data[:EV_lleg][t, k] * 
            (EV_SoC[Int(data[:EV_h_sal][t, k]), k] - EV.cons_km * data[:EV_km][t, k]))
        @constraint(model, p_sc_ev[t=1:T], 
            sum(P_sc_EVj[t, 1:J]) == sum(P_sc_EVk[t, 1:K]))
    end
    
    # ASHP constraints
    if ASHP == 1
        @constraint(model, thermal_balance[t=1:T, j=1:J], 
            data[:load_Thermal][t, j] == 
            (W_pur_heating[t, j] + W_sc_heating[t, j]) .* data[:COP_Heating][t] - 
            (W_pur_cooling[t, j] + W_sc_cooling[t, j]) .* data[:COP_Cooling][t])
        
        @constraint(model, w_heating[t=1:T, j=1:J], 
            W_pur_heating[t, j] + W_sc_heating[t, j] <= ASHP_capheating[t, j])
        
        @constraint(model, w_cooling[t=1:T, j=1:J], 
            W_pur_cooling[t, j] + W_sc_cooling[t, j] <= ASHP_capcooling[t, j])
    end
    
    # DHW constraints
    if DHW == 1
        @constraint(model, dhw_net[j=1:J], DHW_SoC[1,j] == DHW_SoC[T,j])
        
        @constraint(model, dhw_soc[t=2:T, j=1:J], 
            DHW_SoC[t,j] == DHW_SoC[t-1,j] + 
            (DHW_pur[t,j] + DHW_sc[t,j]) * data[:DHW_COP] - 
            data[:DHW_flux][t,j])
    end
    
    # OBJECTIVE FUNCTION
    @objective(model, Min,
        sum(sum(P_pur[:, :] .* PI_pur -
                P_sold[:, :] .* PI_sell, dims=1)) +
        sum(P_cont[1, :] .* PI_cont) +
        sum(sum(BESS_cha_pur[:, :] + BESS_cha_sc[:, :], dims=1) .+ 
            sum(BESS_dis[:, :], dims=1)) .* BESS_LCOS * BESS +
        sum(sum(P_pur_EVk[1:T, :] .* PI_pur, dims=1)) * EVbool
    )
    
    # OPTIMIZE
    optimize!(model)
    
    # Check termination status
    status = termination_status(model)
    if status != MOI.OPTIMAL
        println("Model not feasible. Status: $status")
        return Dict(:status => status)
    end
    
    # PROCESS RESULTS
    results = Dict()
    
    # Store scenario parameters
    results[:scenario] = scenario
    results[:BESS_COM] = BESS_COM
    results[:BESS_num] = BESS_num
    results[:BESS_LCOS] = BESS_LCOS
    results[:BESS] = BESS
    results[:EV] = EV
    results[:EVbool] = EVbool
    results[:K] = K
    results[:ASHP] = ASHP
    results[:DHW] = DHW

    # Store weekday mapping used for time-varying coefficients
    results[:weekday] = data[:weekday]
    
    # Store model variables
    coef_allo_values = value_array(model[:Coef_allo])
    if scenario.COEF == 1
        results[:Coef_allo] = coef_allo_values[1, :]
    elseif scenario.COEF == 2
        results[:Coef_allo] = coef_allo_values
    elseif scenario.COEF == 3
        results[:Coef_allo] = coef_allo_values
    end

    results[:Coef_surp] = value_array(model[:Coef_surp])
    results[:P_pur] = value_array(model[:P_pur])
    results[:P_sold] = value_array(model[:P_sold])
    results[:P_cont] = value_array(model[:P_cont])
    results[:P_sc] = value_array(model[:P_sc])
    results[:P_gift] = value_array(model[:P_gift])

    results[:BESS_SoC] = value_array(model[:BESS_SoC])
    results[:BESS_cha_pur] = value_array(model[:BESS_cha_pur])
    results[:BESS_cha_sc] = value_array(model[:BESS_cha_sc])
    results[:BESS_dis] = value_array(model[:BESS_dis])
    
    if K > 0
        results[:EV_SoC] = value_array(model[:EV_SoC])
        results[:P_sc_EVj] = value_array(model[:P_sc_EVj])
        results[:P_sc_EVk] = value_array(model[:P_sc_EVk])
        results[:P_pur_EVk] = value_array(model[:P_pur_EVk])
        results[:EV_km] = data[:EV_km][:, 1:K]
    else
        results[:EV_SoC] = zeros(T, 1)
        results[:P_sc_EVj] = zeros(T, J)
        results[:P_sc_EVk] = zeros(T, 1)
        results[:P_pur_EVk] = zeros(T, 1)
        results[:EV_km] = zeros(T, 1)
    end
    
    results[:W_pur_heating] = value_array(model[:W_pur_heating])
    results[:W_sc_heating] = value_array(model[:W_sc_heating])
    results[:W_pur_cooling] = value_array(model[:W_pur_cooling])
    results[:W_sc_cooling] = value_array(model[:W_sc_cooling])

    results[:DHW_SoC] = value_array(model[:DHW_SoC])
    results[:DHW_pur] = value_array(model[:DHW_pur])
    results[:DHW_sc] = value_array(model[:DHW_sc])
    
    # Cost calculations
    results[:Cost_REC_j] = (
        sum(results[:P_pur] .* PI_pur .- results[:P_sold] .* PI_sell, dims=1) .+
        results[:P_cont][1, :]' .* PI_cont .+
        (sum(results[:W_pur_heating] + results[:W_pur_cooling], dims=1)) * ASHP
    )
    
    results[:Cost_REC_j_LCOS] = (
        sum(results[:P_pur] .* PI_pur .- results[:P_sold] .* PI_sell, dims=1) .+
        results[:P_cont][1, :]' .* PI_cont .+
        (sum(results[:BESS_cha_pur] + results[:BESS_cha_sc], dims=1) .+ 
         sum(results[:BESS_dis], dims=1)) .* BESS_LCOS * BESS .+
        (sum(results[:W_pur_heating] + results[:W_pur_cooling], dims=1)) * ASHP
    )
    
    results[:Cost_EV] = sum(results[:P_pur_EVk] .* PI_pur, dims=1) * EVbool

    # Calculated allocation coefficients for scenarios 1 and 2
    if scenario.COEF == 1 || scenario.COEF == 2
        if scenario.COEF == 1
            # For static coefficients, properly reshape
            results[:P_allo] = zeros(T, J)
            for j in 1:J
                results[:P_allo][:, j] = gen_pow .* results[:Coef_allo][j]
            end
        else
            # For variable coefficients
            coef_temp = zeros(T, J)
            for t in 1:T
                coef_temp[t, :] = results[:Coef_allo][data[:weekday][t, 3], :]
            end
            # Careful broadcasting - loop through columns
            results[:P_allo] = zeros(T, J)
            for j in 1:J
                results[:P_allo][:, j] = gen_pow .* coef_temp[:, j]
            end
        end
    else
        # For dynamic coefficients
        # Ensure gen_pow is column vector for correct broadcasting
        gen_pow_col = reshape(gen_pow, :, 1)
        results[:P_allo] = results[:Coef_allo] .* gen_pow_col
    end
    
    # Store original data
    results[:gen_pow] = gen_pow
    results[:load_DC] = data[:load_DC]
    results[:load_Thermal] = data[:load_Thermal]
    results[:DHW_flux] = data[:DHW_flux]
    results[:COP_Heating] = data[:COP_Heating]
    results[:COP_Cooling] = data[:COP_Cooling]
    
    return results
end

"""
    calculate_energy_balance(results::Dict) -> Vector{Float64}

Verify energy conservation at each timestep.

Computes the energy balance residual at each hour, which should be approximately
zero if the optimization solution is valid. This function is useful for debugging
and validating results.

# Arguments
- `results`: Dict returned by `optimize_scenario`

# Returns
Vector of length T (8760) with energy balance residuals (kWh).
Values close to zero indicate conservation; large values suggest issues.

# Balance Equation
```
generation + purchase - load - sale - gift - BESS_charge + BESS_discharge
- EV_charge - ASHP_consumption - DHW_consumption = 0
```

# Example
```julia
balance = calculate_energy_balance(results)
max_error = maximum(abs.(balance))
println("Max balance error: \$max_error kWh")
```
"""
function calculate_energy_balance(results)
    # Check energy balance at each timestep
    T = size(results[:gen_pow], 1)
    balance = zeros(T)
    
    for t = 1:T
        bess_eff = results[:BESS_COM].Eff
        balance[t] = results[:gen_pow][t] + sum(results[:P_pur][t, :]) -
                     sum(results[:load_DC][t, :]) - sum(results[:P_sold][t, :]) -
                     sum(results[:P_gift][t, :]) -
                     sum(results[:BESS_cha_pur][t, :] + results[:BESS_cha_sc][t, :]) /
                     sqrt(bess_eff) + sum(results[:BESS_dis][t, :]) * sqrt(bess_eff) -
                     sum(results[:P_sc_EVj][t, :]) -
                     sum(results[:W_pur_heating][t, :] + results[:W_pur_cooling][t, :] + 
                         results[:W_sc_heating][t, :] + results[:W_sc_cooling][t, :]) -
                     sum(results[:DHW_pur][t, :] + results[:DHW_sc][t, :])
    end
    
    return balance
end

end # module

# Use this module inside the FLEXECOM module
using .Optimizer