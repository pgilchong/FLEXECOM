@testset "Financial" begin
    # Test NPV calculation
    cashflows = [-1000.0, 200.0, 300.0, 400.0, 500.0]
    discount_rate = 0.05

    npv = FLEXECOM.calculate_npv(cashflows, discount_rate)

    expected_npv = -1000.0 +
                   200.0 / (1 + 0.05)^1 +
                   300.0 / (1 + 0.05)^2 +
                   400.0 / (1 + 0.05)^3 +
                   500.0 / (1 + 0.05)^4

    @test isapprox(npv, expected_npv, rtol=1e-10)

    # Test IRR calculation
    irr = FLEXECOM.calculate_irr(cashflows)
    npv_at_irr = FLEXECOM.calculate_npv(cashflows, irr)
    @test isapprox(npv_at_irr, 0.0, atol=1e-6)

    @testset "Initial costs" begin
        scenario = FLEXECOM.Scenario(1, "Init", 1, 1.0, 1.0, 1.0, 0.0, 1, 1.0, 1)

        results = Dict(
            :load_DC => reshape(Float64[1.0, 2.0], 2, 1),
            :K => 1,
            :ASHP => 1,
            :DHW => 1,
            :EV_km => reshape(Float64[5.0, 5.0], 2, 1),
            :load_Thermal => reshape(Float64[3.0, -2.0], 2, 1),
            :COP_Cooling => fill(3.0, 2, 1),
            :DHW_flux => fill(0.5, 2, 1)
        )

        input_data = Dict(
            :PI_pur => [0.3, 0.3],
            :PI_sell => [0.2, 0.2],
            :PI_cont => FLEXECOM.FINANCIAL_DEFAULT.PI_power
        )

        params = FLEXECOM.FINANCIAL_DEFAULT
        prices = (
            PV_price = 1.0,
            Fuel_price = 1.2,
            NG_price = 0.08,
            EV_price = 2000.0,
        )
        costs = FLEXECOM.calculate_initial_costs(results, input_data, scenario, params, prices)

        expected_elec = only(vec(sum(results[:load_DC] .* input_data[:PI_pur], dims=1) .+
                        maximum(results[:load_DC], dims=1) .* params.PI_power))
        expected_icev = FLEXECOM.Constants.DEFAULT_EV.cons_fuel * prices.Fuel_price *
                        sum(results[:EV_km])
        expected_heat = sum(max.(results[:load_Thermal], 0.0)) * prices.NG_price
        demand_cool = max.(-results[:load_Thermal], 0.0)
        # Correct formula: multiply by price per-hour BEFORE summing over time
        expected_cool = sum(demand_cool ./ (results[:COP_Cooling] .- 1) .* input_data[:PI_pur])
        expected_dhw = sum(results[:DHW_flux]) * prices.NG_price

        @test isapprox(only(vec(costs[:elec])), expected_elec, atol=1e-6)
        @test isapprox(only(vec(costs[:icev])), expected_icev, atol=1e-6)
        @test isapprox(only(vec(costs[:heat])), expected_heat, atol=1e-6)
        @test isapprox(sum(costs[:cool]), expected_cool, atol=1e-6)
        @test isapprox(sum(costs[:dhw]), expected_dhw, atol=1e-6)

        total_components = sum(costs[:elec]) + sum(costs[:icev]) + sum(costs[:heat]) +
                           sum(costs[:cool]) + sum(costs[:dhw])
        @test isapprox(costs[:total], total_components, atol=1e-6)
    end

    @testset "Investments" begin
        scenario = FLEXECOM.Scenario(2, "Invest", 1, 4.0, 1.1, 1.2, 6.0, 2, 5.0, 1)
        params = FLEXECOM.FINANCIAL_DEFAULT

        results = Dict(
            :load_DC => fill(1.0, 2, 2),
            :K => scenario.EVnum,
            :ASHP => 1,
            :DHW => 1,
            :BESS_COM => FLEXECOM.Constants.DEFAULT_BESS,
            :BESS_num => 3
        )

        prices = (
            PV_price = 1.0,
            Fuel_price = 1.5,
            NG_price = 0.08,
            EV_price = 2500.0,
        )

        investments = FLEXECOM.Financial.calculate_investments(results, scenario, params, prices)

        J = size(results[:load_DC], 2)
        expected_pv = params.PV_CAPEX * scenario.PV * J * prices.PV_price
        expected_inv = FLEXECOM.Constants.INVERTER_COST_SLOPE * scenario.PV * J +
                       FLEXECOM.Constants.INVERTER_COST_INTERCEPT
        expected_bess = results[:BESS_COM].Inv * results[:BESS_num] * scenario.BESS_price
        expected_ev = results[:K] * (prices.EV_price + FLEXECOM.Constants.RECHARGE_POINT_COST)
        expected_ashp = FLEXECOM.Constants.ASHP_BASE_PRICE * J
        expected_dhw = params.DHW_price * J
        expected_total = expected_pv + expected_inv + expected_bess + expected_ev + expected_ashp + expected_dhw

        @test isapprox(investments[:pv], expected_pv, atol=1e-6)
        @test isapprox(investments[:inv], expected_inv, atol=1e-6)
        @test isapprox(investments[:bess], expected_bess, atol=1e-6)
        @test isapprox(investments[:ev], expected_ev, atol=1e-6)
        @test isapprox(investments[:ashp], expected_ashp, atol=1e-6)
        @test isapprox(investments[:dhw], expected_dhw, atol=1e-6)
        @test isapprox(investments[:total], expected_total, atol=1e-6)

        shares = FLEXECOM.Financial.calculate_investment_shares(investments)
        @test isapprox(shares[:pv], (investments[:pv] + investments[:inv]) / investments[:total], atol=1e-10)
        @test isapprox(shares[:bess], investments[:bess] / investments[:total], atol=1e-10)
        @test isapprox(shares[:ashp], investments[:ashp] / investments[:total], atol=1e-10)
        @test isapprox(shares[:dhw], investments[:dhw] / investments[:total], atol=1e-10)
        @test isapprox(shares[:ev], investments[:ev] / investments[:total], atol=1e-10)

        operation_costs = FLEXECOM.Financial.calculate_operation_costs(results, investments, scenario, params)
        @test operation_costs[:pv] > 0
        @test operation_costs[:total] >= operation_costs[:pv]
    end

    @testset "Price sensitivity grid" begin
        scenario = FLEXECOM.Scenario(3, "PriceGrid", 1, 2.0, 1.0, 1.0, 1.0, 1, 1.0, 1)

        load_dc = fill(1.0, 2, 2)
        thermal = [1.0 -0.5; 0.8 -0.4]
        bess_specs = FLEXECOM.Constants.DEFAULT_BESS

        scenario_results = Dict(
            3 => Dict(
                :scenario => scenario,
                :load_DC => load_dc,
                :K => scenario.EVnum,
                :ASHP => scenario.ASHP,
                :DHW => scenario.DHW,
                :EV_km => fill(6.0, 2, scenario.EVnum),
                :load_Thermal => thermal,
                :COP_Cooling => fill(3.0, 2, 2),
                :DHW_flux => fill(0.2, 2, 2),
                :BESS_COM => bess_specs,
                :BESS_num => 1,
                :P_allo => fill(0.5, 2, 2),
                :gen_pow => fill(2.0, 2, 2),
                :Coef_surp => zeros(2, 2),
                :BESS_cha_pur => fill(0.05, 2, 2),
                :BESS_cha_sc => fill(0.05, 2, 2),
                :P_pur => fill(0.4, 2, 2),
                :P_pur_EVk => fill(0.1, 2, scenario.EVnum),
                :P_sc => fill(0.2, 2, 2),
                :P_sold => fill(0.05, 2, 2),
                :BESS_dis => fill(0.05, 2, 2),
                :W_pur_heating => fill(0.2, 2, 2),
                :W_pur_cooling => fill(0.1, 2, 2),
                :W_sc_heating => fill(0.1, 2, 2),
                :W_sc_cooling => fill(0.05, 2, 2),
                :DHW_pur => fill(0.05, 2, 2),
                :DHW_sc => fill(0.02, 2, 2),
                :Cost_REC_j => fill(4.0, 2, 2),
                :Cost_REC_j_LCOS => fill(4.0, 2, 2),
                :P_gift => zeros(2, 2),
                :BESS => 1,
                :EVbool => 1,
                :EV => FLEXECOM.Constants.DEFAULT_EV,
                :BESS_SoC => fill(0.5, 2, 2),
                :EV_SoC => fill(0.6, 2, scenario.EVnum),
                :P_sc_EVj => fill(0.0, 2, 2),
            )
        )

        input_data = Dict(
            :PI_pur => fill(0.2, 2),
            :PI_sell => fill(0.15, 2),
            :PI_cont => FLEXECOM.FINANCIAL_DEFAULT.PI_power,
        )

        price_values = FLEXECOM.Financial.price_sensitivity_values()
        expected_rows = prod(length.(Tuple(price_values)))

        fin_results = FLEXECOM.calculate_financial_metrics(scenario_results, FLEXECOM.FINANCIAL_DEFAULT, input_data)
        df = fin_results[3][:price_sensitivity]

        @test size(df, 1) == expected_rows

        means = FLEXECOM.Constants.PRICE_SENSITIVITY_MEANS
        mask = (df.PV_price .== means.PV_price) .&
               (df.Fuel_price .== means.Fuel_price) .&
               (df.NG_price .== means.NG_price) .&
               (df.EV_price .== means.EV_price)

        idx = findfirst(mask)
        @test !isnothing(idx)
        @test fin_results[3][:baseline_prices] == means
        @test df.initial_cost_total[idx] ≈ fin_results[3][:initial_costs][:total]
    end
end
