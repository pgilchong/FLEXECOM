@testset "Utils" begin
    scenario = FLEXECOM.Scenario(1, "Summary", 1, 1.0, 1.0, 1.0, 0.0, 0, 0.0, 0)
    zeros_matrix = zeros(2, 1)

    scenario_results = Dict(
        :scenario => scenario,
        :gen_pow => [2.0, 2.0],
        :P_pur => zeros_matrix,
        :load_DC => fill(2.0, 2, 1),
        :P_sold => zeros_matrix,
        :P_gift => zeros_matrix,
        :BESS_cha_pur => zeros_matrix,
        :BESS_cha_sc => zeros_matrix,
        :BESS_dis => zeros_matrix,
        :BESS_COM => FLEXECOM.Constants.DEFAULT_BESS,
        :BESS => 0,
        :P_sc_EVj => zeros_matrix,
        :W_pur_heating => zeros_matrix,
        :W_pur_cooling => zeros_matrix,
        :W_sc_heating => zeros_matrix,
        :W_sc_cooling => zeros_matrix,
        :DHW_pur => zeros_matrix,
        :DHW_sc => zeros_matrix,
        :Cost_REC_j => fill(10.0, 2, 1)
    )

    financial_results = Dict(
        1 => Dict(
            :initial_costs => Dict(:total => 200.0),
            :investments => Dict(:total => 150.0),
            :investment_shares => Dict(
                :pv => 0.4,
                :bess => 0.2,
                :ashp => 0.1,
                :dhw => 0.1,
                :ev => 0.2
            ),
            :npv => (120.0, 0.0, 0.0),
            :irr => (0.08, 0.0, 0.0),
            :baseline_prices => FLEXECOM.Constants.PRICE_SENSITIVITY_MEANS
        )
    )

    summary = FLEXECOM.create_summary(Dict(1 => scenario_results), financial_results)

    @test size(summary, 1) == 1
    @test summary.Scenario_ID[1] == 1
    @test summary.Total_Cost[1] == sum(scenario_results[:Cost_REC_j])
    @test summary.Initial_Cost[1] == financial_results[1][:initial_costs][:total]
    @test summary.Cost_Reduction_Pct[1] ≈ 90.0
    @test summary.PV_Investment_Share[1] == 0.4

    balance = FLEXECOM.verify_energy_balance(scenario_results)
    @test all(isapprox.(balance, 0.0, atol=1e-8))
end

