@testset "Optimizer" begin
    bess = FLEXECOM.Constants.DEFAULT_BESS
    zeros_matrix = zeros(2, 1)

    results = Dict(
        :gen_pow => [2.0, 2.0],
        :P_pur => zeros_matrix,
        :load_DC => fill(2.0, 2, 1),
        :P_sold => zeros_matrix,
        :P_gift => zeros_matrix,
        :BESS_cha_pur => zeros_matrix,
        :BESS_cha_sc => zeros_matrix,
        :BESS_dis => zeros_matrix,
        :BESS_COM => bess,
        :P_sc_EVj => zeros_matrix,
        :W_pur_heating => zeros_matrix,
        :W_pur_cooling => zeros_matrix,
        :W_sc_heating => zeros_matrix,
        :W_sc_cooling => zeros_matrix,
        :DHW_pur => zeros_matrix,
        :DHW_sc => zeros_matrix,
    )

    balance = FLEXECOM.calculate_energy_balance(results)
    @test length(balance) == 2
    @test all(isapprox.(balance, 0.0, atol=1e-8))
end
