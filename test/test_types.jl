@testset "Types" begin
    # Test scenario creation
    scenario = FLEXECOM.Scenario(
        1,
        "Test Scenario",
        2,      # COEF
        5.0,    # PV
        1.0,    # GRID
        1.0,    # BESS_price
        10.0,   # BESS_cap
        2,      # EVnum
        3.0,    # ASHP
        1,      # DHW
    )
    
    @test scenario.id == 1
    @test scenario.name == "Test Scenario"
    @test scenario.COEF == 2
    @test scenario.PV == 5.0
    @test scenario.GRID == 1.0
    @test scenario.BESS_price == 1.0
    @test scenario.BESS_cap == 10.0
    @test scenario.EVnum == 2
    @test scenario.ASHP == 3.0
    @test scenario.DHW == 1
    
    # Test BESS specs
    bess = FLEXECOM.BESS_specs(
        3.552,  # cap
        74.0,   # capAh
        0.95,   # DoD
        0.9,    # Eff
        48.0,   # V
        3.552,  # Pmax
        1245.0, # Inv
        6000    # Cycles
    )
    
    @test bess.cap == 3.552
    @test bess.capAh == 74.0
    @test bess.DoD == 0.95
    @test bess.Eff == 0.9
    @test bess.V == 48.0
    @test bess.Pmax == 3.552
    @test bess.Inv == 1245.0
    @test bess.Cycles == 6000
    
    # Test EV specs
    ev = FLEXECOM.EV_specs(
        42.0,    # cap
        0.159,   # cons_km
        0.90,    # Eff
        7.4,     # Pch
        0.064,   # cons_fuel
        18000.0, # ICEV_price
        850.0    # RP_price
    )
    
    @test ev.cap == 42.0
    @test ev.cons_km == 0.159
    @test ev.Eff == 0.90
    @test ev.Pch == 7.4
    @test ev.cons_fuel == 0.064
    @test ev.ICEV_price == 18000.0
    @test ev.RP_price == 850.0

    # Test ASHP specs
    ashp = FLEXECOM.ASHP_specs(1.0, 890.0)
    @test ashp.Eff == 1.0
    @test ashp.price == 890.0

    # Test Financial params
    fin = FLEXECOM.Financial_params(
        0.02,   # d
        20,     # N
        650.0,  # PV_CAPEX
        10,     # INV_life
        0.07,   # fuel_price
        20.0,   # OM_PV
        0.02,   # OM_BESS
        0.02,   # OM_RP
        0.02,   # OM_ASHP
        0.01,   # OM_DHW
        131.0,  # DHW_price
        32.1    # PI_power
    )
    
    @test fin.d == 0.02
    @test fin.N == 20
    @test fin.PV_CAPEX == 650.0
    @test fin.INV_life == 10
    @test fin.fuel_price == 0.07
    @test fin.OM_PV == 20.0
    @test fin.OM_BESS == 0.02
    @test fin.OM_RP == 0.02
    @test fin.OM_ASHP == 0.02
    @test fin.OM_DHW == 0.01
    @test fin.DHW_price == 131.0
    @test fin.PI_power == 32.1
end
