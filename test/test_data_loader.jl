@testset "DataLoader" begin
    # Create a temporary directory with test data
    test_dir = mktempdir()
    
    # Create a test scenario file
    scenario_file = joinpath(test_dir, "test_scenarios.csv")
    open(scenario_file, "w") do f
        write(f, "COEF;PV;GRID;BESS_price;BESS_cap;EVnum;ASHP;DHW;PV_price;Fuel_price;NG_price;EV_price\n")
        write(f, "1;5.0;1.0;1.0;10.0;2;3.0;1;0.95;1.4;0.07;2500\n")
        write(f, "2;3.0;0.9;1.2;5.0;1;2.0;0;1.05;1.8;0.09;1800\n")
    end
    
    # Test loading scenarios
    scenarios = FLEXECOM.load_scenarios(scenario_file)
    
    @test length(scenarios) == 2
    @test scenarios[1].COEF == 1
    @test scenarios[1].PV == 5.0
    @test scenarios[1].GRID == 1.0
    @test scenarios[1].BESS_price == 1.0
    @test scenarios[1].BESS_cap == 10.0
    @test scenarios[1].EVnum == 2
    @test scenarios[1].ASHP == 3.0
    @test scenarios[1].DHW == 1

    @test scenarios[2].COEF == 2
    @test scenarios[2].PV == 3.0
    @test scenarios[2].GRID == 0.9
    @test scenarios[2].BESS_price == 1.2
    @test scenarios[2].BESS_cap == 5.0
    @test scenarios[2].EVnum == 1
    @test scenarios[2].ASHP == 2.0
    @test scenarios[2].DHW == 0

    # Clean up
    rm(test_dir, recursive=true)
end

@testset "DataLoader - multi-column" begin
    test_dir = mktempdir()
    scenario_file = joinpath(test_dir, "scenarios_multi.csv")

    open(scenario_file, "w") do f
        write(f, "COEF,PV,GRID,BESS_price,BESS_cap,EVnum,ASHP,DHW\n")
        write(f, "3,4.0,1.1,0.8,8.0,0,0.0,1\n")
    end

    scenarios = FLEXECOM.load_scenarios(scenario_file)

    @test length(scenarios) == 1
    @test scenarios[1].COEF == 3
    @test scenarios[1].PV == 4.0
    @test scenarios[1].GRID == 1.1
    @test scenarios[1].BESS_price == 0.8
    @test scenarios[1].BESS_cap == 8.0
    @test scenarios[1].EVnum == 0
    @test scenarios[1].ASHP == 0.0
    @test scenarios[1].DHW == 1

    rm(test_dir, recursive=true)
end
