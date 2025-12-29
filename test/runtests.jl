using Test
using FLEXECOM

@testset "FLEXECOM Tests" begin
    include("test_types.jl")
    include("test_data_loader.jl")
    include("test_optimizer.jl")
    include("test_financial.jl")
    include("test_utils.jl")
end
