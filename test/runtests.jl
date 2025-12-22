using Pkg

project_root = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_root)
Pkg.instantiate()

push!(LOAD_PATH, joinpath(project_root, "src"))

using Test
using FLEXECOM

@testset "FLEXECOM Tests" begin
    include("test_types.jl")
    include("test_data_loader.jl")
    include("test_optimizer.jl")
    include("test_financial.jl")
    include("test_utils.jl")
end
