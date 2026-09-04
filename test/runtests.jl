using ExaModelsPEtab
using MadNLP
using Test

include("backends.jl")

@testset "ExaModelsPEtab" begin
    @testset "helpers" begin
        for file in [
            "structs",
            "utils",
            "get_PEtabInfo",
            "create_variables",
            "create_constraints",
            "create_objective",
            "create_variables_steadystate",
            "create_constraints_steadystate"
        ]
            include("helpers/$file.jl")
        end
    end

    @testset "api" begin
        for file in [
            "examodel_petab"
        ]
            include("api/$file.jl")
        end
    end
end