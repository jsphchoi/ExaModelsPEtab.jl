using ExaModelsPEtab
using MadNLP
using Test

include("backends.jl")

@testset "ExaModelsPEtab" begin
    @testset "helpers" begin
        for file in [
            "",
            ""
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