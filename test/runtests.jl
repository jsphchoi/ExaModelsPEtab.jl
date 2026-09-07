using ExaModelsPEtab
using MadNLP
using Test

const EMP = ExaModelsPEtab

include("backends.jl")

const MODELS = [
    "Boehm_JProteomeRes2014",
    "Bertozzi_PNAS2020",
    "Bruno_JExpBot2016",
    "Lucarelli_CellSystems2018",
    "Fujita_SciSignal2010",
    "Brannmark_JBC2010",
    "Blasi_CellSystems2016",
    "Weber_BMC2015",
    "Zhao_QuantBiol2020",
    "Schwen_PONE2014",
]

function find_yaml(model)
    dir = joinpath(@__DIR__, "Benchmark-Models-PEtab", model)
    path = joinpath(dir, model * ".yaml")
    return isfile(path) ? path : only(filter(endswith(".yaml"), readdir(dir; join = true)))
end

const PEINFO = Dict{String, EMP.PEtabInfo}()
peinfo(model) = get!(() -> EMP._get_PEtabInfo(find_yaml(model)), PEINFO, model)

function revise_model(model, table, edit)
    dir = mktempdir()
    cp(dirname(find_yaml(model)), dir; force = true)
    path = joinpath(dir, basename(EMP._read_yaml(find_yaml(model))[table]))
    write(path, edit(read(path, String)))
    return joinpath(dir, basename(find_yaml(model)))
end

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
