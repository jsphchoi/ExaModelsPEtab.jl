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
    "Alkan_SciSignal2018",
    "Borghans_BiophysChem1997",
    "Zheng_PNAS2012",
    "Elowitz_Nature2000",
    "Raia_CancerResearch2011",
    "Crauste_CellSystems2017",
    "Perelson_Science1996",
    "Sneyd_PNAS2002",
    "Rahman_MBS2016",
    "Okuonghae_ChaosSolitonsFractals2020",
    "Bachmann_MSB2011",
    "Armistead_CellDeathDis2024",
    "Laske_PLOSComputBiol2019",
    "Giordano_Nature2020",
    "Lang_PLOSComputBiol2024",
    "Chen_MSB2009",
    "SalazarCavazos_MBoC2020",
    "Isensee_JCB2018",
    "Raimundez_PCB2020",
    # Froehlich_CellSystems2018 : too large for the test suite
    # Fiedler_BMCSystBiol2016 : initial condition expression too large
]

const UNSUPPORTED = Dict(
    "Liu_IFACPapersOnLine2025" => "SBML event on a state",
    "Oliveira_NatCommun2021" => "estimated event time",
    "Beer_MolBioSystems2014" => "event time estimated in a condition",
)

function find_yaml(model)
    dir = joinpath(@__DIR__, "Benchmark-Models-PEtab", model)
    path = joinpath(dir, model * ".yaml")
    return isfile(path) ? path : only(filter(endswith(".yaml"), readdir(dir; join = true)))
end

const PEINFO = Dict{String, EMP.PEtabInfo}()
peinfo(model) = get!(() -> EMP._get_PEtabInfo(find_yaml(model)), PEINFO, model)

block(w, variable) = reshape(w[variable.offset .+ (1:variable.length)], EMP.ExaModels.size(variable.size)...)

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
