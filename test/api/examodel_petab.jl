@testset "examodel_petab" begin
    @testset "builds $model" for model in MODELS_SOLVED
        PEinfo, nlp = peinfo(model), examodel(model)
        @test nlp.meta.ncon == nlp.meta.nvar - EMP._get_Ntheta(PEinfo)
        @test EMP.ExaModels.NLPModels.obj(nlp, nlp.meta.x0) ≈ EMP._evaluate_objective(PEinfo, PEinfo.theta0) rtol = 1e-10
    end

    @testset "builds $model on $backend" for model in ["Bertozzi_PNAS2020", "Blasi_CellSystems2016"], backend in BACKENDS
        nlp = examodel_petab(find_yaml(model); backend)
        @test EMP.ExaModels.NLPModels.obj(nlp, nlp.meta.x0) ≈ EMP.ExaModels.NLPModels.obj(examodel(model), examodel(model).meta.x0) rtol = 1e-10
    end

    @testset "rejects the collocation keywords" begin
        yaml = find_yaml("Boehm_JProteomeRes2014")
        @test_throws ArgumentError examodel_petab(yaml; roots = nothing)
        @test_throws ArgumentError examodel_petab(yaml; basis = nothing)
        @test_throws ArgumentError examodel_petab(yaml; polynomial = nothing)
        @test_throws ArgumentError examodel_petab(yaml; unknown_horizon = nothing)
    end

    @testset "detects steady-state measurements" begin
        @test EMP._is_steadystate(find_yaml("Blasi_CellSystems2016"))
        @test !EMP._is_steadystate(find_yaml("Bertozzi_PNAS2020"))

        function mixed_times(content)
            lines = filter(!isempty, split(content, '\n'))
            icol = EMP._get_index("time", split(lines[1], '\t'))
            cells = String.(split(lines[2], '\t'))
            cells[icol] = "0"
            return join([lines[1]; join(cells, '\t'); lines[3:end]], '\n')
        end
        @test_throws ArgumentError EMP._is_steadystate(revise_model("Blasi_CellSystems2016", :measurements, mixed_times))
    end
end
