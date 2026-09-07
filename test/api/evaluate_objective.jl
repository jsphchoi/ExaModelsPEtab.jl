@testset "evaluate_objective" begin
    @testset "evaluates $model at x0" for model in MODELS_SOLVED
        nlp = examodel(model)
        @test evaluate_objective(find_yaml(model), block(nlp.meta.x0, nlp.theta)) ≈ EMP.ExaModels.NLPModels.obj(nlp, nlp.meta.x0) rtol = 1e-10
    end

    @testset "evaluates $model at the collocation optimum" for model in MODELS_SOLVED
        nlp = examodel(model)
        result = madnlp(nlp;
            tol = 1e-6, acceptable_tol = 1e-4, acceptable_iter = 15,
            kkt_system = MadNLP.SparseCondensedKKTSystem, 
            equality_treatment = MadNLP.RelaxEquality, 
            fixed_variable_treatment = MadNLP.RelaxBound,
            print_level = MadNLP.ERROR,
        )
        @test result.status in (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
        @test evaluate_objective(find_yaml(model), EMP.ExaModels.solution(result, nlp.theta)) ≈ result.objective rtol = 1e-3
    end

    @testset "the negative log prior" begin
        prior(kind, parameters, scale = :lin) = EMP._get_prior(EMP.PEtabParameter("p", true, 1.0, 0.0, 10.0, scale, kind, parameters), 2.0)
        @test prior(:none, Float64[]) == 0.0
        @test prior(:parameterScaleNormal, [1.0, 0.5]) == 0.5 * ((2.0 - 1.0) / 0.5)^2 + log(0.5) + 0.5 * log(2pi)
        @test prior(:parameterScaleLaplace, [1.0, 0.5]) == abs(2.0 - 1.0) / 0.5 + log(2 * 0.5)
        @test prior(:normal, [1.0, 0.5], :log10) == 0.5 * ((100.0 - 1.0) / 0.5)^2 + log(0.5) + 0.5 * log(2pi)
        @test prior(:laplace, [1.0, 0.5], :log10) == abs(100.0 - 1.0) / 0.5 + log(2 * 0.5)
        @test prior(:uniform, [0.0, 10.0]) == log(10.0)
        @test_throws ArgumentError prior(:cauchy, [1.0, 0.5])
    end
end
