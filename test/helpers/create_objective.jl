@testset "create_objective" begin
    core_of(PEinfo) = isempty(PEinfo.nodes) ?
        EMP._create_constraints_steadystate(EMP._create_variables_steadystate(EMP.ExaModels.ExaCore(), PEinfo), PEinfo) :
        EMP._create_constraints(EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K), PEinfo), PEinfo)

    @testset "assembles the objective of $model" for model in MODELS
        PEinfo = peinfo(model)
        Nm = EMP._get_Nm(PEinfo)
        _, rows = EMP._get_sigmas(PEinfo, EMP._get_arguments(PEinfo), EMP._get_yvalue())
        core = core_of(PEinfo)
        ncon = core.ncon
        core = EMP._create_objective(core, PEinfo)
        Nsum, Nsigma = EMP._get_Nsum(core), maximum(rows)

        @test core.y.length == Nm
        @test (:sigma in propertynames(core)) == (Nsigma > 0)
        Nsigma == 0 || @test core.sigma.length == Nsigma
        @test core.ncon == ncon + Nsum + Nm + Nsigma

        nlp = EMP.ExaModels.ExaModel(core)
        c = similar(nlp.meta.x0, nlp.meta.ncon)
        EMP.ExaModels.NLPModels.cons!(nlp, nlp.meta.x0, c)
        @test maximum(abs, c[ncon+1:end]) <= 1e-8 * (1 + maximum(abs, block(nlp.meta.x0, core.y)))
        @test EMP.ExaModels.NLPModels.obj(nlp, nlp.meta.x0) ≈ EMP._evaluate_objective(PEinfo, PEinfo.theta0) rtol = 1e-10
    end

    @testset "the measurement points of $model" for model in MODELS
        PEinfo = peinfo(model)
        points = EMP._get_measurement_points(PEinfo)
        node(i, k) = k == 0 ? i : i + 1
        @test all(k in (0, PEinfo.K) for (i, k) in points)
        @test all(
            isinf(measurement.time) ? points[m] == (0, 0) : PEinfo.nodes[measurement.cidx][node(points[m]...)] == measurement.time
            for (m, measurement) in enumerate(PEinfo.measurements)
        )
    end

    @testset "binds the sums of an observable formula" begin
        EMP.Symbolics.@variables theta[1:2] z[1:10]
        unwrap, zsum = EMP.Symbolics.unwrap, EMP._get_zsum(1)
        bind(expr, sums) = unwrap(EMP._bind_sums(unwrap(expr), false, sums, (1, 1, 0), nothing, zsum))

        small = z[1] / (z[2] + z[3] + z[4])
        sums = []
        @test isequal(bind(small, sums), unwrap(small))
        @test isempty(sums)

        large = z[1] / sum(z[j] for j in 2:10)
        sums = []
        @test isequal(bind(large, sums), unwrap(z[1] / zsum[1]))
        @test length(sums) == 1 && isequal(sums[1][1], unwrap(sum(z[j] for j in 2:10)))

        top = sum(z[j] for j in 1:9) / (theta[1] + z[10])
        sums = []
        @test isequal(bind(top, sums), unwrap(top))
        @test isempty(sums)
        @test length(EMP._get_y_terms(unwrap(top))) == 9
    end

    @testset "the measurement constants" begin
        measurement = EMP.PEtabMeasurement(1.0, 4.0, Union{Float64, Int}[], Union{Float64, Int}[], 1, 1)
        @test EMP._get_ymeas(measurement, :lin) == 4.0
        @test EMP._get_ymeas(measurement, :log) == log(4.0)
        @test EMP._get_ymeas(measurement, :log10) == log10(4.0)
        @test EMP._get_constant(measurement, :lin) == 0.5 * log(2pi)
        @test EMP._get_constant(measurement, :log) == 0.5 * log(2pi) + log(4.0)
        @test EMP._get_constant(measurement, :log10) == 0.5 * log(2pi) + log(4.0) + log(log(10))
    end

    @testset "rejects a noise formula that reads a state outside its observable" begin
        yaml = revise_model("Boehm_JProteomeRes2014", :observables, content -> replace(content,
            "\tnoiseParameter1_pSTAT5A_rel\t" => "\tnoiseParameter1_pSTAT5A_rel * STAT5A\t"))
        PEinfo = EMP._get_PEtabInfo(yaml)
        @test_throws ArgumentError EMP._get_sigmas(PEinfo, EMP._get_arguments(PEinfo), EMP._get_yvalue())
    end
end
