@testset "create_constraints_steadystate" begin
    function residual(core)
        nlp = EMP.ExaModels.ExaModel(core)
        c = similar(nlp.meta.x0, nlp.meta.ncon)
        EMP.ExaModels.NLPModels.cons!(nlp, nlp.meta.x0, c)
        return c
    end

    @testset "the steady-state constraints of $model" for model in filter(model -> EMP._has_zss(peinfo(model)), MODELS)
        PEinfo = peinfo(model)
        Nz, Nss = EMP._get_Nz(PEinfo), EMP._get_Nss(PEinfo)
        cvfixed = EMP._get_cvfixed(PEinfo, PEinfo.preeq_conditions)
        times = EMP._get_event_times(PEinfo.events, PEinfo.preeq_conditions, PEinfo.parameters, PEinfo.model.parametermap)
        u = [
            something(EMP._get_u_value(PEinfo, id, times[:,ssidx], Inf), EMP._get_default(PEinfo.model, id))
            for id in EMP._get_u_ids(PEinfo), ssidx in 1:Nss
        ]
        W, b, keep_rows = EMP._get_conservation_laws(PEinfo, cvfixed, u)
        dfdz, f = EMP._get_dfdz(PEinfo), EMP._get_f(PEinfo)

        @test length(W) == Nss && length(b) == Nss && length(keep_rows) == Nss
        for ssidx in 1:Nss
            zss = PEinfo.zss0[ssidx]
            @test size(W[ssidx]) == (length(b[ssidx]), Nz)
            @test length(keep_rows[ssidx]) + length(b[ssidx]) == Nz
            @test W[ssidx] * zss ≈ b[ssidx]

            J = dfdz(PEinfo.theta0, zss, Float64[], cvfixed[:,ssidx], u[:,ssidx], 0.0)
            @test maximum(abs, W[ssidx] * J; init = 0.0) <= 1e-6 * maximum(abs, J)

            shifted(v, h) = (z = copy(zss); z[v] += h; [f[w](PEinfo.theta0, z, (), cvfixed[:,ssidx], u[:,ssidx], 0.0) for w in 1:Nz])
            differences = reduce(hcat, [
                (h = 1e-6 * max(abs(zss[v]), 1.0); (shifted(v, h) - shifted(v, -h)) / (2h))
                for v in 1:Nz
            ])
            @test maximum(abs, J - differences) <= 1e-4 * maximum(abs, J)
        end

        core = isempty(PEinfo.nodes) ?
            EMP._create_constraints_steadystate(EMP._create_variables_steadystate(EMP.ExaModels.ExaCore(), PEinfo), PEinfo) :
            EMP._create_zss_constraints(EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K), PEinfo), PEinfo)
        @test core.ncon == Nz * Nss
        @test maximum(abs, residual(core)) <= 1e-4 * (1 + maximum(maximum(abs, zss) for zss in PEinfo.zss0))
    end

    @testset "rejects a condition variable with pre-equilibration" begin
        yaml = revise_model("Brannmark_JBC2010", :conditions, content -> replace(content,
            "Dose_001\tinsulin, 0.01 nM\t0.000\t0.010" => "Dose_001\tinsulin, 0.01 nM\t0.000\tk1a"))
        PEinfo = EMP._get_PEtabInfo(yaml)
        @test EMP._has_cv(PEinfo) && EMP._has_zss(PEinfo)
        core = EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K), PEinfo)
        @test_throws ArgumentError EMP._create_zss_constraints(core, PEinfo)
    end
end
