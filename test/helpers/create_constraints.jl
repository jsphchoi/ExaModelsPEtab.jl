@testset "create_constraints" begin
    mesh_models = filter(model -> !isempty(peinfo(model).nodes), MODELS)
    core_of(PEinfo) = EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K), PEinfo)
    function residual(core)
        model = EMP.ExaModels.ExaModel(core)
        c = similar(model.meta.x0, model.meta.ncon)
        EMP.ExaModels.NLPModels.cons!(model, model.meta.x0, c)
        return c
    end

    paths = Set{Symbol}()
    @testset "assembles the constraints of $model" for model in mesh_models
        PEinfo = peinfo(model)
        Nz, Nc = EMP._get_Nz(PEinfo), EMP._get_Nc(PEinfo)
        N, K = length(PEinfo.nodes[1]) - 1, PEinfo.K
        Ncv, Nss = EMP._get_Ncv(PEinfo), EMP._get_Nss(PEinfo)
        core = core_of(PEinfo)
        forms, _ = EMP._analyze_rhs(core, PEinfo)
        push!(paths, length(forms) + 1 < Nz ? :grouped : :perstate)
        core = EMP._create_constraints(core, PEinfo)

        collocation, continuity, ic, cv = Nz * Nc * N * K, Nz * Nc * (N - 1), Nz * Nc, Ncv * Nc
        @test core.ncon == collocation + continuity + ic + cv + Nz * Nss

        c, scale, from = residual(core), maximum(abs, PEinfo.z0), 0
        @test maximum(abs, c[from+1:from+collocation]) <= 1e-3 * scale
        from += collocation
        @test maximum(abs, c[from+1:from+continuity]) <= 1e-10 * scale
        from += continuity
        @test maximum(abs, c[from+1:from+ic]) <= 1e-8 * scale
        from += ic
        @test cv == 0 || all(iszero, c[from+1:from+cv])
    end
    @test paths == Set([:grouped, :perstate])

    @testset "the right-hand side of $model matches the ODE problem" for model in mesh_models
        PEinfo = peinfo(model)
        Nz, Nc = EMP._get_Nz(PEinfo), EMP._get_Nc(PEinfo)
        N = length(PEinfo.nodes[1]) - 1
        f, u = EMP._get_f(PEinfo), EMP._get_u(PEinfo)
        cvfixed = EMP._get_cvfixed(PEinfo, PEinfo.conditions)
        u_ids = EMP._get_u_ids(PEinfo)

        for cidx in 1:Nc, i in unique([1, cld(N, 2)])
            ssidx = PEinfo.preeq_idxs[cidx]
            op = EMP._get_op(PEinfo, PEinfo.theta0, PEinfo.conditions[cidx], ssidx == 0 ? nothing : PEinfo.zss0[ssidx])
            key = Dict(EMP._get_id(symbol) => symbol for symbol in keys(op))
            for (uidx, id) in enumerate(u_ids)
                op[key[id]] = u[uidx,cidx,i]
            end
            t, z = PEinfo.nodes[cidx][i], PEinfo.z0[:,cidx,i,1]
            prob = EMP.ODE.ODEProblem(PEinfo.model.sys, op, (0.0, 1.0); build_initializeprob = false)
            du = similar(z)
            prob.f(du, z, prob.p, t)
            ours = [f[v](PEinfo.theta0, z, PEinfo.cv0[:,cidx], cvfixed[:,cidx], u[:,cidx,i], t) for v in 1:Nz]
            @test ours ≈ du rtol = 1e-8
        end
    end

    @testset "the condition targets of $model" for model in mesh_models
        PEinfo = peinfo(model)
        Nc = EMP._get_Nc(PEinfo)
        conditions_table = EMP._read_tsv(EMP._read_yaml(find_yaml(model)).conditions)
        cv_ids, cvfixed_ids = EMP._get_cv_ids(PEinfo), EMP._get_cvfixed_ids(PEinfo)
        target_ids = unique(id for condition in [PEinfo.conditions; PEinfo.preeq_conditions] for id in condition.target_ids)

        @test sort([cv_ids; cvfixed_ids]) == sort(target_ids)
        @test isempty(intersect(cv_ids, cvfixed_ids))

        cvfixed = EMP._get_cvfixed(PEinfo, PEinfo.conditions)
        @test size(cvfixed) == (length(cvfixed_ids) + length(EMP._get_ifelses(PEinfo)), Nc)
        row(cidx) = EMP._get_index(PEinfo.conditions[cidx].condition_id, conditions_table.conditionId)
        @test all(
            cvfixed[cvfixedidx,cidx] == something(tryparse(Float64, conditions_table[Symbol(id)][row(cidx)]), cvfixed[cvfixedidx,cidx])
            for (cvfixedidx, id) in enumerate(cvfixed_ids), cidx in 1:Nc
        )
    end

    @testset "the event values of $model" for model in filter(model -> !isempty(peinfo(model).events), mesh_models)
        PEinfo = peinfo(model)
        Nc, N = EMP._get_Nc(PEinfo), length(peinfo(model).nodes[1]) - 1
        u, u_ids = EMP._get_u(PEinfo), EMP._get_u_ids(PEinfo)

        @test all(
            u[uidx,cidx,i] == u[uidx,cidx,i+1] || PEinfo.nodes[cidx][i+1] in PEinfo.event_times[:,cidx]
            for uidx in eachindex(u_ids), cidx in 1:Nc, i in 1:(N - 1)
        )
        @test all(
            u[uidx,cidx,1] == EMP._get_default(PEinfo.model, id)
            for (uidx, id) in enumerate(u_ids), cidx in 1:Nc if !any(iszero, PEinfo.event_times[:,cidx])
        )
    end

    @testset "groups the right-hand side terms" begin
        EMP.Symbolics.@variables theta[1:5] z[1:5] s
        unwrap = EMP.Symbolics.unwrap

        @test length(EMP._get_terms(z[1] * theta[1] + z[2])) == 2
        @test length(EMP._get_terms(z[1] * theta[1])) == 1
        @test EMP._get_terms(3.0) == [3.0]

        @test EMP._get_leaf(unwrap(theta[3])) == (:theta, 3)
        @test EMP._get_leaf(unwrap(z[2])) == (:z, 2)
        @test EMP._get_leaf(2.5) == (:data, 2.5)
        @test EMP._get_leaf(unwrap(s)) == (:t, 0)
        @test isnothing(EMP._get_leaf(unwrap(z[1] * theta[1])))

        @test EMP._get_form(unwrap(theta[1] * z[2])) == EMP._get_form(unwrap(theta[4] * z[5]))
        @test EMP._get_form(unwrap(theta[1] * z[2])) == EMP._get_form(unwrap(z[2] * theta[1]))
        @test EMP._get_form(unwrap(theta[1] * z[2])) != EMP._get_form(unwrap(theta[1] * theta[2]))

        @test EMP._get_children(unwrap(z[1]^3))[1] === (*)
        @test length(EMP._get_children(unwrap(z[1]^3))[2]) == 3
        @test EMP._get_children(unwrap(z[1]^5))[1] === (^)

        @test EMP._count_leaves(unwrap(z[1])) == 1
        @test EMP._count_leaves(unwrap(z[1] + z[2])) == 2
    end
end
