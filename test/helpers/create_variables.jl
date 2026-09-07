@testset "create_variables" begin
    @testset "creates the collocation variables of $model" for model in filter(model -> !isempty(peinfo(model).nodes), MODELS)
        PEinfo = peinfo(model)
        Ntheta, Nz, Nc = EMP._get_Ntheta(PEinfo), EMP._get_Nz(PEinfo), EMP._get_Nc(PEinfo)
        Ncv, Nss = EMP._get_Ncv(PEinfo), EMP._get_Nss(PEinfo)
        N, K = length(PEinfo.nodes[1]) - 1, PEinfo.K
        core = EMP._create_variables(EMP.EMC.CollocationExaCore(PEinfo.nodes, K), PEinfo)

        @test core.nvar == Ntheta + Nz * Nc * N * (K + 1) + Ncv * Nc + Nz * Nss
        @test EMP.ExaModels.size(core.theta.size) == (Ntheta,)
        @test EMP.ExaModels.size(core.z.size) == (Nz, Nc, N, K + 1)
        @test EMP.ExaModels.size(core.cv.size) == (Ncv, Nc)
        @test core.theta.offset == 0
        @test core.z.offset == Ntheta
        @test core.cv.offset == Ntheta + Nz * Nc * N * (K + 1)

        @test block(core.x0, core.theta) == PEinfo.theta0
        @test block(core.x0, core.z) == PEinfo.z0
        @test block(core.x0, core.cv) == PEinfo.cv0

        parameters_table = EMP._read_tsv(EMP._read_yaml(find_yaml(model)).parameters)
        rows = findall(parameters_table.estimate .== "1")
        scale(i) = Symbol(parameters_table.parameterScale[i])
        @test block(core.lvar, core.theta) ==
            [EMP._logscale(EMP._float(parameters_table.lowerBound[i]), scale(i)) for i in rows]
        @test block(core.uvar, core.theta) ==
            [EMP._logscale(EMP._float(parameters_table.upperBound[i]), scale(i)) for i in rows]
        @test all(
            EMP._linscale(PEinfo.theta0[j], scale(i)) ≈ parse(Float64, parameters_table.nominalValue[i])
            for (j, i) in enumerate(rows) if parameters_table.nominalValue[i] != "0"
        )

        @test all(isinf, block(core.lvar, core.z)) && all(<(0), block(core.lvar, core.z))
        @test all(isinf, block(core.uvar, core.z)) && all(>(0), block(core.uvar, core.z))
        @test all(isinf, block(core.lvar, core.cv)) && all(isinf, block(core.uvar, core.cv))

        if Nss > 0
            @test EMP.ExaModels.size(core.zss.size) == (Nz, Nss)
            @test core.zss.offset == Ntheta + Nz * Nc * N * (K + 1) + Ncv * Nc
            @test block(core.x0, core.zss) == reduce(hcat, PEinfo.zss0)
            @test all(isinf, block(core.lvar, core.zss)) && all(isinf, block(core.uvar, core.zss))
        else
            @test !(:zss in propertynames(core))
        end
    end
end
