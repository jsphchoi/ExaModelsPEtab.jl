@testset "create_variables_steadystate" begin
    @testset "creates the steady-state variables of $model" for model in filter(model -> EMP._has_zss(peinfo(model)), MODELS)
        PEinfo = peinfo(model)
        Ntheta, Nz, Nc = EMP._get_Ntheta(PEinfo), EMP._get_Nz(PEinfo), EMP._get_Nc(PEinfo)
        Ncv, Nss = EMP._get_Ncv(PEinfo), EMP._get_Nss(PEinfo)
        core = EMP._create_variables_steadystate(EMP.ExaModels.ExaCore(), PEinfo)

        @test core.nvar == Ntheta + Nz * Nss + (EMP._has_cv(PEinfo) ? Ncv * Nc : 0)
        @test EMP.ExaModels.size(core.theta.size) == (Ntheta,)
        @test EMP.ExaModels.size(core.zss.size) == (Nz, Nss)
        @test EMP._has_cv(PEinfo) == (:cv in propertynames(core))

        @test block(core.x0, core.theta) == PEinfo.theta0
        @test block(core.x0, core.zss) == reduce(hcat, PEinfo.zss0)
        @test all(isinf, block(core.lvar, core.zss)) && all(isinf, block(core.uvar, core.zss))
    end
end
