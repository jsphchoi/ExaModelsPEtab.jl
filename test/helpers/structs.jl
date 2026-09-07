@testset "structs" begin
    @testset "$model" for model in MODELS
        PEinfo = peinfo(model)
        Nz = EMP._get_Nz(PEinfo)
        Nc = length(PEinfo.conditions)
        Nss = length(PEinfo.preeq_conditions)
        Nu = length(PEinfo.events)
        Ncv = size(PEinfo.cv0, 1)

        @test PEinfo isa EMP.PEtabInfo
        @test length(PEinfo.theta0) == count(parameter -> parameter.estimate, PEinfo.parameters)
        @test length(PEinfo.preeq_idxs) == Nc
        @test all(in(0:Nss), PEinfo.preeq_idxs)
        @test size(PEinfo.event_times) == (Nu, Nc)
        @test size(PEinfo.cv0) == (Ncv, Nc)
        @test length(PEinfo.zss0) == Nss
        @test all(zss -> length(zss) == Nz, PEinfo.zss0)
        @test all(measurement -> measurement.cidx in 1:Nc, PEinfo.measurements)
        @test all(measurement -> measurement.yidx in eachindex(PEinfo.observables), PEinfo.measurements)

        if isempty(PEinfo.nodes)
            @test PEinfo.K == 0
            @test isempty(PEinfo.z0)
        else
            N = length(PEinfo.nodes[1]) - 1
            @test length(PEinfo.nodes) == Nc
            @test all(nodes -> length(nodes) == N + 1, PEinfo.nodes)
            @test size(PEinfo.z0) == (Nz, Nc, N, PEinfo.K + 1)
        end
    end
end
