@testset "get_PEtabInfo" begin
    @testset "parses the tables of $model" for model in MODELS
        PEinfo = peinfo(model)
        files = EMP._read_yaml(find_yaml(model))
        parameters_table = EMP._read_tsv(files.parameters)
        observables_table = EMP._read_tsv(files.observables)
        measurements_table = EMP._read_tsv(files.measurements)
        Ntheta = EMP._get_Ntheta(PEinfo)

        @test [parameter.parameter_id for parameter in PEinfo.parameters] == parameters_table.parameterId
        @test [parameter.estimate for parameter in PEinfo.parameters] == (parameters_table.estimate .== "1")
        @test [parameter.value for parameter in PEinfo.parameters] == parse.(Float64, parameters_table.nominalValue)
        @test [parameter.scale for parameter in PEinfo.parameters] == Symbol.(parameters_table.parameterScale)

        @test [observable.observable_id for observable in PEinfo.observables] == observables_table.observableId
        @test [observable.observable_formula for observable in PEinfo.observables] == observables_table.observableFormula
        @test all(observable -> observable.transform in (:lin, :log, :log10), PEinfo.observables)

        @test [condition.condition_id for condition in PEinfo.conditions] == unique(measurements_table.simulationConditionId)
        @test all(
            cell isa Float64 || cell in 1:Ntheta
            for condition in [PEinfo.conditions; PEinfo.preeq_conditions] for cell in condition.target_values
        )

        @test [measurement.time for measurement in PEinfo.measurements] == parse.(Float64, measurements_table.time)
        @test [measurement.measurement for measurement in PEinfo.measurements] == parse.(Float64, measurements_table.measurement)
        @test [PEinfo.conditions[measurement.cidx].condition_id for measurement in PEinfo.measurements] == measurements_table.simulationConditionId
        @test [PEinfo.observables[measurement.yidx].observable_id for measurement in PEinfo.measurements] == measurements_table.observableId

        preeq_column = EMP._get_column(measurements_table, :preequilibrationConditionId, "")
        preeq_ids = [condition.condition_id for condition in PEinfo.preeq_conditions]
        if all(measurement -> isinf(measurement.time), PEinfo.measurements)
            @test preeq_ids == [condition.condition_id for condition in PEinfo.conditions]
            @test PEinfo.preeq_idxs == eachindex(PEinfo.conditions)
        else
            @test preeq_ids == filter(!isempty, unique(preeq_column))
            @test [ssidx == 0 ? "" : preeq_ids[ssidx] for ssidx in PEinfo.preeq_idxs] == [
                only(unique(preeq_column[measurements_table.simulationConditionId .== condition.condition_id]))
                for condition in PEinfo.conditions
            ]
        end
    end

    @testset "solves $model at theta0" for model in MODELS
        PEinfo = peinfo(model)
        sys = PEinfo.model.sys
        estimated = [parameter for parameter in PEinfo.parameters if parameter.estimate]

        @test !any(equation -> isequal(EMP.Symbolics.value(equation.rhs), 0), EMP.MTK.equations(sys))
        @test length(EMP.MTK.equations(sys)) == length(EMP.MTK.unknowns(sys))
        species_ids = [EMP._get_id(symbol) for (symbol, value) in PEinfo.model.speciemap]
        @test all(state -> EMP._get_id(state) in species_ids, EMP.MTK.unknowns(sys))

        @test all(isfinite, PEinfo.theta0)
        @test all(
            estimated[j].value == 0 || EMP._linscale(PEinfo.theta0[j], estimated[j].scale) ≈ estimated[j].value
            for j in eachindex(estimated)
        )

        condition = PEinfo.conditions[1]
        op = EMP._get_op(PEinfo, PEinfo.theta0, condition, nothing)
        key = Dict(EMP._get_id(symbol) => symbol for symbol in keys(op))
        scales = [parameter.scale for parameter in estimated]
        @test all(
            cell isa Int ?
                op[key[id]] == EMP._linscale(PEinfo.theta0[cell], scales[cell]) :
                isnan(cell) || op[key[id]] == cell
            for (id, cell) in zip(condition.target_ids, condition.target_values)
        )

        @test all(isfinite, PEinfo.z0)
        @test all(all(isfinite, zss) for zss in PEinfo.zss0)
    end

    @testset "meshes $model" for model in MODELS
        PEinfo = peinfo(model)
        if isempty(PEinfo.nodes)
            @test PEinfo.K == 0
        else
            @test PEinfo.K in (3, 4)
            @test all(nodes -> nodes[1] == 0 && issorted(nodes) && allunique(nodes), PEinfo.nodes)

            t_stops = EMP._get_t_stops(PEinfo)
            @test all(all(in(PEinfo.nodes[cidx]), t_stops[cidx]) for cidx in eachindex(t_stops))
            @test all(measurement.time in t_stops[measurement.cidx] for measurement in PEinfo.measurements)

            K = PEinfo.K
            @test all(
                PEinfo.z0[:,cidx,i,K+1] ≈ PEinfo.z0[:,cidx,i+1,1]
                for cidx in axes(PEinfo.z0, 2), i in 1:(size(PEinfo.z0, 3) - 1)
            )

            state_ids = [EMP._get_id(state) for state in EMP.MTK.unknowns(PEinfo.model.sys)]
            for (cidx, ssidx) in enumerate(PEinfo.preeq_idxs)
                ssidx == 0 && continue
                carried = [v for v in eachindex(state_ids) if !(state_ids[v] in PEinfo.conditions[cidx].target_ids)]
                @test PEinfo.z0[carried,cidx,1,1] ≈ PEinfo.zss0[ssidx][carried]
            end
        end
    end

    @testset "places the mesh" begin
        condition(id) = EMP.PEtabCondition(id, String[], Union{Float64, Int}[])
        measurement(time, cidx) = EMP.PEtabMeasurement(time, 0.0, Union{Float64, Int}[], Union{Float64, Int}[], cidx, 1)
        petab = (
            conditions = [condition("c1"), condition("c2")],
            measurements = [measurement(0.0, 1), measurement(1.0, 1), measurement(4.0, 1), measurement(2.0, 1), measurement(0.0, 2)],
            event_times = [3.0 0.0; 10.0 0.0],
        )

        t_stops = EMP._get_t_stops(petab)
        @test t_stops[1] == [0.0, 1.0, 2.0, 3.0, 4.0]
        @test t_stops[2] == [0.0, 4.0]

        t = collect(0.0:0.25:1.0)
        @test EMP._get_nodes(t, [0.0, 1.0], 1) == t
        @test EMP._get_nodes(t, [0.0, 1.0], 2) == [0.0, 0.5, 1.0]
        @test EMP._get_nodes(t, [0.0, 0.5, 1.0], 4) == [0.0, 0.5, 1.0]
        @test EMP._split_widest!([0.0, 1.0, 3.0], 4) == [0.0, 0.5, 1.0, 2.0, 3.0]

        sols = [(; t = collect(0.0:0.25:4.0)), (; t = [0.0, 4.0])]
        for (mesh_size, expected) in ((:small, (8, 4)), (:medium, (4, 4)), (:large, (4, 3)), (:massive, (4, 3)))
            nodes, K = EMP._determine_mesh(petab, sols, mesh_size)
            @test (length(nodes[1]) - 1, K) == expected
            @test all(node -> length(node) == length(nodes[1]), nodes)
        end
        @test EMP._determine_mesh(petab, nothing, :small) == (Vector{Float64}[], 0)
        @test EMP._get_mesh_size(petab, sols, 1) == :small
        @test EMP._get_mesh_size(petab, sols, 5000) == :medium
        @test EMP._get_mesh_size(petab, sols, 20000) == :large
        @test EMP._get_mesh_size(petab, sols, 100000) == :massive
        @test EMP._get_mesh_size(petab, nothing, 1) == :small
        @test isempty(EMP._get_z0(nothing, Vector{Float64}[], 0))

        @test EMP._get_odesolver(:small) isa EMP.ODE.Rodas5P
        @test EMP._get_odesolver(:medium) isa EMP.ODE.FBDF
        @test EMP._get_odesolver(:large) isa EMP.ODE.FBDF
    end

    @testset "reads events" begin
        @test EMP._get_id(:x) == "x"
        @test EMP._get_id("x(t)") == "x"

        @test EMP._split_event_formula("t >= 5") == (:>=, 5)
        @test EMP._split_event_formula("5 <= t") == (:>=, 5)
        @test EMP._split_event_formula("-(t, c) >= 0") == (:>=, :c)
        @test_throws ArgumentError EMP._split_event_formula("x > 5")

        condition = EMP.PEtabCondition("c1", ["a", "b"], Union{Float64, Int}[2.0, 1])
        parameters = [
            EMP.PEtabParameter("p", false, 7.0, 0.1, 10.0, :lin, :none, Float64[]),
            EMP.PEtabParameter("q", true, 1.0, 0.1, 10.0, :lin, :none, Float64[]),
        ]
        parametermap = [:r => 3.0]
        evaluate(expression) = EMP._evaluate_event(expression, "t >= x", condition, parameters, parametermap)
        @test evaluate(5) == 5.0
        @test evaluate(:a) == 2.0
        @test evaluate(:p) == 7.0
        @test evaluate(:r) == 3.0
        @test evaluate(:(a + r)) == 5.0
        @test evaluate(:(2 * a - 1)) == 3.0
        @test_throws ArgumentError evaluate(:b)
        @test_throws ArgumentError evaluate(:q)
        @test_throws ArgumentError evaluate(:s)
        @test_throws ArgumentError evaluate(:(a^2))

        events = [EMP.PEtabEvent(["u1"], ["1.0"], "t >= 1"), EMP.PEtabEvent(["u1"], ["2.0"], "t >= 5")]
        petab = (; events)
        @test EMP._get_event_times(events, [condition], parameters, parametermap) == reshape([1.0, 5.0], 2, 1)
        @test EMP._get_u_ids(petab) == ["u1"]

        times = [1.0, 5.0]
        @test EMP._get_u_value(petab, "u1", times, 0.0) === nothing
        @test EMP._get_u_value(petab, "u1", times, 1.0) == 1.0
        @test EMP._get_u_value(petab, "u1", times, 3.0) == 1.0
        @test EMP._get_u_value(petab, "u1", times, 5.0) == 2.0
        @test EMP._get_u_value(petab, "u2", times, 5.0) === nothing

        model = EMP.PEtabModel(nothing, [:x => 1.0], [:p => 2.0], nothing)
        @test EMP._get_default(model, "x") == 1.0
        @test EMP._get_default(model, "p") == 2.0
        @test_throws ArgumentError EMP._get_default(model, "q")
    end

    @testset "resolves cv" begin
        parameters = [
            EMP.PEtabParameter("p", false, 7.0, 0.1, 10.0, :lin, :none, Float64[]),
            EMP.PEtabParameter("q", true, 1.0, 0.1, 10.0, :log10, :none, Float64[]),
        ]
        conditions = [
            EMP.PEtabCondition("c1", ["a", "b"], Union{Float64, Int}[1, 2.0]),
            EMP.PEtabCondition("c2", ["a", "b"], Union{Float64, Int}[3.0, 2.0]),
        ]
        petab = (; parameters, conditions, preeq_conditions = EMP.PEtabCondition[])
        @test EMP._get_cv_ids(petab) == ["a"]
        @test EMP._get_cv0(petab) == [1.0 3.0]

        preeq = [EMP.PEtabCondition("s1", ["a"], Union{Float64, Int}[1])]
        @test_throws ArgumentError EMP._get_cv_ids((; parameters, conditions, preeq_conditions = preeq))

        unset = [conditions[1], EMP.PEtabCondition("c2", ["b"], Union{Float64, Int}[2.0])]
        @test_throws ArgumentError EMP._get_cv0((; parameters, conditions = unset, preeq_conditions = EMP.PEtabCondition[]))
    end

    @testset "rejects what it cannot parse" begin
        @test EMP._get_preeq_idx("c1", ["c1", "c1"], ["s1", "s1"], ["s1"]) == 1
        @test EMP._get_preeq_idx("c1", ["c1"], [""], String[]) == 0
        @test_throws ArgumentError EMP._get_preeq_idx("c1", ["c1", "c1"], ["s1", "s2"], ["s1", "s2"])

        @test_throws ArgumentError EMP._get_PEtabInfo(
            revise_model("Boehm_JProteomeRes2014", :observables, content -> replace(content, "\tnormal" => "\tlaplace"))
        )

        function inf_times(content)
            lines = filter(!isempty, split(content, '\n'))
            icol = EMP._get_index("time", split(lines[1], '\t'))
            rows = map(lines[2:end]) do line
                cells = String.(split(line, '\t'))
                cells[icol] = "inf"
                join(cells, '\t')
            end
            return join([lines[1]; rows], '\n')
        end
        @test_throws ArgumentError EMP._get_PEtabInfo(revise_model("Brannmark_JBC2010", :measurements, inf_times))
    end

    @testset "rejects $model, $reason" for (model, reason) in UNSUPPORTED
        @test_throws ArgumentError EMP._get_PEtabInfo(find_yaml(model))
    end
end
