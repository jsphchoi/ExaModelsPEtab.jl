@testset "utils" begin
    @testset "reads the yaml" begin
        files = EMP._read_yaml(find_yaml("Boehm_JProteomeRes2014"))
        @test keys(files) == (:parameters, :conditions, :measurements, :observables, :sbml)
        @test all(isfile, files)

        path = tempname()
        write(path, "format_version: 1\nparameter_file: parameters.tsv\n")
        @test_throws ArgumentError EMP._read_yaml(path)
    end

    @testset "reads a tsv" begin
        path = tempname()
        write(path, "﻿a\tb\tc\r\n1\t2\t3\r\n4\t5\r\n\r\n")
        table = EMP._read_tsv(path)
        @test keys(table) == (:a, :b, :c)
        @test table.a == ["1", "4"]
        @test table.c == ["3", ""]

        @test EMP._get_column(table, :b, "x") == ["2", "5"]
        @test EMP._get_column(table, :d, "x") == ["x", "x"]
    end

    @testset "resolves cells" begin
        @test EMP._get_index("b", ["a", "b"]) == 2
        @test_throws ArgumentError EMP._get_index("c", ["a", "b"])

        @test EMP._float("1.5") == 1.5
        @test isnan(EMP._float(""))
        @test isnan(EMP._float("abc"))

        parameters = [
            EMP.PEtabParameter("k1", true, 1.0, 0.1, 10.0, :log10, :none, Float64[]),
            EMP.PEtabParameter("k2", false, 2.0, 0.1, 10.0, :lin, :none, Float64[]),
            EMP.PEtabParameter("k3", true, 3.0, 0.1, 10.0, :log, :none, Float64[]),
        ]
        @test EMP._resolve_cell("0.5", parameters) === 0.5
        @test EMP._resolve_cell("k2", parameters) === 2.0
        @test EMP._resolve_cell("k1", parameters) === 1
        @test EMP._resolve_cell("k3", parameters) === 2
        @test_throws ArgumentError EMP._resolve_cell("k4", parameters)

        cells = EMP._resolve_cells("k1;0.5;k3", parameters)
        @test cells isa Vector{Union{Float64, Int}}
        @test cells == [1, 0.5, 2]
        @test isempty(EMP._resolve_cells("", parameters))
    end

    @testset "scales" begin
        @test EMP._linscale(2.0, :log10) == 100.0
        @test EMP._linscale(0.0, :log) == 1.0
        @test EMP._linscale(3.0, :lin) == 3.0
        @test EMP._logscale(100.0, :log10) == 2.0
        @test EMP._logscale(1.0, :log) == 0.0
        @test EMP._logscale(3.0, :lin) == 3.0
        for scale in (:lin, :log, :log10), x in (0.5, 2.0)
            @test EMP._logscale(EMP._linscale(x, scale), scale) ≈ x
        end
    end
end
