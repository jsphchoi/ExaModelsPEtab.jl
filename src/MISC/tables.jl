# PEtab yaml + TSV parsing into PEtabTables

"""
    PEtabTables(yaml_path)

Raw PEtab problem files: the four TSV tables as NamedTuples of `String` columns, plus paths.
"""
struct PEtabTables
    dir::String
    sbml_file::String
    parameters::NamedTuple
    conditions::NamedTuple
    measurements::NamedTuple
    observables::NamedTuple
end

function PEtabTables(yaml_path::String)
    files = _read_petab_yaml(yaml_path)
    return PEtabTables(
        dirname(abspath(yaml_path)),
        files.sbml,
        _read_tsv(files.parameters),
        _read_tsv(files.conditions),
        _read_tsv(files.measurements),
        _read_tsv(files.observables),
    )
end

# Read a tab-separated table into a NamedTuple of String columns
function _read_tsv(path::String)
    lines = [rstrip(line, '\r') for line in readlines(path)]
    filter!(!isempty, lines)
    isempty(lines) && error("$path: empty table")
    header = Symbol.(split(lines[1], '\t'))
    ncol = length(header)
    cols = [String[] for _ in 1:ncol]
    for line in lines[2:end]
        cells = split(line, '\t'; keepempty = true)
        length(cells) > ncol && error("$path: row has $(length(cells)) cells, header has $ncol")
        for j in 1:ncol
            push!(cols[j], j <= length(cells) ? String(cells[j]) : "")
        end
    end
    return NamedTuple{Tuple(header)}(Tuple(cols))
end

# Read the stereotyped petab.yaml shape: one problem, one file per list
function _read_petab_yaml(path::String)
    dir = dirname(abspath(path))
    scalars = Dict{String, String}()
    lists = Dict{String, Vector{String}}()
    key = ""
    nproblems = 0
    for raw in readlines(path)
        s = strip(rstrip(raw, '\r'))
        isempty(s) && continue
        if startswith(s, "-")
            item = _unquote(strip(s[2:end]))
            if endswith(item, ':')
                nproblems += 1
                nproblems > 1 && error("$path: only one PEtab problem entry is supported")
                key = item[1:end-1]
            else
                push!(get!(lists, key, String[]), item)
            end
        else
            sep = findfirst(':', s)
            sep === nothing && error("$path: unrecognized yaml line: '$s'")
            k = String(strip(s[1:sep-1]))
            v = _unquote(strip(s[sep+1:end]))
            isempty(v) ? (key = k) : (scalars[k] = v)
        end
    end
    version = get(scalars, "format_version", "")
    version in ("1", "1.0.0") || error("$path: unsupported PEtab format_version '$version'")
    file_of(k) = begin
        entries = haskey(scalars, k) ? [scalars[k]] : get(lists, k, String[])
        length(entries) == 1 || error("$path: expected exactly one $k, got $(length(entries))")
        joinpath(dir, entries[1])
    end
    return (
        parameters   = file_of("parameter_file"),
        conditions   = file_of("condition_files"),
        measurements = file_of("measurement_files"),
        observables  = file_of("observable_files"),
        sbml         = file_of("sbml_files"),
    )
end

_unquote(s::AbstractString) = String(strip(s, ['"', '\'']))

# Column helpers over the NamedTuple tables
_nrows(table::NamedTuple) = isempty(table) ? 0 : length(first(table))
_getcol(table::NamedTuple, name::Symbol, default::String = "") =
    haskey(table, name) ? getproperty(table, name) : fill(default, _nrows(table))

function _cellfloat(cell::AbstractString, context::String)
    x = tryparse(Float64, cell)
    x === nothing && error("$context: not a number: '$cell'")
    return x
end
