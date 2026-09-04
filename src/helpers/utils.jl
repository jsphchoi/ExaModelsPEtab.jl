# Helper functions for parsing PEtab files and table cells

# Paths of the PEtab problem files listed in the yaml
function _read_yaml(filename)
    dir = dirname(abspath(filename))
    lines = [lstrip(strip(line), ['-', ' ']) for line in readlines(filename)]
    function file_of(key)
        i = findfirst(startswith(key), lines)
        isnothing(i) && throw(ArgumentError("$filename: missing $key"))
        value = strip(lines[i][length(key)+1:end])
        isempty(value) && (value = lines[i+1])
        return joinpath(dir, value)
    end
    return (
        parameters   = file_of("parameter_file:"),
        conditions   = file_of("condition_files:"),
        measurements = file_of("measurement_files:"),
        observables  = file_of("observable_files:"),
        sbml         = file_of("sbml_files:"),
    )
end

# Tab-separated table as a NamedTuple of String columns
function _read_tsv(path)
    lines = filter(!isempty, rstrip.(readlines(path), '\r'))
    header = Symbol.(split(lines[1], '\t'))
    rows = [split(line, '\t') for line in lines[2:end]]
    columns = [String[strip(get(row, j, "")) for row in rows] for j in eachindex(header)]
    return NamedTuple{Tuple(header)}(Tuple(columns))
end

# Column of a table or `default`
_get_column(table, key, default) = haskey(table, key) ? table[key] : fill(default, length(first(table)))

# Position of id in ids
function _get_index(id, ids)
    i = findfirst(==(id), ids)
    isnothing(i) && throw(ArgumentError("'$id' not found"))
    return i
end

# Number
_float(cell) = something(tryparse(Float64, cell), NaN)

# Table cell as a number, a theta index, or NaN (pre-equilibration value)
function _resolve_cell(cell, parameters)
    number = tryparse(Float64, cell)
    isnothing(number) || return number
    i = findfirst(parameter -> parameter.parameter_id == cell, parameters)
    isnothing(i) && throw(ArgumentError("'$cell' is neither a number nor a parameterId"))
    parameters[i].estimate || return parameters[i].value
    return count(parameter -> parameter.estimate, parameters[1:i])
end

# `;`-separated cells
_resolve_cells(cell, parameters) =
    Union{Float64, Int}[_resolve_cell(part, parameters) for part in split(cell, ';'; keepempty = false)]

# Value on the linear scale of a value on `scale`
_unscale(x, scale) = scale == :log10 ? exp10(x) : scale == :log ? exp(x) : x
