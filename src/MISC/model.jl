# SBML -> ModelingToolkit system via SBMLImporter

"""
    PEtabModelSys

Compiled symbolic model: the `mtkcompile`d MTK system, SBMLImporter callbacks, initial-value
and parameter maps, the piecewise event dict, and the independent variable.
"""
struct PEtabModelSys{S, C, E}
    sys::S
    callbacks::C
    speciemap::Dict{Any, Any}
    parametermap::Dict{Any, Any}
    events::E
    t_sym::Symbolics.Num
end

function _load_model(tables::PEtabTables)::PEtabModelSys
    _assert_supported_events(tables.sbml_file)
    reactionsys, callbacks = SBMLImporter.load_SBML(tables.sbml_file;
        ifelse_to_callback = true, inline_assignment_rules = false)
    odesys = SBMLImporter.Catalyst.ode_model(reactionsys)
    sys = MTK.mtkcompile(odesys)
    _assert_ode_only(sys)
    speciemap = Dict{Any, Any}(SBMLImporter.get_u0_map(reactionsys))
    parametermap = Dict{Any, Any}(SBMLImporter.get_parameter_map(reactionsys))
    # Re-parse for the event dict, the source of the static control trigger strings
    events = SBMLImporter.parse_SBML(tables.sbml_file, false; ifelse_to_callback = true,
        model_as_string = false, inline_assignment_rules = false).events
    t_sym = Symbolics.Num(only(MTK.independent_variables(sys)))
    return PEtabModelSys(sys, callbacks, speciemap, parametermap, events, t_sym)
end

# GUARD: only support fixed-time discrete events (variable 'x' changes to value 'y' at time 't')
# CATCHES: continuous or conditional events, e.g., Liu, Smith
function _assert_supported_events(sbml_file::String)
    isfile(sbml_file) || error("ExaModelsPEtab: SBML file not found: $sbml_file")
    nev = count(r"<event[ >]", read(sbml_file, String))
    nev == 0 && return nothing
    error("ExaModelsPEtab: unsupported SBML <event> ($nev in $(basename(sbml_file))); " *
          "only fixed-time piecewise(time) controls supported.")
end

# GUARD: pure ODE systems only, every equation differential
# CATCHES: DAE models (SBML algebraic rules)
function _assert_ode_only(sys)
    for eq in MTK.equations(sys)
        lhs = Symbolics.unwrap(eq.lhs)
        isdiff = Symbolics.iscall(lhs) && Symbolics.operation(lhs) isa Symbolics.Differential
        isdiff || error("ExaModelsPEtab: algebraic equation '$eq'. DAE models unsupported.")
    end
    return nothing
end
