#
function _create_continuity(
    c::ExaCore,
    PEmodel::PEtabModel,
    PEprob::PEtabODEProblem,
    PEinfo::PEInfo
)
    _create_interval_continuity(c, PEmodel, PEprob, PEinfo)
    _create_initial_conditions(c, PEmodel, PEprob, PEinfo)
end