#######################################################
# STATE AS OF: 05/30/26
# TODO: support ev: _create_ev, intervene interval continuity. idea: create dict i=1:N -> f(t_event) to continuity constraint!
#######################################################

# (*) Main function for creating collocation continuity constraints (*)
function _create_continuity(
    c::ExaCore,
    PEmodel::PEtabModel,
    PEprob::PEtabODEProblem,
    PEinfo::PEInfo
)
    c = _create_interval_continuity(c, PEmodel, PEprob, PEinfo)
    c = _create_initial_conditions(c, PEmodel, PEprob, PEinfo)
    return c
end

# Create cross-interval continuity constraints
function _create_interval_continuity(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    ###############################################################
    # Unpack problem info
    ###############################################################
    (; Nz, N, Nc, L1, K) = PEinfo
    z = c.z

    ###############################################################
    # Interval continuity equations
    ###############################################################
    itr_cont1 = [(v,i,cidx) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc]
    con_interval = ExaModels.@add_con(c,
        -z[v,i+1,0,cidx]
        for (v,i,cidx) in itr_cont1
    )
    itr_cont1! = [(v,i,cidx,j,L1[j+1]) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc, j in 0:K]
    ExaModels.@add_con!(c,
        con_interval,
        (v,i,cidx) => L1j*z[v,i,j,cidx]
        for (v,i,cidx,j,L1j) in itr_cont1!
    )
    return c
end

# Create initial condition continuity constraints
function _create_initial_conditions(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    ###############################################################
    # Unpack problem info
    ###############################################################
    (; Nz, Nc, Np, Ncv, pscale) = PEinfo
    z = c.z
    p = c.p
    if Ncv >= 1
        cv = c.cv
    end

    # Check which type of initial condition
    if _check_x0SSpre(PEprob)
        ###############################################################
        # Initial condition equations: steady-state pre-equilibration 
        ###############################################################
        # if x0SSpre(p)...
        
        # Unpack steady state variable
        zss = c.zss

        # RHS functions for the steady-state residual (same funcs the collocation uses)
        fs, has_t = _get_rhs_funcs(PEmodel, PEprob)

        # Constraint 1: simulation initial condition = pre-equilibration steady state.
        # zss is indexed by simulation condition cidx (zss[:, cidx]).
        itr_ss1 = [(v,cidx) for v in 1:Nz, cidx in 1:Nc]
        ExaModels.@add_con(c,
            # z[:,1,0,cidx] = zss[:,cidx]
            z[v,1,0,cidx] - zss[v,cidx]
            for (v,cidx) in itr_ss1
        )

        # Constraint 2: steady-state residual f(zss[:,cidx]) = 0 evaluated under the
        # PRE-EQUILIBRATION condition's inputs cv[:, sscidx], where sscidx is the
        # canonical index of cidx's pre-equilibration condition.
        dict_cidx_sscidx = _get_dict_cidx_sscidx(PEmodel, PEprob)
        itr_ss2 = [(cidx, dict_cidx_sscidx[cidx]) for cidx in 1:Nc]
        if has_t
            # Pre-equilibration is autonomous; evaluate any explicit time-dependence at t=0.
            for f in fs
                ExaModels.@add_con(c,
                    f(
                        ntuple(v -> zss[v,cidx], Nz)...,
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,sscidx], Ncv)...,
                        0.0
                    )
                    for (cidx,sscidx) in itr_ss2
                )
            end
        else
            for f in fs
                ExaModels.@add_con(c,
                    f(
                        ntuple(v -> zss[v,cidx], Nz)...,
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,sscidx], Ncv)...
                    )
                    for (cidx,sscidx) in itr_ss2
                )
            end
        end

    else
        ###############################################################
        # Initial condition equations
        ###############################################################
        # if x0fix, x0 = p, x0 = f(p)...

        # Obtain initial condition symbolic expressions
        dict_z0sym_expr = Dict(PEprob.model_info.model.speciemap) # get mapping of initial condition: symbolic state variable => number/var(p?cv?)/expr

        # Substitute fixed (numeric) constants (if any)
        dict_all_val = Dict(PEprob.model_info.model.parametermap) # Mapping: symbolics of all parameters => nominal values
        fixed_syms = setdiff( # Symbolics of fixed constants
            keys(Dict(dict_all_val)), 
            union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
        )
        dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms)) # Mapping: symbolics of fixed constants => values
        dict_z0sym_expr = Dict( # substitute fixed constant into all z0 expressions
            z0sym => Symbolics.simplify(Symbolics.substitute(expr, dict_fixed_val))
            for (z0sym, expr) in dict_z0sym_expr
        )
        
        # Create iterators
        itr_z0_fix  = Tuple{Int, Int, Float64}[]   # x0fixed
        itr_z0_p    = Tuple{Int, Int, Int}[]           # x0 = p
        itr_z0_cv   = Tuple{Int, Int, Int}[]          # x0 = cv
        itr_z0_func = Tuple{Int, Any}[]             # x0 = f(p,cv)

        z_syms = _get_z_syms(PEprob)
        p_syms = _get_p_syms(PEprob)
        cv_syms = _get_cv_syms(PEmodel)

        for v in 1:Nz
            val = dict_z0sym_expr[z_syms[v]]
            if Symbolics.value(val) isa Number
                # if z0 is a numeric value...
                append!(itr_z0_fix, ((v, cidx, Float64(Symbolics.value(val))) for cidx in 1:Nc))
            elseif (pidx = findfirst(x -> isequal(x, val), p_syms)) !== nothing
                # if z0 is an unknown parameter p...
                append!(itr_z0_p, ((v, cidx, pidx) for cidx in 1:Nc))
            elseif (cvidx = findfirst(x -> isequal(x, val), cv_syms)) !== nothing
                # if z0 is a condition-dependent variable cv...
                append!(itr_z0_cv, ((v, cidx, cvidx) for cidx in 1:Nc))
            else
                # if z0 is some arbitrary function of [p,cv]...
                z0_func = Symbolics.build_function(
                    val,
                    [p_syms; cv_syms]...,
                    expression = Val{false}
                )
                push!(itr_z0_func, (v, z0_func))
            end
        end

        # Create constraints
        if !isempty(itr_z0_fix)
            # Initial condition is a fixed value
            ExaModels.@add_con(c,
                z[v,1,0,cidx] - val
                for (v, cidx, val) in itr_z0_fix
            )
        end
        if !isempty(itr_z0_p)
            # Initial condition is the PHYSICAL value of unknown parameter p[pidx].
            # pidx is a per-entry data index, so partition by scale (see _p_phys).
            for sc in (:log10, :log, :lin)
                grp = [t for t in itr_z0_p if pscale[t[3]] === sc]
                isempty(grp) && continue
                if sc === :log10
                    ExaModels.@add_con(c, z[v,1,0,cidx] - exp(log(10.0)*p[pidx]) for (v,cidx,pidx) in grp)
                elseif sc === :log
                    ExaModels.@add_con(c, z[v,1,0,cidx] - exp(p[pidx])           for (v,cidx,pidx) in grp)
                else
                    ExaModels.@add_con(c, z[v,1,0,cidx] - p[pidx]                for (v,cidx,pidx) in grp)
                end
            end
        end
        if !isempty(itr_z0_cv)
            # Initial condition is a condition-dependent variable, cv
            ExaModels.@add_con(c,
                z[v,1,0,cidx] - cv[cvidx,cidx]
                for (v, cidx, cvidx) in itr_z0_cv
            )
        end
        if !isempty(itr_z0_func)
            # Initial condition is some arbitrary function, f(p,cv)
            for (v, z0_func) in itr_z0_func
                ExaModels.@add_con(c,
                    z[v,1,0,cidx] - z0_func(
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for cidx in 1:Nc
                )
            end
        end
    end

    return c
end