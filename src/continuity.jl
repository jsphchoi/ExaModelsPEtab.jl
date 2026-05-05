#
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
    # Unpack problem info
    (; Nz, N, Nc, L1) = PEinfo
    z = c.z

    itr_cont1 = [(v,i,cidx) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc]
    # TODO support ev (from ODEProblem callbacks) 
    # idea: create dict i=1:N -> f(t_event) to continuity constraint!
    ExaModels.@add_con(c,
        -z[v,i+1,0,cidx]
        for (v,i,cidx) in itr_cont1
    )
    itr_cont1! = [(v,i,cidx,j,L1[j+1]) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc, j in 0:K]
    ExaModels.@add_con(c,
        (v,i,cidx) => L1j*z[v,i,j,cidx]
        for (v,i,cidx,j,L1j) in itr_cont1!
    )
    return c
end

# Create initial condition continuity constraints
function _create_initial_conditions(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    # Unpack problem info
    (; Nz, Nc, Np) = PEinfo
    z = c.z
    p = c.p
    if Ncv >= 1
        cv = c.cv
    end

    # Check which type of initial condition
    if _check_x0SSpre(PEprob)
        # if x0SSpre(p)...
        zss = c.zss
        itr_ss1 = [(z,cidx) for v in 1:Nz, cidx in 1:Nc]
        ExaModels.@add_con(c,
            z[v,1,0,cidx] - zss[v,cidx]
            for (z,cidx) in itr_ss1
        )
        dict_cidx_sscidx = _get_dict_cidx_sscidx(PEmodel, PEprob)
        itr_ss2 = [dict_cidx_sscidx(cidx) for cidx in 1:Nc]
        for f in fs
            ExaModels.@add_con(c,
                f(
                    ntuple(v -> zss[v,cidx], Nz)...,
                    ntuple(m -> p[m], Np)...,
                    ntuple(m -> cv[m], Ncv)...
                )
                for cidx in itr_ss2
            )
        end

    else
        # if x0fix, x0 = p, x0 = f(p)...
        conditions_df = PEmodel.petab_tables[:conditions]
        dict_pstr_pidx = _get_dict_pstr_pidx(PEprob)
        # Create iterators
        itr_z0_fixed = Tuple{Int, Int, Float64}[]
        itr_z0_p = Tuple{Int, Int, Int}[]
        # itr_z0_fp = Tuple{Int, Int, Int}[]
        for v in 1:Nz
            for cidx in 1:Nc
                val = conditions_df[cidx,cvidx+2]
                if val isa Number
                    push!(itr_cv_fixed, (cvidx, cidx, Float64(val)))
                elseif val isa String || val isa Symbol
                    str_val = String(val)
                    parsed_val = tryparse(Float64, str_val)
                    if parsed_val !== nothing
                        push!(itr_cv_fixed, (cvidx, cidx, parsed_val))
                    else
                        if haskey(dict_pstr_pidx, str_val)
                            pidx = dict_pstr_pidx[str_val]
                            push!(itr_cv_p, (cvidx, cidx, pidx))
                        else
                            error("Condition variable '$str_val' not found in unknown parameter list.")
                        end
                    end
                end
            end
        end
        # Create constraints
        if !isempty(itr_z0_fixed)
            # x0fix
            ExaModels.@add_con(c,
                z[v,1,0,cidx] - val
                for (v, cidx, val) in itr_z0_fixed
            )
        end
        if !isempty(itr_z0_p)
            # x0 = p
            ExaModels.@add_con(c,
                z[v,1,0,cidx] - p[pidx]
                for (v, cidx, pidx) in itr_z0_p
            )
        end
        if !isempty(itr_z0_fp)
            # x0 = x0f(p)
            # x0fs = _
            # for x0f in x0fs
            #     ExaModels.@add_con(c,
            #         z[v,1,0,cidx] - x0f(ntuple(m -> p[m], Np))
            #         for (v, cidx) in itr_z0_fp
            #     )
            # end
        end
    end

    return c
end