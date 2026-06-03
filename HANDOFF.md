# ExaModelsPEtab — Session Handoff (2026-06-03)

Package: `~/.julia/dev/ExaModelsPEtab` — builds an ExaModels orthogonal-collocation NLP from a
PEtab parameter-estimation problem and solves it with MadNLP on the GPU (CUDA/cuDSS).
Goal of recent work: **benchmark ExaModelsPEtab (MadNLP/CUDA) vs PEtab.jl (Optim.IPNewton)**
across the 35 Benchmark-Models, and along the way fix the model features that prevented several
models from compiling.

## Environment
- Julia 1.12.6; project env at the package root (`julia --project=.`).
- 2× Quadro GV100 (32 GB). **`nvidia-smi` is broken** (NVML driver/library version mismatch) but
  **CUDA.jl is functional** (`CUDA.functional() == true`). Monitor GPU work via result files, not nvidia-smi.
- 96 CPU cores, 376 GB RAM.
- Ipopt was added to the project this session (`Pkg.add("Ipopt")`) for the PEtab Ipopt extension.
- Fides.jl deliberately NOT added (wraps a Python pkg; heavy/fragile).

## Git state
- Last commit: `b9c551f` "testing petab: 29/35, 20/20 w/o event callbacks".
- **UNCOMMITTED working-tree changes (this session's core fixes):**
  `src/utils.jl`, `src/initialize.jl`, `src/objective.jl`, `src/collocation.jl`, `src/variables.jl`.
- Committed earlier: x0SSpre fix + sigma improvements (`f7a9e4d`), benchmark scripts + phase-1 results.
- `examples/results2/*` and the new `examples/*.jl|*.sh` are staged (from an earlier `git add -A`).

---

## Fixes implemented this session (all validated by the objective check below unless noted)

### 1. x0SSpre (steady-state pre-equilibration) — COMMITTED (f7a9e4d) + refined
- `initialize.jl _solve_conds` / `_get_zss_init`: pre-eq models reject plain condition ids in
  `PEtab.get_odeproblem`; must pass the documented **Pair `pre_eq_id => sim_id`**. That call
  internally pre-equilibrates and returns an ODEProblem whose **`u0` IS the steady-state initial
  condition** — read `u0[1:Nz]` for `zss` init (no separate SteadyStateProblem).
- `utils.jl _get_dict_cidx_sscidx`: was declared `::Dict` but built a `Vector` (errored on every
  call) → now returns `Vector{Int}`, nothing-guarded.
- `continuity.jl` x0SSpre block: defined `fs, has_t = _get_rhs_funcs(...)` (was undefined); fixed
  `itr_ss1` to `(v,cidx)`; constraint `z[v,1,0,cidx]=zss[v,cidx]`; residual
  `f(zss[:,cidx]; cv[:,sscidx])=0` evaluated under the PRE-EQ condition's cv; `has_t`→pass t=0.

### 2. SBML assignment-rule substitution — UNCOMMITTED
Several models referenced SBML assignment rules (`MTK.observed`, e.g. `total_pop = Σstates`,
`gamma_2 = f(Trigger…)`, `pY1173 = Σspecies/c1`) in the ODE RHS or observable formulas; these
symbols leaked into `build_function` unbound → `UndefVarError: <sym> not defined in Symbolics`.
- `utils.jl _assignment_substitutor(PEprob; bare)`: NEW. Builds a substitution dict from all
  non-state `MTK.observed` rules and applies it to a **fixpoint** (keeps substituting while any
  rule symbol remains among `get_variables(expr)`; uses `isequal`-membership not `intersect`/`in`
  because Symbolics `==` returns a symbolic eq, not Bool; capped at 100 iters vs cyclic rules).
  `bare=false` keeps MTK `(t)` form (for the RHS); `bare=true` strips `(t)` to match the
  objective's bare-symbol parsing of table formulas.
- `utils.jl _get_rhs_funcs`: replaced the single-pass `dict_utp` (only direct-in-RHS rules) with
  `subst_rules = _assignment_substitutor(...; bare=false)` applied to fixpoint, then `dict_fixed_val`.
  This resolves nested rules AND constant params introduced by rules (e.g. Laske `D_rib=160`).
- `objective.jl _create_objective`: added `apply_rules = _assignment_substitutor(...; bare=true)`
  and apply it to parsed observable formula, parsed noise formula, and the observable used by
  `_reduce_sigma_to_obs`. Restructured the observable block to compute `obs_sym` (with rules
  applied) first, then branch on whether it's a single state (`zidx !== nothing`) vs an expression.

### 3. Condition-id alignment (pre-eq-only conditions) — UNCOMMITTED
Zheng has a `preequilibration` conditions-table row that is **pre-eq-only** (never a
simulationConditionId). `_get_cids` used to return ALL conditions-table rows, so `_get_z_init`
tried `sol[:preequilibration]` → `KeyError`. Fix = make the canonical `cidx` dimension = the
SIMULATION conditions:
- `utils.jl _get_cids`: now returns conditions-table conditionIds that appear as a
  `simulationConditionId` in measurements, in table order. (Unchanged for non-pre-eq models.)
- `utils.jl _get_cond_rows` (NEW): cidx → conditions-table row index (since `_get_cids` may now be
  a subset, `cidx` ≠ row number). Used for condition-variable lookups.
- `variables.jl _create_cv` and `collocation.jl` cv-constraint construction: read
  `conditions_df[cond_rows[cidx], col]` instead of `conditions_df[cidx, col]`.
- `_get_dict_cidx_sscidx`: robust — if a sim condition's pre-eq condition isn't itself in
  `_get_cids`, falls back to cidx (residual uses that condition's own cv; exact when the pre-eq cv
  equals the sim cv or isn't in the RHS; the steady-state *start* is always correct via PEtab u0).

### 4. Heterogeneous-span mesh — "Option A" — UNCOMMITTED, **NOT yet verified**
Models like Zhao have conditions with different time spans (e.g. 6, 14, 24, 39). The single shared
collocation mesh was built from `argmax(length(sol.t))` (most ODE points) → could top out at a
short condition's tmax and miss later measurement times → `_get_dict_t_tidx` `findfirst`→`nothing`
→ `Nothing→Int64`. Design intent (correct): `tstops` = union of all measurement times forced on
every solve, so the mesh should hit them all. Fix (Option A, chosen by user):
- `initialize.jl _solve_conds`: integrate EVERY condition over the full mesh span
  `[tstart, max(t_meas)]` via `ODE.remake(odesys; tspan=(tstart, max(maximum(tstops), tspan[2])))`,
  so `z_init` has a real ODE profile at every shared-mesh node.
- `initialize.jl _get_z_init`: reverted to original `argmax(length(sol.t))` mesh selection and plain
  `sol[cid](t)` (no clamp) — valid now that all conditions span the full mesh.
- (An interim "clamp shorter conditions" version was replaced by Option A; user preferred providing
  the true full ODE profile since the collocation enforces the ODE over the full mesh anyway.)
- **TODO: verify Zhao still matches PEtab and Crauste/Brannmark/Zheng don't regress** (not run yet).

---

## Validation methodology — the "matches PEtab to <tol>" check
Fixed-point **objective-consistency** check at the NOMINAL parameters (NOT a solve):
```julia
PEp   = PEtab.PEtabODEProblem(PEtab.PEtabModel(yaml))
petab = PEp.nllh(PEtab.get_x(PEp))                  # PEtab's exact neg-log-likelihood @ nominal θ
m     = petab_examodel(yaml; backend=nothing, K=5)  # build our collocation NLP (CPU; fast, no GPU)
exa   = ExaModels.NLPModels.obj(m, m.meta.x0)       # OUR objective at the model's START point
reldiff = abs(exa-petab)/abs(petab)
```
`m.meta.x0` = our warm start (z = nominal ODE sampled on the mesh, p = nominal, y/σ computed). Both
numbers are "the likelihood at the nominal θ" via independent code paths, so a small reldiff means
our whole objective formulation (observables, σ, assignment rules, condition indexing, mesh,
state-at-measurement-time mapping) reproduces PEtab. Residual ~1e-7 = K=5 collocation discretization
+ roundoff. Use `backend=nothing` (CPU) for these — much faster than GPU and the obj is identical.
(Separately, a real MadNLP *solve* check was done for Brannmark earlier in the session.)

## Validation results (objective vs PEtab @ nominal, K=5, CPU)
| Model | feature exercised | reldiff |
|---|---|---|
| Crauste | control (non-pre-eq) | 4.2e-10 ✅ |
| Brannmark | pre-eq (x0SSpre) | 8.4e-9 ✅ |
| Zheng | pre-eq-only condition alignment | 9.7e-8 ✅ |
| Rahman | nested RHS assignment rule | 3.6e-11 ✅ |
| Okuonghae | `total_pop=Σstates` rule in RHS | 1.9e-9 ✅ |
| SalazarCavazos | `pY1173` rule in observable | matched (validated mid-session) |
| Zhao | heterogeneous-span mesh | 1.15e-7 ✅ (with interim clamp; **re-verify under Option A**) |
| Laske | `D_rib=160` const-param via rule | builds + resolves, but compile is VERY slow (~30 min+); not re-confirmed under Option A |

All 5 originally-failing models (Zhao, Rahman, Okuonghae, Laske, SalazarCavazos) + Zheng now
compile; only Laske's compile time is a concern (29 assignment rules → large substituted exprs).

---

## Benchmark infrastructure (examples/)
Two-phase pipeline, results in `examples/results2/`, key=value `.txt` files per model.
- `bench2_petab.jl <model>` — PHASE 1 worker (one process per model; run in parallel). Builds
  PEtabODEProblem, triggers its nllh/grad/**hess** JIT during the COMPILE phase (so the solve is
  JIT-free and `time_limit` governs it), solves with **Optim.IPNewton()** on the default problem
  (size-auto Hessian: ForwardDiff small / GaussNewton medium), 1-hr wall cap, records compile/solve
  status+time, objective, `optimum_found` (= Optim `res.converged::Bool`), `has_events`.
- `run_petab.sh [P]` — runs phase 1 over all 35 models, P-way parallel (default 12), resumable.
- `bench2_exa.jl <gpu_id> <ninst> <idx>` — PHASE 2 worker: single long-lived **warmed** resumable
  process pinned to a GPU; for each model the PEtab phase RAN, builds the ExaModel (K=5, 30-min
  compile watchdog) and solves with MadNLP/cuDSS. MadNLP wall = **2× PEtab solve time if PEtab
  converged, else 1 hr**.
- `run_exa.sh` — launches 2 instances (one per GPU), each with a restart wrapper (resume after a
  watchdog SIGKILL on a compile timeout).
- `master2.sh` — phase1 → phase2 → `report2.jl` → `results2/FINAL_REPORT2.txt`.
- `report2.jl` — combined report: per-model PEtab & ExaModels comp/solve status+time, relative
  objective gap, and proportions of PEtab/ExaModels optima found (overall + no-event subset).
- (Older single-GPU harness: `benchmarks.jl`, `compare.jl`, `run_all.sh`, `report.jl`, `master.sh`.)

### Critical harness gotchas (do not regress)
- **Run MadNLP/GPU at `-t 1`** and enforce timeouts with an **external `bash -c "sleep N; kill -9 <pid>"`
  watchdog**, NOT a Julia watchdog thread. A busy Julia watchdog thread STARVES cuDSS and makes the
  GPU solve hang (a 20 s solve became 690 s). `-t 1` + external killer is the proven config.
- Per-model JIT: a fresh process JITs ~5 min of generic pipeline. Phase 2 uses ONE warmed process +
  a JIT warmup so per-model compile times are real. Phase-1 workers absorb the Hessian JIT in the
  COMPILE phase.
- Resumable: per-model result files; `compiling`/`solving` sentinels convert to `timeout` on restart.

## PEtab optimizer findings (from PEtab.jl docs)
`docs/.../optimizers.md`: small (<10 ODE,<20 par) → `Optim.IPNewton()` + exact Hessian; medium →
`Fides.CustomHessian()`+GaussNewton; large → (L)BFGS. **LBFGS does NOT certify convergence** on
these stiff problems (objective converges but gradient won't hit tol → status -5, runs to the cap →
"optimum found" ≈ 0%). **IPNewton converges in a few iters** with the exact/GN Hessian and was
chosen. `res.converged` is a Bool for Optim, an Int Ipopt status (`:Optmisation_failed` if it threw)
for IpoptOptimizer.

## Prior benchmark run (FINAL_REPORT2.txt, LBFGS era — superseded)
The committed report used Ipopt **LBFGS** (before switching to IPNewton) AND before the
assignment-rule/alignment/mesh fixes. Key takeaways still relevant:
- 29/35 PEtab ran; 20/20 no-event models found an optimum (PEtab side).
- **The "2× PEtab solve time" budget STARVED ExaModels**: IPNewton/PEtab converges in seconds, so
  2× gave the collocation solver too little wall time → many `MAXIMUM_WALLTIME_EXCEEDED` rows that
  were actually essentially converged (objective gaps 1e-8..1e-11). Where it had adequate time,
  ExaModels matched PEtab to ~1e-8.
- `has_events` in the harness is **callback-based and over-flags**: true SBML `<event>` models are
  only **Liu** and **Smith**; the harness also flags pre-eq / time-input discontinuity models. The
  collocation cannot represent ANY continuity-breaking discontinuity (events, time-input jumps,
  pre-eq dilution resets), so for the "no-event subset" exclude all of those.

---

## OUTSTANDING / NEXT STEPS
1. **Verify Option A** (mesh extension): re-run the objective check on Zhao (expect match) +
   Crauste/Brannmark/Zheng (no regression). Quick CPU script pattern above; `/tmp/val_*.jl` examples.
2. **Re-run the full Ipopt-filtered benchmark** with ALL current fixes (5 more models now compile)
   and a **better ExaModels solve budget** — the 2× rule starves it; use e.g. `max(2×petab, 600 s)`
   floor (edit `madnlp_limit` in `bench2_exa.jl`). User's stated rule: PEtab converged → 2×T; PEtab
   hit its 1-hr cap → 1 hr for ExaModels.
3. **Laske compile speed**: 29 nested assignment rules → very large substituted RHS/observable
   expressions → ~30-min compile. Consider caching/CSE or limiting expansion if it matters.
4. **Event callbacks unsupported** (collocation can't represent discontinuities): Liu, Smith (SBML
   events) and time-input/pre-eq-reset models. Out of scope unless tackled deliberately.
5. Decide whether to keep the single-shared-mesh (Option A, extends short conditions to max span —
   wasted DOF + possible blow-up if a condition's ODE diverges past its window) vs per-condition
   meshes (cleaner, ragged z, bigger refactor).
6. Commit the src fixes once Option A is verified.

## Useful one-liners
- Objective check (CPU): see "Validation methodology" above.
- Event ground-truth: `grep -c '<event[ >]' <model>/*.xml` (only Liu, Smith > 0).
- Per-model PEtab/Exa status: read `examples/results2/<model>.{petab,exa}.txt`.
