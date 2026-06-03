# ExaModelsPEtab — Session Handoff (2026-06-03, constraint-debug session)

Package: `~/.julia/dev/ExaModelsPEtab` — builds an ExaModels orthogonal-collocation NLP from a
PEtab parameter-estimation problem and solves it with MadNLP on the GPU (CUDA/cuDSS).
Goal of recent work: **benchmark ExaModelsPEtab (MadNLP/CUDA) vs PEtab.jl (Optim.IPNewton)**
across the 35 Benchmark-Models, and along the way fix the model features that prevented several
models from compiling.

## TL;DR of THIS session — two real constraint-generation bugs found + fixed
Debugged the four "problem models". The two convergence-quality ones turned out to be genuine
constraint-generation bugs (NOT scaling/hardware), and crucially were **invisible to the nominal-θ
objective check** (it only uses warm-start z at measurement nodes + observable formulas — never the
ODE RHS or the IC/continuity constraints). Lesson: always also check the **warm-start constraint
violation** ‖c(x₀)‖∞, split by block.
- **Armistead** (was: drifted to wrong −288.33 vs PEtab −301.92) → **FIXED** (`src/continuity.jl`):
  IC constraints ignored conditions-table initial-state overrides. Now SOLVE_SUCCEEDED −306.13,
  PEtab nllh@θ*=−300.88 (≈ PEtab −301.92; −306 vs −301 is K=6 discretization, not a bug).
- **Bertozzi** (was: garbage 1.166e12, MadNLP stall) → **FIXED** (`src/utils.jl` + `src/continuity.jl`):
  a parameter defined by an SBML initialAssignment (`beta_N=R0_*gamma_/N_`) was re-evaluated with
  per-condition cv values (4.86e-9) instead of PEtab's frozen-at-default constant (0.01), so the RHS
  was off ~2.6e6× → warm start 100% collocation-infeasible. Now SOLVE_SUCCEEDED 5.708e9 = PEtab
  nllh@θ* exactly (genuine local min; PEtab's 1.94e9 is a different/better local min → multistart
  concern, not a bug).
- **Bachmann** (GPU OOM) and **Fiedler** (sm_70 kernel-param overflow) — **deferred, untouched**.

⚠️ Src is changed but **UNCOMMITTED**. The live benchmark process was launched BEFORE these edits, so
it still runs the buggy code — its Armistead/Bertozzi rows will be stale (pre-fix). After committing,
**re-run Armistead + Bertozzi** (strip their `exa_*` keys, re-benchmark).

## Environment
- Julia 1.12.6; project env at the package root (`julia --project=.`).
- 2× Quadro GV100 (32 GB). **`nvidia-smi` is broken** (NVML driver/library version mismatch) but
  **CUDA.jl is functional** (`CUDA.functional() == true`). Monitor GPU work via result files, not nvidia-smi.
- 96 CPU cores, 376 GB RAM.
- Ipopt was added to the project this session (`Pkg.add("Ipopt")`) for the PEtab Ipopt extension.
- Fides.jl deliberately NOT added (wraps a Python pkg; heavy/fragile).

## Git state
- Last commit: `3cf5fdb` "edit handoff" (benchmark scripts committed in `20c28af`).
- **UNCOMMITTED this session**: `src/continuity.jl` + `src/utils.jl` (the two constraint bug fixes
  below, #5 and #6) and this HANDOFF.md. Both fixes are validated (see below) but not yet committed.
- Prior src fixes (assignment-rule substitution, condition-id alignment, Option A mesh extension,
  x0SSpre) and the benchmark scripts (K=6, SGM reruns, Crauste warmup, 19-model list) are committed.

---

## Fixes implemented (#1–#4 COMMITTED; #5–#6 are THIS session, validated but UNCOMMITTED; kept as implementation reference)

### 1. x0SSpre (steady-state pre-equilibration)
- `initialize.jl _solve_conds` / `_get_zss_init`: pre-eq models reject plain condition ids in
  `PEtab.get_odeproblem`; must pass the documented **Pair `pre_eq_id => sim_id`**. That call
  internally pre-equilibrates and returns an ODEProblem whose **`u0` IS the steady-state initial
  condition** — read `u0[1:Nz]` for `zss` init (no separate SteadyStateProblem).
- `utils.jl _get_dict_cidx_sscidx`: was declared `::Dict` but built a `Vector` (errored on every
  call) → now returns `Vector{Int}`, nothing-guarded.
- `continuity.jl` x0SSpre block: defined `fs, has_t = _get_rhs_funcs(...)` (was undefined); fixed
  `itr_ss1` to `(v,cidx)`; constraint `z[v,1,0,cidx]=zss[v,cidx]`; residual
  `f(zss[:,cidx]; cv[:,sscidx])=0` evaluated under the PRE-EQ condition's cv; `has_t`→pass t=0.

### 2. SBML assignment-rule substitution
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

### 3. Condition-id alignment (pre-eq-only conditions)
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

### 4. Heterogeneous-span mesh — "Option A" (committed; objective-check re-verify still pending — Zhao compiling in the live benchmark)
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

### 5. Conditions-table initial-state override in IC constraints (THIS session — UNCOMMITTED)
**Bug:** A conditions-table column whose header matches a STATE id sets that state's initial value
PER CONDITION (PEtab spec). This override is NOT in the SBML/MTK `speciemap` (which holds one
condition-independent default), so `continuity.jl _create_initial_conditions` took the speciemap
numeric default for every condition → the IC constraint `z[v,1,0,cidx]=<default>` forced all
conditions to the same wrong initial state. (Armistead: mutant condition forced to wild_type ICs;
the warm-start z and cv were correct, only the IC *constraint* was wrong — invisible to the obj check.)
**Fix** (`continuity.jl`): in `_create_initial_conditions`, before the speciemap branch, detect
states whose name (z_sym stripped of `(t)`) matches a cv column name (`_get_cv_colnames`) and route
them to `z[v,1,0,cidx] = cv[cvidx,cidx]` (cv already holds the correct per-condition value).
**Validated:** Armistead warm-start IC viol 0.519→0; GPU solve −288.33→−306.13; PEtab nllh@θ*=−300.88.
**Blast radius (19-model benchmark): only Armistead** (Lang/Smith also have it but Lang isn't
benchmarked and Smith has SBML events). No-op for models without a state/cv-column name match.

### 6. initialAssignment-defined parameters frozen at defaults (THIS session — UNCOMMITTED)
**Bug:** A PARAMETER defined by an SBML `initialAssignment` (e.g. Bertozzi `beta_N => R0_*gamma_/N_`,
with `R0_/gamma_/N_` being condition vars) is FROZEN by PEtab at the value of that expression
evaluated with the model's DEFAULT parameter values (`0.1*0.1/1 = 0.01`), even though the condition
overrides apply to `R0_/gamma_/N_` themselves (verified in `oprob.p`). But `_get_rhs_funcs`'s
`dict_fixed_val` kept `beta_N` SYMBOLIC, so it got re-evaluated with the per-condition cv values
(`4.86e-9`) → the `beta_N`-containing RHS terms (dI/dt, dS/dt) were off by ~2.6e6× while `dR/dt=γI`
(no beta_N) was correct → the warm start (from PEtab's ODE) was ~100% collocation-infeasible
(residual 1.16e6, unchanged at K=12 → NOT mesh coarseness) → MadNLP stalled at 1.166e12.
**Fix** (`utils.jl`): new `_resolve_fixed_vals(PEmodel, PEprob)` fixpoint-substitutes the parametermap
into itself until every default is numeric, then returns the fixed (non-estimated, non-cv) params as
numeric constants (falls back to the raw symbolic value if a param can't reduce — no regression).
Used in both `_get_rhs_funcs` and the `continuity.jl` IC fixed-constant substitution.
**Validated:** RHS-vs-PEtab ratio →1.0 (all states/conditions); colloc residual 1.16e6→0.22
(mean rel 0.58→0.001); GPU solve SOLVE_SUCCEEDED 5.708e9 = PEtab nllh@θ* exactly.
**Blast radius (19-model benchmark): only Bertozzi** (Laske also has param initialAssignments but
they reference fixed constants → `_resolve_fixed_vals` yields identical numeric values; neutral).
NOTE: Bertozzi now finds a genuine local min 5.708e9; PEtab's 1.94e9 is a different/better local
min → a multistart/landscape concern, NOT a bug.

---

## Diagnostic methodology added THIS session (how the two bugs were found)
The nominal-θ objective check (below) does NOT exercise the ODE RHS or the IC/continuity constraints
— it only uses warm-start z at measurement nodes + observable/noise formulas — so it passed cleanly
for BOTH buggy models. The bugs only showed up in the **solve** and in the **warm-start constraint
violation**. Procedure that nailed them (CPU, no GPU needed):
1. **Warm-start violation, split by block.** `‖c(x₀)‖∞` = unscaled ∞-norm of the equality-constraint
   residual at `m.meta.x0`. Split into collocation / cv / interval-continuity / IC / y+σ blocks to
   localize. (`examples/scratch_tests/warm_start.jl` lumps collocation+cv and all continuity; the
   session used finer per-sub-block splits.) Armistead → all in IC; Bertozzi → all in collocation.
2. **Is it a real RHS bug or mesh coarseness?** Compare our `_get_rhs_funcs` f to PEtab's actual ODE
   RHS: `oprob,_=PEtab.get_odeproblem(θ,PEprob;condition=cid); oprob.f(du,u0,oprob.p,0.0)` vs
   `fs[v](u0..., pphys..., cvval...)`. Ratio should be 1.0. Also recompute the collocation residual
   manually: A=`Σ_j z[v,i,j,cidx]·DLDTAU[j+1,k]` vs B=`h[i]·f(...)` — if A≫B at a node where
   `z_init==sol(t_node)` exactly, the RHS is wrong (raising K won't help; Bertozzi K=6==K=12).
3. **Is the warm-start z correct?** Check `max|z_init − sol[cid](t_node)|` over all nodes (was 0.0
   for Bertozzi → sampling fine, so the residual was purely the RHS).
(Throwaway scripts lived in `/tmp/diag_*.jl`, `/tmp/val_check.jl`, `/tmp/exp_solve.jl` — not kept.)

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

⚠️ **This check is necessary but NOT sufficient.** It passed to ~1e-11 for BOTH Armistead and
Bertozzi while each had a constraint-generation bug, because the warm-start objective only touches z
at measurement nodes + observable/noise formulas — it never evaluates the ODE RHS or the
IC/continuity constraints. Always pair it with the **warm-start constraint-violation** check and, on
any nonzero violation, the RHS-vs-PEtab and per-block localization above.

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

## Benchmark infrastructure (examples/Benchmarks/)
All benchmark code in `examples/Benchmarks/`. Results in `examples/Benchmarks/results/`,
combined key=value `{ModelName}_results.txt` per model (petab_* and exa_* keys coexist).

### Scripts
- `benchmark_petab.jl / .sh` — one Julia process per model; builds PEtabODEProblem (JITs
  nllh/grad/hess in compile phase), solves with Optim.IPNewton(), 1-hr cap. Writes `petab_*` keys.
  Run from repo root: `bash examples/Benchmarks/benchmark_petab.sh`.
- `benchmark_examodels.jl / .sh` — two long-lived Julia processes (one per GPU), strided over
  19 models. Warmup model: **Crauste** (excluded from ALL_MODELS — pre-warms generic JIT only;
  timing of the warmup model is invalid). K=6, compile deadline 4 hr. Writes `exa_*` keys.
  Run from repo root: `bash examples/Benchmarks/benchmark_examodels.sh` (**no trailing `&`** —
  the script uses `wait` internally; the harness must track the outer process to report true completion).
- `final_report.jl` — prints/writes `results/final_report.txt`.
  `--sgm` flag swaps exa Solve(s) column to SGM times (fair comparison with PEtab).

### Per-model sequence (examodels)
1. **Warmup** (Crauste, once per process) — pre-compiles generic Julia pipeline.
2. **Compile** — Phase 1: PEtab setup + ODE presolve at nominal θ (`exa_presolve_time`).
   Phase 2: Symbolics code-gen + ExaModels build (`exa_compile_time` = Phase1+2 total).
3. **First solve** — MadNLP/cuDSS; includes GPU kernel JIT on first call (`exa_solve_time`).
4. **SGM solves** — `madnlp(model)` called N_SGM_RERUNS=3 more times on the same in-memory
   model (no recompile; GPU kernels cached; `model.meta.x0` not mutated so all runs share
   the same warm start). Geometric mean → `exa_sgm_solve_time`.
   If resuming after a restart, model is rebuilt silently (untimed) then one priming solve
   warms the GPU kernels before the 3 timed runs.

### Result keys
- `exa_compile_time` / `exa_presolve_time` — cold first-run build (comparable to `petab_compile_time`)
- `exa_solve_time` — first solve; **inflated by GPU kernel JIT**, not directly comparable
- `exa_sgm_solve_time` / `exa_sgm_n` — kernel-cached solve SGM; **comparable to `petab_solve_time`**
- `exa_compile_status`, `exa_solve_status`, `exa_sgm_status`, `exa_term_status`, `exa_objective`,
  `exa_iter`, `exa_nvar`, `exa_ncon`, `exa_error`
- `%EXA` = (exa_compile_time − exa_presolve_time) / exa_compile_time × 100

### Models benchmarked (19)
petab_optimum_found=true AND petab_has_events=false, Crauste excluded (warmup):
Armistead, Bachmann, Bertozzi, Blasi, Boehm, Borghans, Bruno, Elowitz, Fiedler,
Laske, Lucarelli, Okuonghae, Perelson, Rahman, SalazarCavazos, Schwen, Sneyd, Zhao, Zheng.

### Critical harness notes
- **Run julia at `-t 1`**; enforce timeouts with external `bash -c "sleep N; kill -9 <pid>"`.
  A Julia watchdog thread starves cuDSS (20 s solve → 690 s observed).
- **Launch without trailing `&`** when using `run_in_background`; the script uses `wait` so
  the outer process only exits when both GPU instances finish.
- Resumable: `compiling`/`solving`/`exa_sgm_status=running` sentinels → `timeout`/`interrupted`
  on restart; `exa_finished()` re-queues anything not in a terminal state.
- To re-run from scratch: strip `exa_*` keys, keep `petab_*`:
  `grep "^petab_" file > tmp && mv tmp file`

## Scratch / exploratory test files (examples/scratch_tests/)
- `gpu_run.jl` — end-to-end GPU solve sanity check
- `sigma_constraints.jl` — debug sigma constraint construction
- `stage_build.jl` — step through model build stages manually
- `verify_models.jl` — objective-consistency check (our obj vs PEtab @ nominal θ); main validation tool
- `warm_start.jl` — test warm-start initialization from ODE solution

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

## Benchmark run (2026-06-03) — FIVE PROBLEM MODELS (2 fixed this session, 3 deferred)
Benchmark relaunched via `bash examples/Benchmarks/benchmark_examodels.sh` (2 GPUs, K=6).
Clean matches: **Boehm** (exaObj 138.2 = petab 138.2, SGM 0.44s vs petab 10.2s), **Bruno** (−46.71 =
−46.69, 0.39s vs 3.34s) — obj matches PEtab, SGM solve 10–25× faster. The five problems:

1. **Bachmann** 🔴 DEFERRED — GPU OOM. Biggest model; single 8.945 GiB alloc fails on already-99.9%-full
   32 GB GV100. Hardware capacity, not a bug. Fix candidates (untested this session): retry alone at
   lower K (K=4); and/or `CUDA.reclaim()` between models in the long-lived benchmark process — the
   "99.9% full BEFORE the alloc" strongly suggests prior models' GPU memory isn't being freed
   (accumulation), so reclaiming between solves may let it fit even at K=6. **Test alone first** to
   tell "needs lower K" from "needs cache reclaim".
2. **Fiedler + Lucarelli** 🔴 DEFERRED — CUDA kernel param-memory overflow. ExaModels fuses a giant
   expression (Fiedler 36.711 KiB / 4371-element Compressor SIMDFunction; Lucarelli 35.344 KiB) into
   one kernel, exceeding the **31.996 KiB kernel-param limit on sm_70** (GV100 = cc 7.0). Structural
   ExaModels/GV100 limit on oversized fused exprs, NOT OOM. Fix candidate: split the fused
   objective/constraint expression, or run on sm_80+ hardware.
3. **Bertozzi** ✅ FIXED this session — was garbage 1.166e12 / `SEARCH_DIRECTION_BECOMES_TOO_SMALL`.
   Root cause = the `beta_N` initialAssignment RHS bug (Fix #6 above). Now SOLVE_SUCCEEDED 5.708e9 =
   PEtab nllh@θ* (genuine local min; PEtab's 1.94e9 is a different/better local min → multistart, not
   a bug).
4. **Armistead** ✅ FIXED this session — was −288.33 vs petab −301.92. Root cause = the conditions-table
   IC-override bug (Fix #5 above). Now SOLVE_SUCCEEDED −306.13, PEtab nllh@θ*=−300.88 (≈ PEtab).

⚠️ CORRECTION to the prior handoff's claim "the src is NOT implicated — all compile fine": #3 and #4
were genuine **constraint-generation bugs** (they compiled fine but built the WRONG NLP). Only #1/#2
are hardware-structural. The live benchmark ran pre-fix code, so re-run #3/#4 after committing.

### Relaunch / resume the benchmark
```bash
# From repo root — no trailing &, harness must track the outer process
bash examples/Benchmarks/benchmark_examodels.sh
```
Resumable: re-running re-queues any model not in a terminal `exa_*` state. To re-run a model from
scratch, strip its `exa_*` keys (keep `petab_*`): `grep "^petab_" file > tmp && mv tmp file`.
Warmup (Crauste, ~5 min) runs once per GPU; then 19 models sequentially per GPU. Slow compiles:
Laske, Blasi, Lucarelli (Lucarelli petab-compile alone was 2676s). When all done, run
`final_report.jl --sgm` for the canonical table.

## OUTSTANDING / NEXT STEPS
0. **Commit Fixes #5 + #6** (`src/continuity.jl`, `src/utils.jl`), then **re-run Armistead + Bertozzi**
   in the benchmark (the live run used pre-fix code): strip their `exa_*` keys
   (`grep "^petab_" file > tmp && mv tmp file`) and relaunch. Also worth a quick regression re-check
   of an assignment-rule model (Rahman/Okuonghae) and Laske, since #6 touches the shared
   `_get_rhs_funcs`/IC fixed-constant path (expected no-op for them — they reference fixed constants).
1. **Bachmann** (DEFERRED, OOM): test alone in a fresh process at K=6 first (GPU-mem accumulation vs
   genuine size); add `CUDA.reclaim()` between models in the benchmark; fall back to K=4 if needed.
2. **Fiedler + Lucarelli** (DEFERRED, sm_70 kernel-param overflow): split the oversized fused
   objective/constraint expression so no single kernel exceeds 31.996 KiB of params, or run on sm_80+.
3. **Bertozzi local-min gap** (NOT a bug): we converge to 5.708e9, PEtab to 1.94e9 — different local
   minima from the same nominal start. If matching PEtab's optimum matters, try multistart / a better
   warm start; otherwise document it.
4. **Verify Option A** (mesh extension) by objective check on Zhao (expect match) +
   Crauste/Brannmark/Zheng (no regression). CPU pattern in "Validation methodology".
5. **Laske compile speed**: 29 nested assignment rules → very large substituted RHS/observable
   expressions → ~30-min compile. Consider caching/CSE or limiting expansion if it matters.
6. **Event callbacks unsupported** (collocation can't represent discontinuities): Liu, Smith (SBML
   events) and time-input/pre-eq-reset models. Out of scope unless tackled deliberately.
7. Decide whether to keep the single-shared-mesh (Option A, extends short conditions to max span —
   wasted DOF + possible blow-up if a condition's ODE diverges past its window) vs per-condition
   meshes (cleaner, ragged z, bigger refactor).

## Useful one-liners
- Objective check (CPU): see "Validation methodology" above.
- Event ground-truth: `grep -c '<event[ >]' <model>/*.xml` (only Liu, Smith > 0).
- Per-model PEtab/Exa status: read `examples/Benchmarks/results/<Model>_results.txt` (combined
  `petab_*` + `exa_*` keys).
- Warm-start constraint violation (CPU): `examples/scratch_tests/warm_start.jl <Model> <K>`.
