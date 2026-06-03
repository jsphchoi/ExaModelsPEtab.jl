# ExaModelsPEtab — Session Handoff (2026-06-03, updated end-of-session)

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
- Last commit: `7cbda66` "clean up repo, organized benchmark and results".
- **UNCOMMITTED working-tree changes:** `src/structs.jl`, `src/userfuncs.jl` (whitespace only).
- The core src fixes (assignment-rule substitution, condition-id alignment, mesh extension,
  x0SSpre) are all committed — confirmed working via objective-consistency checks.
- `examples/Benchmarks/benchmark_examodels.jl` and `final_report.jl` have significant
  uncommitted changes from this session (K=6, SGM solve-only reruns, Crauste warmup,
  19-model list) — commit these before the next benchmark run.

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

## Benchmark run status (end of session 2026-06-03)
No results yet — all 19 model result files have `exa_*` keys cleared (only `petab_*` remain).
The benchmark processes were killed mid-warmup (SIGTERM during LLVM JIT of Crauste/CSV.File).
**To start the benchmark in a new session:**
```bash
# From repo root — no trailing &, harness must track the outer process
bash examples/Benchmarks/benchmark_examodels.sh
```
Commit `examples/Benchmarks/benchmark_examodels.jl` and `final_report.jl` first.
Expected: warmup (Crauste, ~5 min compile + solve), then 19 models sequentially per GPU.
Fast models (Bruno, Perelson, ~1 min) first; slow ones (Laske, Blasi, Fiedler, Lucarelli,
Bachmann) may approach the 4-hr compile limit or fail. Results appear in
`examples/Benchmarks/results/` as each model finishes.

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
