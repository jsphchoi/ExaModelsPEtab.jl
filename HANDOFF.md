# ExaModelsPEtab — Handoff

Package `~/.julia/dev/ExaModelsPEtab` builds an ExaModels orthogonal-collocation NLP from a PEtab
parameter-estimation problem and solves it with MadNLP on the GPU (CUDA/cuDSS). Active work:
**benchmark ExaModelsPEtab (MadNLP/CUDA) vs PEtab.jl (Optim.IPNewton)** across the 35 Benchmark-Models,
fixing the model features that block compilation/convergence along the way.

This doc is **stable reference only**. Live per-model pass/fail is NOT snapshotted here (it rots) —
read it from the source of truth:
- `examples/Benchmarks/results/{Model}_results.txt` — combined `petab_*` + `exa_*` key=value per model.
- `julia --project=. examples/Benchmarks/final_report.jl --sgm` — regenerates the canonical table.
- `ps aux | grep benchmark_examodels` — is the run live, and which PID (it changes on each auto-restart).

## Git state
- Fixes #1–#10 below are COMMITTED on `main`. #5/#6 in `95e7729`; **#7 fixed-time events in `7454a85`**
  (`events-fixed-time` fast-forwarded into `main` 2026-06-05); #8 Blasi steady-state path in `dc1b295`;
  #9 Bruno-warmup restructure in `08273a3`; **#10 observable build_function-blowup fixes (ov + zN) merged
  in `399fb43`**, then large-expr cap + event tweaks in `e848fb8` (`ov-work` merged 2026-06-05; both side
  branches/worktrees deleted — single `main`, working tree clean). Benchmark scripts (K=6, SGM reruns,
  **Bruno** warmup) committed.
- **Canonical model lists + results layout (2026-06-05; committed in `be131ce`).** All three
  scripts now `include("examples/Benchmarks/list_benchmarks.jl")` — the single source of truth: `BENCHMARK_MODELS` (35) ⊇
  `PETAB_SOLVED_MODELS` (28, drops the 7 PEtab-side failures) ⊇ `EXA_SUPPORTED_MODELS` (26, drops Beer +
  Liu). benchmark_examodels.jl derives its 24 in-loop models = `EXA_SUPPORTED − {Bruno, Crauste}`;
  final_report.jl rows = `PETAB_SOLVED_MODELS`. `results/` now holds ONLY the 35 benchmark-model
  `*_results.txt`; `scratch_tests/` renamed to `debugging/`, run logs moved to `examples/debugging/logs/`;
  the Bachmann-K=3 scratch test (`benchmark_bachmann_k3.jl` + its result/log) moved to `examples/debugging/`
  + `debugging/results/`. Solve wall-time cap also lowered 24h→**2h** (7200s) — see Critical harness notes.
- **NEEDS A FRESH BENCHMARK RUN.** Fix #10 changes the observable/IC transcription (ov/zN); the result
  files predate it. Re-run `benchmark_examodels.sh` and re-run `validate_events.jl <Model>` for the event
  models (whose solves drifted off θ* on the OLD objective).
- Detailed change history lives in git, not here — `git log`/`git show <sha>` rather than re-narrating.

---

## Fixes implemented (#1–#10, all committed on `main`)

### 1. x0SSpre (steady-state pre-equilibration)
- `initialize.jl _solve_conds` / `_get_zss_init`: pre-eq models reject plain condition ids in
  `PEtab.get_odeproblem`; must pass the documented **Pair `pre_eq_id => sim_id`**. That call
  pre-equilibrates internally and returns an ODEProblem whose **`u0` IS the steady-state IC** — read
  `u0[1:Nz]` for `zss` init (no separate SteadyStateProblem).
- `utils.jl _get_dict_cidx_sscidx`: was declared `::Dict` but built a `Vector` → now `Vector{Int}`, nothing-guarded.
- `continuity.jl` x0SSpre block: `fs, has_t = _get_rhs_funcs(...)`; `itr_ss1 = (v,cidx)`; constraint
  `z[v,1,0,cidx]=zss[v,cidx]`; residual `f(zss[:,cidx]; cv[:,sscidx])=0` under the PRE-EQ condition's cv; `has_t`→t=0.

### 2. SBML assignment-rule substitution
Models referencing SBML assignment rules (`MTK.observed`, e.g. `total_pop=Σstates`, `pY1173=Σspecies/c1`)
in the RHS or observables leaked unbound symbols into `build_function` → `UndefVarError`.
- `utils.jl _assignment_substitutor(PEprob; bare)` (NEW): substitution dict from all non-state
  `MTK.observed` rules, applied to a **fixpoint** (re-substitutes while any rule symbol remains in
  `get_variables(expr)`; uses `isequal`-membership since Symbolics `==` returns a symbolic eq, not Bool;
  capped at 100 iters vs cycles). `bare=false` keeps MTK `(t)` form (RHS); `bare=true` strips `(t)`
  (objective's bare-symbol parsing of table formulas).
- `utils.jl _get_rhs_funcs`: uses the fixpoint `subst_rules` then `dict_fixed_val` (resolves nested
  rules + constant params introduced by rules, e.g. Laske `D_rib=160`).
- `objective.jl _create_objective`: `apply_rules = _assignment_substitutor(...; bare=true)` applied to
  parsed observable + noise formulas and the observable used by `_reduce_sigma_to_obs`; computes
  `obs_sym` (rules applied) first, then branches single-state (`zidx !== nothing`) vs expression.

### 3. Condition-id alignment (pre-eq-only conditions)
Zheng has a `preequilibration` conditions-table row that is pre-eq-only (never a simulationConditionId);
`_get_cids` used to return ALL rows → `_get_z_init` tried `sol[:preequilibration]` → `KeyError`. Fix =
make the canonical `cidx` dimension = the SIMULATION conditions.
- `utils.jl _get_cids`: returns conditionIds appearing as a `simulationConditionId` in measurements, in
  table order (unchanged for non-pre-eq models).
- `utils.jl _get_cond_rows` (NEW): cidx → conditions-table row index (since `_get_cids` may be a subset).
- `variables.jl _create_cv` + `collocation.jl`: read `conditions_df[cond_rows[cidx], col]`.
- `_get_dict_cidx_sscidx`: falls back to cidx if a sim condition's pre-eq condition isn't in `_get_cids`.

### 4. Heterogeneous-span mesh — "Option A" (VALIDATED; kept design — do NOT refactor to per-condition h[i,cidx])
Conditions with different time spans (Zhao: 6/14/24/39). The shared mesh built from
`argmax(length(sol.t))` could top out at a short condition's tmax and miss later measurement times →
`_get_dict_t_tidx` `findfirst→nothing` → `Nothing→Int64`.
- `initialize.jl _solve_conds`: integrate EVERY condition over the full span
  `[tstart, max(t_meas)]` via `ODE.remake(odesys; tspan=(tstart, max(maximum(tstops), tspan[2])))`.
- `initialize.jl _get_z_init`: `argmax(length(sol.t))` mesh + plain `sol[cid](t)` (valid now that all
  conditions span the full mesh).
- Decision recorded in the `project-heterogeneous-span-decision` memory: keep the shared mesh.
  NOT-yet-objective-checked heterogeneous models: Lucarelli, Schwen, Laske, Bachmann.

### 5. Conditions-table initial-state override in IC constraints (`95e7729`)
A conditions-table column whose header matches a STATE id sets that state's IC per condition (PEtab
spec), but this is NOT in the SBML/MTK speciemap, so `continuity.jl _create_initial_conditions` took the
speciemap default for every condition → forced all conditions to the same wrong IC (Armistead: mutant
condition forced to wild_type ICs). **Invisible to the nominal-θ objective check** — the warm-start z/cv
were correct, only the IC *constraint* was wrong. **Fix:** detect states whose name (z_sym minus `(t)`)
matches a cv column (`_get_cv_colnames`) and route their IC to `z[v,1,0,cidx]=cv[cvidx,cidx]`.
Blast radius: only Armistead in the 19-set (no-op without a state/cv-column name match).

### 6. initialAssignment-defined parameters frozen at defaults (`95e7729`)
A PARAMETER defined by an SBML `initialAssignment` (Bertozzi `beta_N = R0_*gamma_/N_`) is FROZEN by PEtab
at the value with DEFAULT params (`0.1*0.1/1=0.01`), even though condition overrides apply to
`R0_/gamma_/N_`. `_get_rhs_funcs`'s `dict_fixed_val` kept `beta_N` symbolic → re-evaluated with
per-condition cv (4.86e-9) → RHS off ~2.6e6× → warm start ~100% collocation-infeasible → MadNLP stalled.
**Fix:** new `_resolve_fixed_vals(PEmodel, PEprob)` fixpoint-substitutes the parametermap into itself
until every default is numeric, returns fixed (non-estimated, non-cv) params as numeric constants (falls
back to symbolic if irreducible). Used in `_get_rhs_funcs` and the `continuity.jl` IC fixed-constant
substitution. Blast radius: only Bertozzi (Laske's param initialAssignments reference fixed constants →
identical numeric values, neutral).

### 7. Fixed-time event support (piecewise(time) gates) — COMMITTED `7454a85` (merged to `main` 2026-06-05)
Adds the 6 fixed-time event models (Oliveira, Fujita, Giordano + pre-eq Brannmark, Isensee, Raimundez)
to `ALL_MODELS` (18 → 24; **26 total** incl. Bruno + Crauste). All 6 have petab_optimum_found=true.

- **Mechanism.** For these models `PEprob.model_info.model.petab_events` is EMPTY — SBMLImporter folds
  every `piecewise(time>T, a, b)` into the RHS as the smooth form `b*a+(1-b)*c`, where `b =
  __parameter_ifelseN` is an MTK parameter toggled by an init/discrete callback at the fixed time T.
  No `ifelse` survives (ExaModels-AD non-issue) and NO state jump (continuity unchanged). The pre-existing
  bug: `_resolve_fixed_vals` froze those gates at the parametermap DEFAULT (pre-trigger value) instead of
  the value PEtab simulates with.
- **Design (tuple-data, src changes).** `utils.jl _get_gate_syms` (NEW) finds the `__parameter_ifelse*`
  params; `_get_rhs_funcs(PEmodel, PEprob, gate_syms)` keeps them as TRAILING ARGS of one f per state
  (no per-pattern build_function) AND now always appends `t` as the final arg (autonomous RHS ignores the
  unused arg) so `has_t` is gone everywhere. `utils.jl _get_gate_vals` (NEW) reads each gate's per-
  (interval,condition) value from a STEPPED PEtab integrator (`integ.ps[gate]`, post-init — the bare
  `oprob.ps[gate]` carries the stale pre-trigger default; `ODE.getp`/`SavingCallback` are unavailable);
  asserts every toggle lands on a mesh node; `_get_gate_vals_ss` (NEW) for the pure-SS path. `collocation.jl
  _create_lagrange` carries `gate_vals[:,i,cidx]` in the iterator tuple as an NTuple field `gv` (like
  `h[i]`/`t_ij`) and splats it via `ntuple(g->gv[g],Ng)...` (ExaModels: you may `getindex` a tuple data
  field but NOT splat it). `continuity.jl _add_ss_residual` and `steadystate.jl _ss_conservation/_create_
  variables_ss` use the same idiom (residual/IC gate = the t=0 IC value `g0`, verified `‖f(zss_init;g0)‖≈0`).
  `structs.jl PEInfo` gained `gate_syms`/`gate_vals`/`gate_vals_ss`. `Ng==0` ⇒ `gv=()` ⇒ non-event models
  byte-identical.
- **Validation (GPU, K=5).** Non-event regression byte-identical (Boehm 138.22083 pre==post). Event
  transcription reproduces PEtab's EXACT objective at MadNLP **iter 0** (Fujita −53.084, Brannmark 141.889)
  with good warm-start feasibility — gates/collocation/continuity/pre-eq are correct at θ*.
- **Open blocker (NOT events).** Both event models then drift OFF θ* during the solve (Fujita → −323 lower
  feasible obj; Brannmark diverges). Root cause is the unvalidated `objective.jl` observable/noise handling
  for these models (Oliveira/Giordano don't even finish building — `sd_cumulative_cases`,
  `CurrentDiagnosedInfected`), which is the OTHER session's `ov-work` (build_function blowup + observable-
  rule resolution). **Merge `ov-work`'s objective.jl into this branch, then re-run
  `examples/debugging/validate_events.jl <Model>` to confirm end-to-end convergence.**
- **Out of scope (still):** Beer (event TIME is an estimated param), Liu (state-triggered `U<1e-8`), Smith
  (true state-jump events; also petab opt_status=false).

### 8. Steady-state (no-mesh) build path → Blasi resolved (`dc1b295`)
Steady-state models (all measurements at t=∞) hit the mesh-indexing `Nothing→Int64` because they have no
time-course mesh. **Fix:** `build_model` branches on `ExaModelsPEtab._is_steady_state(PEmodel)` to
`_create_variables_ss` / `_create_ss_constraints` / `_create_objective_ss` (uses `zss`, steady-state
residual + conservation, no collocation mesh). Blasi now `SOLVE_SUCCEEDED` −1090.91 vs −1090.56 (−0.03%).

### 9. Benchmark warmup restructure — Bruno is the shared JIT warmup (`08273a3`)
Old design warmed examodels on Crauste and petab on Bruno, but Bruno wasn't excluded from the petab loop →
its `petab_compile_time` was pre-warmed garbage (4.96 s vs ~225 s real). Now BOTH `benchmark_examodels.jl`
and `benchmark_petab.jl` warm on **Bruno** and exclude it from their timed lists; **`benchmark_bruno.jl`**
(Crauste-warmed) benchmarks Bruno itself with both backends; old `benchmark_crauste.jl` deleted.
`benchmark_petab.jl` `COMPILE_LIMIT` is env-overridable via `PETAB_COMPILE_LIMIT` (default 1800 s).
`benchmark_bachmann_k3.jl` (NEW) = the one-off Bachmann K=3 OOM-fit test → separate `*_K3_results.txt`.

### 10. Observable build_function blowups — `ov` (assignment-rule binding) + `zN` (final-time aux) (`399fb43`)
Two build blowups, both from `build_function` expanding complex SBML expressions into dense-Hessian
ExaModels kernels (all in `objective.jl`; lazy, so unaffected models are byte-identical):
- **`ov` (observed variables) — COMPILE blowup.** SBML assignment rules used in observable/noise formulas
  (`MTK.observed`; e.g. SalazarCavazos `EGFRtot=Σ72 species` inlined into 4 ratio observables) were
  inlined and re-differentiated into every formula → `build_function` codegen explosion. Now each occurring
  FLAT rule is bound to an auxiliary variable `ov[rule,node]` (defined once per evaluation node via the
  `@add_con`/`@add_con!` idiom over `_rule_table`), and the formula references that single leaf — ExaModels
  never re-expands it. SalazarCavazos `_create_objective` **2711s→67s** (K=2); objective matches PEtab nllh.
  Nested rules (rule-of-rule) fall back to inlining. (`cse=true` in `build_function` does NOT help — ExaModels
  traces through the compiled fn and expands the let-bindings; the aux-variable leaf is what works.)
- **`zN` (final-time endpoint state) — SOLVE-time kernel-param-memory.** `idx==N` observable/noise inlined the
  `K+1`-term L1 endpoint sum `Σ_j L1[j+1]·z[v,N,j,cidx]` into the nonlinear obs/noise fn, blowing the CUDA
  kernel parameter budget (Lucarelli ratio observables: 35.3 KiB at K=6). Now one aux var `zN[v,col]` is bound
  to that linear sum (base+aug constraint) and the single var is fed in — mirroring how `idx<N` feeds the
  single node `z[v,idx+1,0,cidx]`. **Lucarelli K=6 (nvar=835588): max kernel 35.3→5.5 KiB, zero oversized
  layers** (was a hard-fail; now builds).
- Both reformulations are mathematically identical and warm-start-feasible by construction; Boehm/Fujita
  regression objective-match unchanged after merge.

### NOT supported: very large arbitrary-function ICs (`z0_func`) — Fiedler (accepted)
Fiedler's `initialAssignment`s give each non-input state a ~130-op rational closed form of estimated params
(the drug-free steady state). Tracing that into the IC `@add_con` (`continuity.jl` `itr_z0_func`) builds a
dense-Hessian kernel that overflows GPU param memory (36.8 KiB) and takes 40min+ of LLVM. **No other model
exercises the `z0_func` (arbitrary-function IC) path** — all solved models have fixed/param/cv ICs — so it is
effectively untested and Fiedler is its only user. Investigated and DEFERRED: `cse=true` refuted; a
steady-state-residual reroute (the IC *is* a SS) is complicated by the drugs being states with per-condition
overrides; a CSE-to-aux-variable decomposition is possible but not worth it for one model. **Accepted as
unsupported for now** — ExaModels just can't take this one very large fused expression.

---

## Known-failing / hard models (diagnosed root causes — durable; live status in result files)
These are the diagnosed blockers, not a live snapshot. The *current* pass/fail of every model is in
`results/*_results.txt`.

| Model | Failure | Root cause / status |
|---|---|---|
| ~~Blasi~~ | ~~compile `Nothing→Int64`~~ | **RESOLVED** (Fix #8, steady-state no-mesh path) — now `SOLVE_SUCCEEDED` −1090.91 vs −1090.56 (−0.03%). |
| **Bachmann** | GPU OOM @ K=6 (4.10M vars, needed ~40 GB) | **memory-only** — verified: **K=3 FITS** (2.34M vars, ~23 GB, compiles 1128 s; `Bachmann_MSB2011_K3_results.txt`). For a real result use A100/H100 80 GB @ K=6. (K=3 *solve* is non-convergent — a separate conditioning issue.) |
| ~~Lucarelli~~ | ~~CUDA kernel param-memory overflow (35.3 KiB)~~ | **RESOLVED** (Fix #10, `zN`) — `idx==N` ratio-observable inlined the L1 endpoint sum; K=6 max kernel 35.3→5.5 KiB, builds. |
| **Fiedler** | CUDA kernel param-memory overflow (36.8 KiB) | **Accepted unsupported** (Fix #10 note) — the IC is a ~130-op `z0_func` rational; the only model on that untested path. NOT the sm_70 limit per se; even split it's an enormous fused IC expression. |
| **Borghans / Elowitz** | solve diverges to all-NaN spin | infeasible collocation warm-start; killed/skipped. **A NaN/stall early-abort + ‖c(x₀)‖∞ diagnosis is the fix.** |
| **Zheng** | ill-conditioned non-convergence | primal-feasible but dual-infeasible stuck (inf_du~0.26, lg_rg maxed = singular KKT); obj wandered −278→−529 (not a real optimum). Likely **unidentifiable params / scaling**. |
| ~~SalazarCavazos~~ | ~~compile blowup (~10×+ PEtab)~~ | **RESOLVED** (Fix #10, `ov`) — assignment-rule observable expansion; `_create_objective` 2711s→67s. (RHS-codegen cost remains but is under budget.) |

### Benchmark campaign results (20-model snapshot, closed 2026-06-05; non-event subset)
Point-in-time closeout for the 19 non-event + Crauste. **9 clean ✅ · 4 converged-marginal ⚠️ · 7 hard-fail 🔴.**
Once compiled, ExaModels' warm SGM solve is **6–60× faster** than PEtab/IPNewton, matching obj to ≤2% on the clean set.
- **✅ Clean (9, warm speedup):** Crauste 60×, Bruno 57×, Blasi 35×, Rahman 26×, Boehm 23×, Armistead 18×, Perelson 14×, Sneyd 6×, Okuonghae 2× (+4.6% high).
- **⚠️ Marginal (4):** Schwen (skipped; converges to RIGHT answer but stalls above dual tol → needs looser tol), Laske (RESTORATION_FAILED +3.6%), Zhao (RESTORATION_FAILED +8.2%, SGM skipped), Bertozzi (SOLVE_SUCCEEDED **+194%** worse local min).
- **🔴 Hard-fail (7 at campaign close; Fix #10 since resolved SalazarCavazos compile + Lucarelli build →
  pending re-run):** Bachmann, Fiedler, Lucarelli, Borghans, Elowitz, Zheng, SalazarCavazos (table above).
- **Event models (6, Fix #7):** transcription reproduces PEtab @ iter 0; solves drifted off θ* on the OLD
  objective — Fix #10 (`ov`) just merged should unblock them; **re-run `validate_events.jl` to confirm**.

**Suboptimality is itself an error class:** a `SOLVE_SUCCEEDED` on a HIGHER (worse) nllh than PEtab is a
correctness flag, not a pass. Judge by DIRECTION: exa *lower* (Armistead −306 vs −302) is usually K=6
discretization; exa *higher* (Bertozzi, Okuonghae, Laske, Zhao) is genuine suboptimal convergence.

**Event models — fixed-time NOW SUPPORTED (Fix #7).** The 6 fixed-time `piecewise(time>T)` models
(Oliveira, Fujita, Giordano, Brannmark, Isensee, Raimundez) are handled by the gate transcription and are
in `ALL_MODELS`. STILL out of scope: **Beer** (event time is an estimated parameter — freeze-at-nominal
invalid), **Liu** (state-triggered `U<1e-8`), **Smith** (true SBML state-jump `<event>`s — collocation
can't represent the discontinuity; also petab opt_status=false). `has_events` over-flags (callback-based);
true SBML `<event>` models are only Liu + Smith (`grep -c '<event[ >]' <model>/*.xml`).

---

## Diagnostic methodology (how the #5/#6 constraint bugs were found)
The nominal-θ objective check (below) does NOT exercise the ODE RHS or IC/continuity constraints — it
only uses warm-start z at measurement nodes + observable/noise formulas — so it passed cleanly for both
buggy models. The bugs showed up only in the **solve** and the **warm-start constraint violation**:
1. **‖c(x₀)‖∞ split by block.** Unscaled ∞-norm of the equality-constraint residual at `m.meta.x0`,
   split into collocation / cv / interval-continuity / IC / y+σ to localize. Armistead → all in IC;
   Bertozzi → all in collocation. (`examples/debugging/warm_start.jl <Model> <K>`.)
2. **Real RHS bug or mesh coarseness?** Compare our `_get_rhs_funcs` f to PEtab's actual RHS
   (`oprob,_=PEtab.get_odeproblem(θ,PEprob;condition=cid); oprob.f(du,u0,oprob.p,0.0)` vs `fs[v](...)`;
   ratio should be 1.0). Manual collocation residual: A=`Σ_j z[v,i,j,cidx]·DLDTAU[j+1,k]` vs B=`h[i]·f(...)`
   — if A≫B at a node where `z_init==sol(t_node)` exactly, the RHS is wrong (raising K won't help;
   Bertozzi K=6==K=12).
3. **Is the warm-start z correct?** `max|z_init − sol[cid](t_node)|` over all nodes (0.0 for Bertozzi).

## Validation methodology — "matches PEtab to <tol>"
Fixed-point **objective-consistency** check at the NOMINAL parameters (not a solve):
```julia
PEp   = PEtab.PEtabODEProblem(PEtab.PEtabModel(yaml))
petab = PEp.nllh(PEtab.get_x(PEp))                  # PEtab nllh @ nominal θ
m     = petab_examodel(yaml; backend=nothing, K=5)  # our collocation NLP (CPU; fast)
exa   = ExaModels.NLPModels.obj(m, m.meta.x0)       # our objective at the warm start
reldiff = abs(exa-petab)/abs(petab)                 # ~1e-7 = K=5 discretization + roundoff
```
Both numbers are the likelihood at nominal θ via independent code paths, so small reldiff means our
objective formulation (observables, σ, assignment rules, condition indexing, mesh, state-at-measurement
mapping) reproduces PEtab. Use `backend=nothing` (CPU). Tool: `examples/debugging/verify_models.jl`.

⚠️ **Necessary but NOT sufficient** — passed to ~1e-11 for BOTH Armistead and Bertozzi while each had a
constraint-generation bug (the check never touches the RHS or IC/continuity constraints). Always pair it
with the warm-start constraint-violation check above.

### Validation results (objective vs PEtab @ nominal, K=5, CPU)
| Model | feature exercised | reldiff |
|---|---|---|
| Crauste | control (non-pre-eq) | 4.2e-10 ✅ |
| Brannmark | pre-eq (x0SSpre) + heterogeneous-span (Option A) | 8.4e-9 ✅ |
| Zheng | pre-eq-only condition alignment | 9.7e-8 ✅ |
| Rahman | nested RHS assignment rule | 3.6e-11 ✅ |
| Okuonghae | `total_pop=Σstates` rule in RHS | 1.9e-9 ✅ |
| SalazarCavazos | `pY1173` rule in observable | matched |
| Zhao | heterogeneous-span mesh | 1.15e-7 ✅ (re-confirmed under Option A) |
| Laske | `D_rib=160` const-param via rule | builds + resolves; compile VERY slow (~30 min, 29 rules) |

---

## Benchmark infrastructure (examples/Benchmarks/)
Results in `results/`, combined key=value `{Model}_results.txt` per model (petab_* + exa_* coexist).

### Scripts
- `benchmark_petab.jl / .sh` — one Julia process per model; builds PEtabODEProblem, solves with
  Optim.IPNewton(), 1-hr cap. Writes `petab_*`. `bash examples/Benchmarks/benchmark_petab.sh` from root.
- `benchmark_examodels.jl / .sh` — long-lived Julia process(es), strided over the 24 `ALL_MODELS`
  (18 non-event + 6 fixed-time event, Fix #7). Warmup = **Bruno** (Fix #9; excluded from ALL_MODELS,
  pre-warms generic JIT). K=6, compile deadline 4 hr. Writes `exa_*`.
  `bash examples/Benchmarks/benchmark_examodels.sh` from root (**no trailing `&`**).
- `benchmark_bruno.jl` (Crauste-warmed, benchmarks Bruno both backends) — see Fix #9. (The Bachmann K=3
  OOM-fit scratch test moved to `examples/debugging/benchmark_bachmann_k3.jl`, writing to
  `debugging/results/Bachmann_MSB2011_K3_results.txt`.)
- `list_benchmarks.jl` — canonical BENCHMARK_MODELS (35) / PETAB_SOLVED_MODELS (28) / EXA_SUPPORTED_MODELS (26),
  included by benchmark_examodels.jl, benchmark_petab.jl, and final_report.jl.
- `final_report.jl [--sgm]` — prints/writes `final_report.txt`; `--sgm` uses SGM solve times.

### Per-model examodels sequence
1. **Warmup** (Bruno, once per process; Fix #9).
2. **Compile** — Phase 1 PEtab setup + ODE presolve (`exa_presolve_time`); Phase 2 Symbolics codegen +
   ExaModels build (`exa_compile_time` = Phase1+2).
3. **First solve** — MadNLP/cuDSS; includes GPU kernel JIT (`exa_solve_time`, inflated, not comparable).
4. **SGM solves** — `madnlp(model)` ×N_SGM_RERUNS=3 on the same in-memory model (kernels cached, x0 not
   mutated). Geometric mean → `exa_sgm_solve_time` (comparable to `petab_solve_time`).

### Result keys
`exa_compile_time`/`exa_presolve_time`, `exa_solve_time`, `exa_sgm_solve_time`/`exa_sgm_n`,
`exa_compile_status`, `exa_solve_status`, `exa_sgm_status`, `exa_term_status`, `exa_objective`,
`exa_iter`, `exa_nvar`, `exa_ncon`, `exa_error`. `%EXA = (compile−presolve)/compile×100`.

### Models benchmarked (26 total pipeline = 24 `ALL_MODELS` + Bruno + Crauste)
**Canonical lists now live in `list_benchmarks.jl`** (`EXA_SUPPORTED_MODELS` = these 26).
**Bruno is now the JIT warmup** (Fix #9), so it's excluded from the timed `ALL_MODELS` loop and
benchmarked via `benchmark_bruno.jl`; **Crauste** is also benchmarked outside the loop (its data captured).
**Non-event (19, original)** — petab_optimum_found=true AND petab_has_events=false:
Armistead, Bachmann, Bertozzi, Blasi, Boehm, Borghans, Bruno, Elowitz, Fiedler, Laske, Lucarelli,
Okuonghae, Perelson, Rahman, SalazarCavazos, Schwen, Sneyd, Zhao, Zheng.
**Fixed-time event (6, NEW — Fix #7)** — petab_optimum_found=true, piecewise(time) gates:
Oliveira, Fujita, Giordano, Brannmark (pre-eq), Isensee (pre-eq), Raimundez (pre-eq).
(`ALL_MODELS` = the 18 non-event in-loop + these 6 = 24; Bruno + Crauste benchmarked separately.)

### Critical harness notes
- **Run julia at `-t 1`**; enforce timeouts with external `bash -c "sleep N; kill -9 <pid>"`. A Julia
  watchdog thread starves cuDSS (20 s solve → 690 s observed).
- **Launch without trailing `&`** with `run_in_background` (script uses `wait`).
- Resumable: `compiling`/`solving`/`exa_sgm_status=running` sentinels → `timeout`/`interrupted` on
  restart; `exa_finished()` re-queues anything not terminal.
- Re-run a model from scratch: strip its `exa_*` keys, keep `petab_*`:
  `grep "^petab_" file > tmp && mv tmp file`.
- A diverging (NaN) solve relies on MadNLP `max_wall_time` + external SIGKILL + auto-resume; with
  `MAX_ITER=1e8` the wall clock is the only real bound. `SOLVE_LIMIT` lowered 86400→**7200 (2 hr)**
  on 2026-06-05 — the longest TRUE successful solve was Lucarelli/PEtab ~2676 s (0.74 hr) and exa Boehm
  ~264 s, so 2 hr caps only diverging/NaN-spin solves (≈12× faster to abandon a dead run than the old 24h).

### Scratch tests (examples/debugging/)
`verify_models.jl` (objective-consistency, main validation tool), `warm_start.jl` (warm-start violation),
`gpu_run.jl` (end-to-end GPU sanity), `sigma_constraints.jl`, `stage_build.jl`.
Event support (Fix #7): `validate_events.jl <Model>` (GPU build+solve, compares to PEtab optimum),
`diag_gates.jl <Model>` (prints gate_syms + per-(interval,condition) gate patterns + gate_vals_ss),
`build_smoke.jl <Model>` (fast CPU build-only smoke), `probe_gates.jl` / `probe_preeq.jl` /
`probe_tupledata.jl` (one-off probes for the gate source and ExaModels tuple-data behavior).
Blowup diagnosis (Fix #10): `stage_build_timed.jl [K] <Model>` (per-stage compile timing — localizes a
codegen blowup to collocation vs objective; K-independent) and `diag_layers.jl <Model> [K]` (per-constraint
SIMDFunction comp1/comp2 lengths — finds the layer whose Hessian compressor overflows the GPU kernel param
limit; do NOT add an expression-stringify, it hangs on the giant exprs).

## PEtab optimizer findings
`docs/.../optimizers.md`: small → `Optim.IPNewton()` + exact Hessian; medium → `Fides.CustomHessian()`+GN;
large → (L)BFGS. **LBFGS does NOT certify convergence** on these stiff problems (objective converges,
gradient won't hit tol → status -5, runs to cap). **IPNewton converges in a few iters** with the exact/GN
Hessian — chosen. `res.converged` is Bool for Optim, an Int status for Ipopt. Ipopt added this work
(`Pkg.add("Ipopt")`); Fides.jl deliberately NOT added (wraps a heavy/fragile Python pkg).

## Environment
- Julia 1.12.6; `julia --project=.` at package root. 96 CPU cores, 376 GB RAM.
- 2× Quadro GV100 (32 GB). **`nvidia-smi` is broken** (NVML version mismatch) but **CUDA.jl is
  functional**. Check per-device free memory via CUDA (`for d in CUDA.devices(); CUDA.device!(d);
  CUDA.available_memory(); end`), not nvidia-smi.
- **SHARED BOX**: `sushin` also runs CUDA jobs (steady tenant on GPU 0 — they default to device 0).
  Prefer GPU 1 for ad-hoc runs; check free memory before a full run. Contention isn't a correctness
  problem (GPUs time-slice) but inflates solve times and can tip the big models (Bachmann) into OOM.

## Outstanding / next steps (post-campaign, 2026-06-05)
1. **RE-RUN THE BENCHMARK with the new objective (Fix #10 just merged).** The `exa_*` result files predate
   `ov`/`zN`, so the campaign snapshot below is stale for the affected models. Re-run
   `benchmark_examodels.sh` (SalazarCavazos should now compile; Lucarelli should now build) and re-run
   `validate_events.jl <Model>` for the 6 event models (their solves drifted off θ* on the OLD objective —
   the integration is verified at iter 0; confirm end-to-end convergence now). Regression-re-check an
   assignment-rule model (Rahman/Okuonghae) + Boehm (objective-match unchanged after merge, but confirm at K=6).
2. **Solver-conditioning — the recurring wall on big/stiff models.** Related symptoms:
   NaN-spin (Borghans, Elowitz → NaN/stall early-abort + ‖c(x₀)‖∞ diagnosis); tail-stall above tol
   (Schwen → looser/adaptive dual tol); RESTORATION_FAILED spiral (Laske, Zhao, Bachmann-K3 → better
   warm start); singular-KKT (Zheng → check unidentifiable params / scaling).
3. **Bigger GPU** — Bachmann (proven K=3 fits 23 GB → K=6 needs 80 GB A100/H100). Hardware blocker, not a bug.
   (Lucarelli's kernel-param overflow is RESOLVED by Fix #10 `zN`; Fiedler's is the accepted-unsupported
   `z0_func` IC, not a GPU/size issue.)
4. **Bertozzi local-min gap** (+194%) — multistart / better warm start if matching PEtab's optimum matters.
5. **Mesh design** — Option A (shared mesh) is the KEPT decision; per-condition ragged meshes only if a
   short condition's ODE ever diverges past its window (not observed).
- *(DONE this campaign: Blasi → Fix #8; Bachmann K-fit test → K=3 fits (memory-only); warmup → Fix #9.)*
- *(Non-canonical PEtab reruns: Chen/Froehlich compile intractable on this CPU (capped ~19.6 h); Lang
  compiled in 2.65 h but solve errored. Out of the canonical set.)*
