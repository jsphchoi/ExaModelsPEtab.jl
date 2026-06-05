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
- Fixes #1–#6 (below) are all COMMITTED. The two constraint-generation bugs (#5 Armistead, #6 Bertozzi)
  landed in `95e7729` — its commit message has the full root-cause writeup. Benchmark scripts (K=6, SGM
  reruns, Crauste warmup) are committed.
- **Fix #7 (fixed-time events) is UNCOMMITTED**, on branch **`events-fixed-time`** in a separate worktree
  `~/.julia/dev/ExaModelsPEtab-events` (isolated from a concurrent session editing `objective.jl` on
  `ov-work`). Touches `src/{utils,collocation,continuity,steadystate,variables,structs}.jl` +
  `examples/Benchmarks/benchmark_examodels.jl` (ALL_MODELS). To merge: bring in `ov-work`'s `objective.jl`
  first, then this branch; the two are complementary (events vs observable/blowup) and edit disjoint files
  except neither touches the other's. Tests for #7 live in `examples/scratch_tests/` (this worktree).
- Detailed change history lives in git, not here — `git log`/`git show <sha>` rather than re-narrating.

---

## Fixes implemented (#1–#6, all committed)

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

### 7. Fixed-time event support (piecewise(time) gates) — NOT yet committed; branch `events-fixed-time` / worktree `~/.julia/dev/ExaModelsPEtab-events`
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
  `examples/scratch_tests/validate_events.jl <Model>` to confirm end-to-end convergence.**
- **Out of scope (still):** Beer (event TIME is an estimated param), Liu (state-triggered `U<1e-8`), Smith
  (true state-jump events; also petab opt_status=false).

---

## Known-failing / hard models (diagnosed root causes — durable; live status in result files)
These are the diagnosed blockers, not a live snapshot. The *current* pass/fail of every model is in
`results/*_results.txt`.

| Model | Failure | Root cause / status |
|---|---|---|
| **Blasi** | compile error `Nothing→Int64` | mesh-indexing (`_get_dict_t_tidx` `findfirst→nothing`); NOT fixed by #4/#5/#6 — needs its own fix |
| **Bachmann** | GPU OOM (8.945 GiB on full GV100) | biggest model; hardware capacity, not a bug. Try alone at K=6 + `CUDA.reclaim()` between models; fall back to K=4 |
| **Fiedler / Lucarelli** | CUDA kernel param-memory overflow (36.8 / 35.3 KiB > 31.996 KiB) | ExaModels fuses one oversized expr; structural sm_70 (GV100=cc7.0) limit. Split the fused expr, or run on sm_80+ |
| **Borghans / Elowitz** | solve diverges to all-NaN spin | starts clean then all-NaN; external watchdog SIGKILLs + auto-resumes (loses ~one watchdog interval). Likely infeasible collocation warm-start — investigate ‖c(x₀)‖∞ per block. **A NaN/stall early-abort is the real fix.** (Being worked in a parallel session — don't kill the live run.) |
| **Bertozzi** | converged but suboptimal | `SOLVE_SUCCEEDED` 5.708e9 vs PEtab 1.94e9 (~2.9× worse nllh) — a different/worse local min from the same nominal start. Multistart / warm-start concern, NOT a build bug. |

**Suboptimality is itself an error class:** a `SOLVE_SUCCEEDED` landing on a HIGHER (worse) nllh than
PEtab is a potential correctness flag, not a pass. Caveat — the K=6 collocation objective ≠ continuous
nllh, so judge by DIRECTION: exa *lower* (Armistead −306.13 vs −301.92) is usually discretization; exa
*higher* (Bertozzi) is genuine suboptimal convergence. `final_report.jl` prints the rel.gap column.

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
   Bertozzi → all in collocation. (`examples/scratch_tests/warm_start.jl <Model> <K>`.)
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
mapping) reproduces PEtab. Use `backend=nothing` (CPU). Tool: `examples/scratch_tests/verify_models.jl`.

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
  (18 non-event + 6 fixed-time event, Fix #7). Warmup =
  **Crauste** (excluded from ALL_MODELS; pre-warms generic JIT, its timing is invalid). K=6, compile
  deadline 4 hr. Writes `exa_*`. `bash examples/Benchmarks/benchmark_examodels.sh` from root
  (**no trailing `&`** — the script uses `wait`; the harness must track the outer process).
- `final_report.jl [--sgm]` — prints/writes `final_report.txt`; `--sgm` uses SGM solve times.

### Per-model examodels sequence
1. **Warmup** (Crauste, once per process).
2. **Compile** — Phase 1 PEtab setup + ODE presolve (`exa_presolve_time`); Phase 2 Symbolics codegen +
   ExaModels build (`exa_compile_time` = Phase1+2).
3. **First solve** — MadNLP/cuDSS; includes GPU kernel JIT (`exa_solve_time`, inflated, not comparable).
4. **SGM solves** — `madnlp(model)` ×N_SGM_RERUNS=3 on the same in-memory model (kernels cached, x0 not
   mutated). Geometric mean → `exa_sgm_solve_time` (comparable to `petab_solve_time`).

### Result keys
`exa_compile_time`/`exa_presolve_time`, `exa_solve_time`, `exa_sgm_solve_time`/`exa_sgm_n`,
`exa_compile_status`, `exa_solve_status`, `exa_sgm_status`, `exa_term_status`, `exa_objective`,
`exa_iter`, `exa_nvar`, `exa_ncon`, `exa_error`. `%EXA = (compile−presolve)/compile×100`.

### Models benchmarked (25 = 24 `ALL_MODELS` + Bruno; Crauste excluded as warmup)
**Non-event (19, original)** — petab_optimum_found=true AND petab_has_events=false:
Armistead, Bachmann, Bertozzi, Blasi, Boehm, Borghans, Bruno, Elowitz, Fiedler, Laske, Lucarelli,
Okuonghae, Perelson, Rahman, SalazarCavazos, Schwen, Sneyd, Zhao, Zheng.
**Fixed-time event (6, NEW — Fix #7)** — petab_optimum_found=true, piecewise(time) gates:
Oliveira, Fujita, Giordano, Brannmark (pre-eq), Isensee (pre-eq), Raimundez (pre-eq).
(`ALL_MODELS` = the 18 non-event in-list + these 6 = 24; Bruno benchmarked separately; total pipeline 26.)

### Critical harness notes
- **Run julia at `-t 1`**; enforce timeouts with external `bash -c "sleep N; kill -9 <pid>"`. A Julia
  watchdog thread starves cuDSS (20 s solve → 690 s observed).
- **Launch without trailing `&`** with `run_in_background` (script uses `wait`).
- Resumable: `compiling`/`solving`/`exa_sgm_status=running` sentinels → `timeout`/`interrupted` on
  restart; `exa_finished()` re-queues anything not terminal.
- Re-run a model from scratch: strip its `exa_*` keys, keep `petab_*`:
  `grep "^petab_" file > tmp && mv tmp file`.
- A diverging (NaN) solve currently relies on the external SIGKILL + auto-resume; `MAX_ITER=1e8` +
  `SOLVE_LIMIT=86400` means without that guard a dead solve would burn ~24h.

### Scratch tests (examples/scratch_tests/)
`verify_models.jl` (objective-consistency, main validation tool), `warm_start.jl` (warm-start violation),
`gpu_run.jl` (end-to-end GPU sanity), `sigma_constraints.jl`, `stage_build.jl`.
Event support (Fix #7): `validate_events.jl <Model>` (GPU build+solve, compares to PEtab optimum),
`diag_gates.jl <Model>` (prints gate_syms + per-(interval,condition) gate patterns + gate_vals_ss),
`build_smoke.jl <Model>` (fast CPU build-only smoke), `probe_preeq.jl` / `probe_tupledata.jl` (one-off
probes for the pre-eq gate source and the ExaModels tuple-data-field behavior).

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

## Outstanding / next steps
1. **NaN-spin (Borghans + Elowitz)** — add a NaN/stall early-abort to the solve (vs relying on the
   external SIGKILL) and diagnose the shared infeasible-warm-start root cause (‖c(x₀)‖∞ per block).
   *(being worked in a parallel session)*.
2. **Blasi** — dedicated mesh-indexing fix for the `Nothing→Int64` (`_get_dict_t_tidx`) compile error.
3. **Bachmann** — test alone at K=6 with `CUDA.reclaim()` between models (GPU-mem accumulation vs genuine
   size); fall back to K=4.
4. **Fiedler + Lucarelli** — split the oversized fused expr (<31.996 KiB/kernel) or run on sm_80+.
5. **Bertozzi local-min gap** — multistart / better warm start if matching PEtab's optimum matters; else document.
6. **Regression re-check** an assignment-rule model (Rahman/Okuonghae) + Laske as they cycle through the
   queue (#6 touches the shared `_get_rhs_funcs`/IC path; expected no-op).
7. **Laske compile speed** — 29 nested rules → ~30-min compile; consider CSE/caching if it matters.
8. **Mesh design** — Option A (shared mesh, extends short conditions to max span) is the kept decision;
   per-condition meshes (ragged z, bigger refactor) only if a short condition's ODE ever diverges past
   its window (not yet observed).
