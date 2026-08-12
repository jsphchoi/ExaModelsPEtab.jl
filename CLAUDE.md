# ExaModelsPEtab.jl — working notes, branch `emc`

Research log for `emc`: ExaModelsCollocation (EMC) as the discretization backend, and the
"possible discontinuity" (gate) models. Everything here is measured unless marked otherwise.

Machine and cross-package facts (HSL/MUMPS, dev-links, ExaModels API rules, file style) are in
`~/.julia/dev/CLAUDE.md`. Benchmark settings and paper framing are in
`~/Git/exa-petab-focapo-2027/CLAUDE.md`.

## Constraints

- Gate values live in the iterator when `adaptive = false`; `add_par` prolongs compile time.
- Keep the entry point minimal: `examodel_petab(filename; backend = nothing, K = 4, kwargs...)`,
  `kwargs...` forwarded straight to `CollocationExaCore`. Do not re-add `roots`/`basis`/
  `adaptive_mesh` as named kwargs, and do not restate EMC's Radau/StateForm defaults here.
- `emc` carries the latest package versions; drift from `main` is intentional. Dev-only until
  both packages publish, so CI resolvability does not matter.

## Environment

Isensee is not solvable via MUMPS at all: it builds fine (487 s, nvar 15.5M, all 11 gate events
correct), then wedges in a *single serial* operation — 317 min against a 60 min `max_wall_time`,
one thread busy, RSS 151 GB. MadNLP only checks the wall limit between iterations, so it never
fires. Kill such a run. Diagnose with per-thread ticks in `/proc/PID/task/*/stat`.

## Task order (fixed by the author)

Tasks 1–6 are committed: state reorder `z[v,cidx,i,k]`, EMC swap, GPU check, `GaussRadau`
default, mesh floor, and `piecewise(time)` gate values.

7. **Adaptive mesh refinement on EMC's `set_nodes!` — premise refuted, see below.**
8. Reorder constraint iterators for GPU coalescing — not started.

## How to judge a solve

Compare `exa objective` against `PEtab.nllh(p*)` **at the solver's own p\***, never against the
nominal, and never from a single run:

- The gap is **not monotone in iterations**. Giordano at tol 1e-6: 1000 iters gap 7045,
  3000 gap 113, 12061 gap 5925 — worse than nominal. It reaches a good point near ~3000 and
  wanders back out.
- **GPU solves are nondeterministic.** Two identical Elowitz runs landed 63 objective units
  apart. Use n ≥ 5 per arm, or judge on quantities computed at iteration 0.
- Run the real benchmark config first; the authoritative source is
  `~/Git/exa-petab-focapo-2027/benchmark/options.jl`. Do not invent settings.

## The gate models

The discretization and objective are **correct** — Bruno agrees to 10 digits, Weber to 7 with a
real 13027.66 → 11946.45 improvement and its `t = 24` kink exactly on the true ODE. Oliveira
solves outright, so a gate model can converge. What limits the rest is NLP convergence.

The real gate set comes from scanning the SBML, **not** the PEtab problem-features flag: 11
models contain `piecewise` (Alkan, Beer, Brannmark, Chen, Fujita, Giordano, Isensee, Oliveira,
Raimundez, Smith, Weber) plus Liu with an SBML `<event>`. Every switch time is a fixed offset,
none depends on an estimated parameter, so freezing `gate_vals` at nominal `p` is sound.

## Why convergence is stuck: the duals are costates

Primal is 1e-5…1e-7 everywhere; **dual** infeasibility is what stalls, which is why
`acceptable_tol = 1e-4` rescues Oliveira and nothing else.

The objective reaches only the `y`/`sigma` aux variables (~0.03% of variables on Bachmann), so
for nearly every variable stationarity reduces to the homogeneous `(Jᵀλ)_z = 0` — a backward
recursion forced only at measurement nodes. λ therefore inherits the ODE's **stiffness
spectrum**. Measured on Elowitz: `|λ|` spans 11.8 decades, and two states' collocation blocks
sit 7 orders apart, so one scalar `tol` is simultaneously unreachable for one state and vacuous
for another. MadNLP's row scaling cannot fix this — `min(1, 100/‖J_i‖∞)` is one-sided and 89.5%
of rows come back exactly 1.0.

**The lever is per-state, two-sided scaling of the collocation residuals at build time**, here,
since MadNLP cannot do it.

### Ruled out — do not re-litigate

- **cuDSS** and **`DualInitializeLeastSquares`** — both tested, both still RESTORATION_FAILED.
  Stop spending effort on the linear solver and on dual initialization.
- **Per-state `max|z|` spread** — does not predict outcome (Weber 5.2e7 works, Brannmark 1.0e7
  fails). Magnitude was the wrong axis; the multiplier spread is the right one.
- **"Just needs more iterations"** — no, see the Giordano non-monotonicity.
- **`RelaxEquality` decouples sigma** — no; it relaxes by `bound_relax_factor` = 1e-8 and is
  already the default under `SparseCondensedKKTSystem`, so the original A/B was confounded.
- Also measured and found to be no-ops: mesh clustering near switches, tighter ODE tolerance,
  equality treatment.

## Mesh: r-refinement buys nothing

Screened all 20 target models by per-interval interpolation defect, then causally tested three
node placements on the top candidate (Sneyd, identical N=125). Equidistribution cut
discretization error **52000×** and made the solve **2.5× longer** with a marginally worse
objective. Conditioning binds, not resolution. Do not re-run the static-mesh A/B; only *dynamic*
re-meshing is still open.

Two traps behind that:

- **The mesh is built at the optimum.** `_get_mesh_nodes` uses the adaptive ODE solver's own
  accepted steps at nominal `p`, and PEtab's `nominalValue` **is** the published best fit. So
  nominal-`p` defect is ~1e-15 by construction and diagnoses nothing — measure at perturbed `p`.
  Nine of the 20 (Armistead, Bertozzi, Blasi, Laske, Okuonghae, Perelson, Rahman,
  SalazarCavazos, Zhao) have no published reference; do not use them as an optimality baseline.
- **Borghans and Elowitz are not mesh problems.** They fail on every placement with a
  `DomainError`: both raise an unbounded state to an *estimated Hill exponent* while
  `variables.jl:75` gives every state `lvar = -Inf`. On GPU it is a NaN instead.
  **Untested cheap fix: `lvar = 0` on species.**

## Open gaps

- **No regression sweep over the 20 `BENCHMARK_MODELS`.** The mesh floor and the sigma floor
  change every model, and only the 3-model suite has run against them.
- **Raimundez never ran** under the benchmark config — it was queued behind the wedged Isensee.
- `objective.jl` and `steadystate.jl` still each implement the same five-way formula dispatch,
  differing only in the node accessor (`z[v,cidx,idx+1,0]` vs `zss[v,cidx]`). Unifying them is
  the real cleanup; doing one file alone would make the codebase less coherent. **Author's call.**

## Repro

Model sets in `options.jl`: 35 total − 12 `EXCLUDED_MODELS` (discontinuities) − 3
`FAILED_MODELS` (Froehlich, Lang, Raia) = 20 `BENCHMARK_MODELS`.

Regression bar is `test/runtests.jl` — Blasi (steady-state path), Schwen (log10, sqrt noise, 19
conditions), Sneyd (assignment rules, 9 conditions) against PEtab.jl at `rtol = 1e-4`. Run it as
`Pkg.test()` writing directly to a log file; piping has swallowed the output before.

EMC's own `CLAUDE.md` covers everything below the `add_con_collocation` boundary.
