# Multistart / better-init findings

Diagnostic campaign (2026-06): why several ExaModels-supported models converge to a
**worse** objective than PEtab from the nominal warm start, and whether that is a
formulation defect or a basin-of-attraction (start-point) problem.

## Method — warm-start from PEtab's optimum θ*

For a model whose exa solve lands a suboptimal/wrong objective from the nominal start:

```julia
PEmodel = PEtab.PEtabModel(yaml); PEprob = PEtab.PEtabODEProblem(PEmodel)
# 1. PEtab-calibrate from nominal to the good optimum θ* (use FULL optim_opts —
#    f_reltol + successive_f_tol + time_limit — or IPNewton never stops)
res = PEtab.calibrate(PEprob, PEtab.get_x(PEprob), Optim.IPNewton(); options=optim_opts())
# 2. Mutate the stored nominal in place → reroutes the ENTIRE exa warm start
#    (decision p AND the forward-simulated collocation states) to θ*
PEtab.get_x(PEprob) .= res.xmin
# 3. Build exa at θ* and solve
m = ExaModelsPEtab._build_petab_examodel(PEmodel, PEprob, CUDA.CUDABackend(), K)
```

`obj(x0 @ θ*)` ≈ PEtab fmin ⟹ the collocation formulation is faithful at the optimum
(no discretization gap). The solve outcome then tells multistart vs conditioning.

## Result — all 5 suboptimal cases are MULTISTART, not bad formulation

| model | exa from nominal | exa from θ* | status from θ* |
|---|---|---|---|
| Bertozzi  | +194%            | −0.0%  | SOLVE_SUCCEEDED (18 it) |
| Zhao      | +8.19%           | −0.0%  | RESTORATION_FAILED (right obj, needs a conditioning assist for a clean KKT cert) |
| Laske     | +3.57%           | +0.26% | SOLVE_SUCCEEDED (16 it) |
| Okuonghae | +1.90%           | −0.55% | SOLVE_SUCCEEDED (383 it) |
| Oliveira  | +3.8e6% (8.9e8 garbage basin) | −0.1% | SOLVE_SUCCEEDED (14 it) — most dramatic |

`obj(x0 @ θ*)` reproduces the PEtab optimum to ~10 digits for Bertozzi/Zhao/Okuonghae/Oliveira
(zero discretization gap). Laske's start-objective is off (364 vs 290) but the solve corrects
it to +0.26% — so the start-objective discrepancy does NOT imply a mesh problem.

**Conclusion:** the good optimum is a stable, reachable point of the collocation NLP; the
nominal start just lands a worse (sometimes garbage) basin. **Fix = a better initial p0 /
multistart** (seed exa from a cheap PEtab or forward-sim pre-solve), NOT objective
reformulation, nondimensionalization, or conditioning work. Oliveira in particular vindicates
dropping the nondim-scaling idea — it was never a scaling/formulation bug.

## Overnight re-run of the 9 failed-exa models (1 hr caps)

Re-ran the failed-exa models with compile/solve caps lowered to 1 hr (no successful run of any
backend has ever exceeded 1 hr; see options.jl). All 9 now terminate cleanly under the caps
(no more multi-hour hangs). Highlights:

- **Oliveira** and **Bachmann** now BUILD (were `error` / K=6 OOM); Bachmann builds at K=4.
- **Lucarelli (−4.9%)** and **Raimundez (−4.1%)** reach objectives *below* PEtab (the
  collocation-relaxation "ghost" direction) but run out of time at the 1 hr solve cap — they are
  descending toward the right answer and only need better conditioning / more iterations.
- The rest are confirmed large-scale KKT-conditioning failures (RESTORATION_FAILED /
  SEARCH_DIRECTION_TOO_SMALL / WALLTIME / Isensee OOM at 9.7M vars on 32 GB).

Per-model numbers are in `results.txt` and `results/<model>_results.txt`.

## Takeaway for the roadmap

A growing set of failures — the 5 multistart models above plus the negative-gap walltime cases
(Lucarelli, Raimundez) — are all "the good optimum exists and is reachable; the solver just
can't get there from the default start or in time." That is exactly what better-conditioned,
better-resolved discretization (AMREE) + stiffness removal (PSSA) target, and AMREE/PSSA may
also supply better starts. See the AMREE (#13) / PSSA (#18) tasks.
