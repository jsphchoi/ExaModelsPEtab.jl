# Running the ExaModelsPEtab benchmarks — operational runbook

Read this BEFORE launching or interpreting a benchmark run. The suite compares
**ExaModelsPEtab + MadNLP** on **GPU** (`exagpu_*`) and **CPU** (`exacpu_*`) against
**PEtab.jl + Optim.IPNewton** (`petab_*`), over the canonical model set in `options.jl`.

---

## 0. The two cardinal rules (the mistakes to never repeat)

1. **Process wall-clock is NOT the build time.** The process clock bundles Julia startup +
   package load + the **Bruno warmup build+solve** + PEtab construction + GPU init — none of
   which are in a target model's `*_compile_time`. The build clock only starts at the
   `[Model] compiling...` log line. Read the **result-file fields** and **`@info` log markers**,
   never "it's been running N minutes so the build is taking N minutes."

2. **A killed run leaves a stale `compiling` sentinel that makes the next run SKIP the model.**
   On startup `main()` converts a leftover `<pfx>compile_status=compiling` → `timeout` (a
   *terminal* status), and the finished-check then treats the model as done → clean exit
   **without building anything**. The `compile_time=3600.0` it writes is a bookkeeping artifact,
   NOT a real 1-hour compile. **Clear stale state first** (§2) before re-running.

---

## 1. Environment

The benchmark scripts run in the **`examples/` environment** (`--project=examples`), *not* the
package env — the package itself does not depend on the solver/GPU stack (MadNLP, MadNLPGPU,
CUDA, CUDSS, Optim); those live in `examples/Project.toml`. On a fresh clone, instantiate once
(also wires the dev path to the package):

```bash
cd /home/jsphchoi/.julia/dev/ExaModelsPEtab
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Pre-flight:
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader   # GPUs free?
pgrep -af run_examodels | grep -v grep                                           # competing run?
```

## 2. The backend knob and result keys

Every script is parametrized by the **`BENCH_BACKEND`** env var, which selects the result-key
prefix (`PFX`) written to `results/{Model}_results.txt`:

| `BENCH_BACKEND` | backend | prefix | linear solver |
|---|---|---|---|
| `gpu` (default) | CUDA   | `exagpu_` | `MadNLPGPU.CUDSSSolver` |
| `cpu`           | CPU    | `exacpu_` | MadNLP default |

PEtab (`run_petab.jl`) writes `petab_*` and is **backend-independent** (only run once, on the
GPU pass). All three prefixes coexist in the same `{Model}_results.txt`, so GPU and CPU runs
**do not collide** and can run concurrently.

Both ExaModels backends use the **identical** LiftedKKT (condensed-space) MadNLP config from
`options.jl`: `SparseCondensedKKTSystem` + `RelaxEquality` + `RelaxBound`. Only the linear
solver differs.

## 3. Clear stale state to FORCE a (re)build

The harness skips a model unless its finished-check is false. To force a rebuild, strip that
backend's keys (keep the others):

```bash
rp=examples/Benchmarks/results/Raimundez_PCB2020_results.txt
# drop exa keys (both backends), keep the PEtab reference so the slow PEtab side isn't re-run:
grep -v '^exagpu_\|^exacpu_' "$rp" > "$rp.tmp" && mv "$rp.tmp" "$rp"
```
With no `<pfx>compile_status`, the model lands in `todo` and rebuilds. (A nonexistent / all-other-prefix
file builds anyway.) Runs are **resumable**: every instance skips models already recorded, so a
watchdog SIGKILL or a re-launch just continues.

## 4. Launch

**GPU side** — the driver strides 2 instances across both cards (default, fastest) and
auto-restarts on a watchdog SIGKILL (exit 137):
```bash
bash examples/Benchmarks/run_examodels.sh            # 2 instances, GPUs 0+1 (default)
bash examples/Benchmarks/run_examodels.sh 1 1        # single instance on GPU 1
BENCH_SUBSET=Zheng_PNAS2012 bash examples/Benchmarks/run_examodels.sh   # ad-hoc subset
```

**CPU side** — no `.sh` driver (the `.sh` is GPU-striding only); stride instances directly.
`OPENBLAS_NUM_THREADS` caps per-instance BLAS so N instances don't oversubscribe the cores:
```bash
for i in 0 1 2 3 4 5 6 7; do
  BENCH_BACKEND=cpu OPENBLAS_NUM_THREADS=4 \
    nohup julia --project=examples -t1 examples/Benchmarks/run_examodels.jl 0 8 "$i" \
    > examples/debugging/logs/exacpu_inst$i.log 2>&1 &
done
```
- Args: `<gpu_id> <ninst> <idx>` (the gpu arg is a no-op under `BENCH_BACKEND=cpu`).
- `BENCH_SUBSET` (comma-separated) is the only model-selection knob; default `EXA_RERUN_INLOOP`.
- `-t1` is REQUIRED: a Julia watchdog thread hangs the cuDSS GPU solve; timeouts are enforced by
  an external `with_hard_deadline` SIGKILL, not a Julia thread.
- **Bruno is the warmup** (built+solved first, discarded). Bruno/Crauste are timed separately:
  ```bash
  julia --project=examples -t1 examples/Benchmarks/run_bruno.jl              # GPU
  BENCH_BACKEND=cpu julia --project=examples -t1 examples/Benchmarks/run_bruno.jl   # CPU
  ```

## 5. Track the RIGHT signals (live)

Log `@info` phase markers (the timeline):
```
instance 0/N ...
instance 0: K models assigned, K remaining   <- MUST say "K remaining", not "0 / nothing to do"
warmup: JIT build+solve on Bruno_JExpBot2016 ...   (warmup — excluded)
[Model] compiling (K=4, compile_limit=3600s)...    <- BUILD CLOCK STARTS HERE
[Model] solving with MadNLP (max_wall_time=3600s)...
[Model] SGM solve i/n ...
[Model] done
```
If you see `0 remaining` / `nothing to do`, the model was skipped — go back to §3.

Result-file fields (`results/{Model}_results.txt`, written incrementally; `<pfx>` ∈ `exagpu_`/`exacpu_`):

| field | meaning |
|---|---|
| `<pfx>compile_status` | `compiling` → `ok` / `timeout` / `error` |
| `<pfx>compile_time` | **build only** (after warmup); authoritative, NOT wall-clock |
| `<pfx>presolve_time` | sub-portion: PEtab construction + ExaCore + `_create_variables` |
| `<pfx>solve_time` | **COLD** first solve (includes one-time GPU-kernel JIT) |
| `<pfx>solve_times` | **WARM** raw per-rerun solve times (CSV, `BENCH_SGM_N` of them) — source of truth; `results.jl` applies the δ shift at report time |
| `<pfx>sgm_solve_time` | **WARM** solve, shifted geometric mean (δ=10s) of `solve_times` — cached aggregate (the representative metric) |
| `<pfx>term_status` | `SOLVE_SUCCEEDED` / `SOLVED_TO_ACCEPTABLE_LEVEL` / `WALLTIME_EXCEEDED` / ... |
| `<pfx>objective` | ExaModels' optimal collocation objective |
| `<pfx>petab_obj` | **`petab.nllh(exa_p*)`** — ExaModels' optimum scored under PEtab's own objective (the fair GAP numerator) |
| `<pfx>constr_viol` | inf-norm (max) constraint violation `max(lcon-c, c-ucon, 0)` at the solution — diagnostic, not shown in `results.jl` |

Cold vs warm matters — always label which solve time you quote.

## 6. Watchdog caps

- `COMPILE_LIMIT = 3600s`, `SOLVE_LIMIT = 3600s`. The CPU giants (Isensee/Bachmann/Lucarelli)
  routinely hit the solve cap → `WALLTIME_EXCEEDED` (status `W`); expected.
- A **real** timeout is consistent with elapsed wall-clock AND logged (`compiling...` then killed).
  A `timeout` written in the first seconds with no `compiling` log line = the stale-sentinel reset
  artifact (§0.2), not a real timeout.

## 7. Regenerate the report

```bash
julia --project=examples examples/Benchmarks/results.jl          # SGM (warm) — DEFAULT, fair comparison
julia --project=examples examples/Benchmarks/results.jl --cold   # cold first-run solve times
```
- Rows = all `BENCHMARK_MODELS`; three column groups: GPU / CPU / PEtab.
- **GAP(%)** = `(petab.nllh(exa_p*) − petab_obj) / |petab_obj| × 100` — ExaModels' params scored
  under PEtab's own objective vs PEtab's optimum (same-objective, fair). ≈0 means ExaModels found
  PEtab's optimum; negative means ExaModels found a lower one.
- **STAT codes** — MadNLP: `0`=succeeded, `0A`=acceptable-level, `0S`/`0AS`=succeeded/acceptable
  but suboptimal (GAP ≥ +2% vs PEtab, excluded from "solved"), `W`=walltime, `R`=restoration-failed,
  `D`=search-direction-too-small, `5`=other, `E`=error, `-`=not-run/compile-failed. PEtab:
  `0`=local min, `1`=converged-but-uncertified, `E`/`-`.
- A `-` in a SOL(s) cell = no SGM recorded for that backend.

## 8. Canonical model lists

`options.jl` is the single source of truth (included by all scripts):
`BENCHMARK_MODELS` (35) ⊇ `CONTINUOUS_MODELS` (23) ⊇ `EXA_SUPPORTED_MODELS` (20).
The 12 dropped at (35→23) carry the "Possible Discontinuities" feature on the official table
(https://benchmarking-initiative.github.io/Benchmark-Models-PEtab/) — events / piecewise(t) /
state-triggered switches the collocation transcription cannot represent (out of scope by
construction). The 3 dropped at (23→20) are continuous models the PEtab.jl reference itself could
not compile/solve (Froehlich, Lang, Raia), so there is no baseline to compare against.
`EXA_RERUN_INLOOP` is the timed exa loop (Bruno/Crauste excluded as warmup).
Solver/benchmark config (K, SGM_N, tols, limits) lives there too — edit it THERE, never in the
individual scripts.
