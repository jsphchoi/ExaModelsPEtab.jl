# Running the ExaModelsPEtab benchmarks — operational runbook

Self-reference guide. Read this BEFORE launching or interpreting a benchmark run.
It exists because the same mistakes keep recurring: (1) mistaking process wall-clock for
build time, and (2) re-launching a killed model and having the harness silently skip it.

---

## 0. The two cardinal rules (the mistakes to never repeat)

1. **Wall-clock of the process is NOT the build time.** The process clock bundles Julia
   startup + package load + the **Bruno warmup build+solve** + PEtab construction + GPU init,
   none of which are in the target model's `exa_compile_time`. The build clock only starts at
   the `[Model] compiling...` log line. Read the **result-file fields** and **`@info` log
   markers**, never "it's been running N minutes so the build is taking N minutes."
   (Zheng: I thought it hung at "17 min"; the real `exa_compile_time` was ~497s — the rest was
   startup + Bruno warmup.)

2. **A killed run leaves a stale `compiling` sentinel that makes the next run SKIP the model.**
   On startup `main()` converts a leftover `exa_compile_status=compiling` → `timeout` (a
   *terminal* status), and `exa_finished()` then treats the model as done → `nothing to do` →
   clean exit, **without building anything**. The `exa_compile_time=14400.0` it writes is a
   bookkeeping artifact, NOT a real 4-hour compile. **Always clear stale exa state first**
   (§2). A 14400.0 compile_time with a fast process exit = this artifact, not a timeout.

---

## 1. Pre-flight checks

```bash
cd /home/jsphchoi/.julia/dev/ExaModelsPEtab
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader   # GPUs free?
pgrep -af run_examodels | grep -v grep                                     # competing run?
```
- Both GPUs should be ~idle (a few MiB, 0%). Other users' julia procs that don't touch the GPU
  are fine.
- Zombies/`<defunct>` survive a soft kill; kill the real PID with `kill -9 <PID>` if needed.

## 2. Clear stale state for any model you intend to (re)run

The harness skips a model unless `exa_finished()` is false. `exa_finished` is true when
`exa_compile_status ∈ {ok→solve-terminal, timeout, error, missing_yaml, skipped}`. So to FORCE
a rebuild, strip the `exa_*` fields (keep `petab_*`):

```bash
rp=examples/Benchmarks/results/Raimundez_PCB2020_results.txt
grep -E "^petab_" "$rp" > "$rp.tmp" && mv "$rp.tmp" "$rp"   # keep petab_, drop exa_
```
With no `exa_compile_status`, `exa_finished` → false → the model lands in `todo` and builds.
(If the file is all-`exa_`/nonexistent, a fresh run builds it anyway.)

## 3. Launch (exa / GPU side)

```bash
BENCH_SUBSET=Raimundez_PCB2020 \
  julia -t1 --project=. examples/Benchmarks/run_examodels.jl 0 1 0 \
  > /tmp/raimundez_bench.log 2>&1 &
```
- Args: `<gpu_id> <ninst> <idx>` (single model: `0 1 0`).
- `BENCH_SUBSET` = comma-separated model list; default is `EXA_RERUN_INLOOP`. This is the ONLY
  runtime env knob — all solver/benchmark settings (K, SGM_N, tols, limits) live in `options.jl`.
- `-t1` (single thread) is REQUIRED: the Julia watchdog thread hangs the cuDSS GPU solve;
  timeouts are enforced by an external `with_hard_deadline` kill, not a Julia thread.
- **Bruno is the warmup** (built+solved first, discarded). Never benchmark Bruno this way — it
  is timed separately via `run_bruno.jl`.

## 4. Track the RIGHT signals (live)

**Log `@info` phase markers** (the timeline):
```
instance 0/1 on GPU 0 (...)
instance 0: 1 models assigned, 1 remaining   <- MUST say "1 remaining", not "0 / nothing to do"
warmup: JIT build+solve on Bruno_JExpBot2016 ...   (warmup — excluded)
warmup done
[Model] compiling (K=4, compile_limit=14400s)...   <- BUILD CLOCK STARTS HERE
[Model] solving with MadNLP (max_wall_time=7200s)...
[Model] SGM solve i/5 ...
[Model] done
```
If you see `0 remaining` / `nothing to do`, the model was skipped — go back to §2.

**Result-file fields** (`results/{Model}_results.txt`, written incrementally):

| field | meaning |
|---|---|
| `exa_compile_status` | `compiling` → `ok` / `timeout` / `error` |
| `exa_compile_time` | **build only** (after warmup); authoritative. NOT wall-clock |
| `exa_presolve_time` | sub-portion: PEtabModel+PEtabODEProblem+ExaCore+`_create_variables` |
| `exa_solve_time` | **COLD** first solve — includes one-time GPU-kernel JIT |
| `exa_sgm_solve_time` | **WARM** solve, geo-mean of `EXA_SGM_N` reruns — the representative metric |
| `exa_term_status` | `SOLVE_SUCCEEDED` / `SOLVED_TO_ACCEPTABLE_LEVEL` / ... |
| `exa_objective` | compare to `petab_objective` (rel.gap) |

Cold vs warm matters: e.g. Zheng `exa_solve_time=128.76` (cold, ~108s of it is kernel JIT) vs
`exa_sgm_solve_time=20.43` (warm). Always label which one you are quoting.

## 5. Watchdog caps (distinguish real timeout from artifact)

- `COMPILE_LIMIT = 14400s` (4 h), `SOLVE_LIMIT = 7200s` (2 h).
- A **real** timeout is consistent with elapsed wall-clock AND logged (`compiling...` then killed).
- A `timeout` written in the first seconds with no `compiling` log line = the stale-sentinel
  reset artifact (§0.2), not a real timeout.

## 6. PEtab (CPU) side + SGM

PEtab numbers come from `run_petab.jl` (same `BENCH_SUBSET` mechanism). SGM (warm) reruns
fill `petab_sgm_solve_time`. **Models currently missing `petab_sgm`** (need an SGM rerun):
Bachmann, Borghans, Brannmark, Elowitz, Fujita, Giordano, Isensee, Oliveira, Raimundez, Zheng.
(Some PEtab solves are slow — Lucarelli SGM ~2659s — so n=5 reruns there are still long.)

## 7. Regenerate the report

```bash
julia --project=. examples/Benchmarks/results.jl          # SGM (warm) — DEFAULT, fair comparison
julia --project=. examples/Benchmarks/results.jl --cold   # cold first-run solve times
```
- Rows = `sort(EXA_SUPPORTED_MODELS)` (full scoreboard). Status codes: `0`=clean optimum,
  `0A`=acceptable-level, `0S`/`0AS`=succeeded/acceptable but suboptimal (obj ≥ +1.5% vs PEtab,
  excluded from "solved"), `1/2/3/4/5`=MadNLP failure classes, `E/T/-`=exception/timeout/not-run.
- A `-` in a Solve(s) cell = no SGM recorded for that backend (didn't converge, or SGM not yet run).

## 8. Canonical model lists

`options.jl` is the single source of truth (included by all three scripts):
`BENCHMARK_MODELS` (35) ⊇ `PETAB_SOLVED_MODELS` (28) ⊇ `EXA_SUPPORTED_MODELS` (25);
`EXA_RERUN_INLOOP` (15) is the timed exa loop (Bruno excluded as warmup).
