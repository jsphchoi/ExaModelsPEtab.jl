# ExaModelsPEtab.jl — branch `emc`

Keep this file small: only fixed conventions, nothing recoverable from the code, the docs, or the
auto-memory. Cross-package facts (HSL/MUMPS, dev-links, ExaModels API rules, file style) are in
`~/.julia/dev/CLAUDE.md`. The `emc` research log and live per-model status are in the auto-memory
(`memory/MEMORY.md` index), not here. EMC's own `CLAUDE.md` covers everything below the
`add_con_collocation` boundary.

## Conventions

- The public API is deliberately minimal: one export,
  `examodel_petab(filename; backend, K = 4, subdivide = 4, p0 = nothing, kwargs...)`, returning
  the `ExaModel` itself with the PEtab maps attached as `model.petab`. `kwargs...` are forwarded
  straight to `CollocationExaCore`, except `roots`/`basis`/`adaptive`, which error: Radau on a
  fixed mesh is the only configuration.
- p is ordered by the parameters table (`estimate == 1` rows), not by PEtab.jl's `xnames`. The
  condition axis indexes unique (preequilibrationConditionId, simulationConditionId) pairs and
  doubles as the EMC mesh axis (`_VARIANT = :percond`, with `:shared` as the fallback, an open
  question to revisit).
- Control values (`u`, the `__parameter_ifelse` booleans) live in the iterator; `add_par`
  prolongs compile time. Only true fixed-time events are supported, so an estimated trigger
  time is a hard error, never a frozen profile.
- `emc` runs the latest package versions and its drift from `main` is intentional, so do not
  "fix" version mismatches or worry about CI resolvability.
- Run the test suite as `Pkg.test()` writing straight to a log file; piping has swallowed the
  output before.
