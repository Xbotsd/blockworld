# Architecture

```
              site.json ─┐
              dem.csv    ├─►  SiteSpec ──build_from()──►  World
              boreholes  │        │                          │
              climate ───┘        │                          ├── Tile
                                  │                          │    ├── LayeredColumn  (geology)
                                  │                          │    ├── temp / precip   (climate)
                                  │                          │    ├── wt / discharge  (hydrology)
                                  │                          │    ├── cohorts         (vegetation)
                                  │                          │    └── animals         (IBM)
                                  │                          │
                                  ▼                          ▼
              Intervention (op list) ──apply()──►  EditSession  (validated, logged, reversible)
                                  │                          │
                                  └──────────►  Study.evaluate()  ──►  Verdict[]  (margin, passed, reasons)
                                                                       │
                                                    ┌──────────────────┼──────────────────┐
                                                    ▼                  ▼                  ▼
                                            monte_carlo()        morris()          report.py
                                            P(fail), quantiles   mu*/sigma         HTML + PNG
```

## Module contracts

| module | owns | invariant |
|---|---|---|
| `site.py` | `SiteSpec`, `GridSpec`, `build_from`, `synthesize` | never invents missing data; synthetic output is labelled |
| `io.py` | `site.json`, DEM/climate CSV, borehole CSV | every parse error names the file and the row |
| `geology.py` | kriging, `Borehole`, `LayeredColumn`, `reconstruct` | a replacement conserves total column thickness |
| `materials.py` | material library, validity domain | `check()` returns errors that callers turn into exceptions |
| `world.py` | `World`, `Tile`, `Region`, `Cohort` | a world is deep-copyable; regions are serialisable |
| `edits.py` | `EditSession` | every mutation is logged and reversible |
| `hydrology.py` | PET, Budyko, D8 routing, water table | mass balance: `et + runoff + infil ≈ precip` |
| `vegetation.py` | cohort competition, recruitment, mortality | biomass is capped; no negative biomass survives |
| `animals.py` | energy-constrained IBM | individuals die at `reserve ≤ 0`; population is bounded |
| `verifiers.py` | the five criteria | every verdict carries `margin`, `passed`, `reasons` |
| `solvers.py` | property and criterion registries | defaults apply only when nothing is registered |
| `api.py` | `Study`, `Criteria`, `Intervention`, `StudyResult` | results are JSON-serialisable |
| `uncertainty.py` | distributions, MC, Morris | all randomness comes from an explicit seed |
| `report.py` | PNG, HTML | no network access at render time |

## Three design decisions worth stating

**1. Data and engine are separate.**
`build(WorldConfig)` still exists, but it is a thin alias for
`build_from(synthesize(cfg))`. Anything that matters goes through `SiteSpec`, so
a real dataset and a generated one take the same code path and differ only in
provenance.

**2. Refusal is a feature.**
`Material.check()`, `LayeredColumn.replace()`, `set_pft()` and `SiteSpec.validate()`
raise rather than clamp. A model that silently substitutes a legal-looking value
for an illegal one produces output that cannot be audited.

**3. Criteria are pinned to sites.**
`Study.evaluate()` always scores at `Criteria.foundation_site` /
`landing_site` (or the auto-selected ones). An intervention elsewhere does not
move the verdict — deliberate, so a local treatment cannot "pass" a global
question by coincidence.

## Units

Lengths m, stresses kPa, unit weights kN m⁻³, moduli MPa, energy kJ, mass kg,
time years (years for ecology, seconds internally for gravity-related maths).
Every dataclass that carries a dimensioned number documents it inline.

## Determinism

`SiteSpec.seed` drives site synthesis; `Study.seed` and the `seed` argument of
`monte_carlo` / `morris` drive sampling. Given the same seeds the output is
bit-identical, which is what the CI regression test relies on.
