# Guide

## Install

```bash
git clone https://github.com/Xbotsd/blockworld.git
cd blockworld
pip install -e .
blockworld --help
```

No network to install? The package is pure Python; run it in place instead:

```bash
PYTHONPATH=src python -m blockworld --help          # Linux / macOS
$env:PYTHONPATH="src"; python -m blockworld --help  # Windows PowerShell
```

Requirements: Python ≥ 3.10, `numpy`, `Pillow`.

## The three commands

### `blockworld template DIR`

Writes an editable dataset: `site.json` plus `dem.csv`, `temp.csv`,
`precip.csv`, `boreholes.csv`. Replace the CSVs with your data, keep the schema.

```
site.json
├── grid        {"nx": 8, "ny": 8, "cell": 100}
├── elevation   {"csv": "dem.csv"}        ny rows × nx columns, metres
├── temp        {"csv": "temp.csv"}       annual mean °C, optional
├── precip      {"csv": "precip.csv"}     annual total mm, optional
└── boreholes   {"csv": "boreholes.csv"}  x,y,ground,top,bottom,material[,source]
```

Borehole rows are **intervals**: consecutive rows for the same `(x, y)` must
cover `0 – bottom` without gaps, otherwise the loader names the offending
depth. Materials must exist in the library (`src/blockworld/materials.py`).

### `blockworld study CONFIG --out DIR [--mc N] [--morris R]`

Runs a baseline, applies each intervention, and reports per-criterion verdicts,
margins and flips. With `--mc` it adds Monte Carlo failure probabilities, with
`--morris` a screening ranking.

```json
{
  "site": "..\\data\\site\\site.json",
  "spin_years": 0,
  "criteria": {
    "purposes": ["infrastructure", "landing"],
    "foundation": {"B": 1.5, "L": 1.5, "D": 1.5, "load_kN": 1200},
    "lander":     {"mass_kg": 3000, "n_pads": 3, "pad_diameter": 0.25}
  },
  "interventions": [
    {"key": "换填砂砾", "ops": [
      {"op": "material", "material": "gravel_dense",
       "top_depth": 0, "bottom_depth": 8, "region": {"kind": "all"}}]}
  ],
  "parameters": [
    {"name": "clay_c",
     "target": {"kind": "material_prop", "material": "clay_soft", "prop": "c"},
     "dist": {"kind": "uniform", "low": 8, "high": 40}},
    {"name": "pad_d",
     "target": {"kind": "lander", "field": "pad_diameter"},
     "dist": {"kind": "uniform", "low": 0.2, "high": 0.9}}
  ]
}
```

Output files: `study.json`, and optionally `montecarlo.json`, `morris.json`.

### `blockworld robot --out DIR [--ticks N] [--transport SPEC] [--no-act]`

Runs the closed loop on a reproducible scenario (see [`ROBOT.md`](ROBOT.md)): a
soft clay layer was logged as stiff, the prior passes, the robot scans the
foundation footprint, the model is corrected, the verdict flips, and the
supervisor prescribes and applies a treatment.

`--transport jsonl:out/run.jsonl` records every measurement and decision; the
same file can be replayed with `blockworld study`-style code via
`make_transport("replay:out/run.jsonl")`. `--no-act` stops at advice.

### `blockworld agent --out DIR [--backend ollama|openai|none] [--model NAME]`

Runs the same closed loop, then asks a **locally deployed** model for a site
brief. See [`AGENT.md`](AGENT.md).

```bash
blockworld agent --out out_agent --model qwen2.5:7b           # Ollama
blockworld agent --backend openai --base-url http://127.0.0.1:8080/v1 --model local
blockworld agent --backend none                               # deterministic fallback
```

Outputs `agent_brief.md` (Markdown), `agent_advice.json` and `robot_report.json`.
The advisor cannot change the verdict: `advice.advisory_only` is always true, and
a contradicting model verdict is printed next to the policy action marked
authoritative.

### `blockworld demo --out DIR`

End-to-end example on a synthetic 48×48 site: build → 55-year vegetation spin-up
→ animals → three interventions (ground replacement, lunar landing site,
southern vegetation change) → sensitivity sweeps → `report.html`, `sandbox.png`,
`summary.json`, `sensitivity.csv`. Takes about 13 s.

## Reading the output

`study.json` has four blocks:

| block | content |
|---|---|
| `baseline` | verdicts before any intervention |
| `interventions.<key>.verdicts` | verdicts after that intervention |
| `interventions.<key>.deltas` | per criterion: `passed_before/after`, `flip`, `margin_delta` |
| `meta` | data source, chosen sites, criteria, active solver registry |

`margin` is normalised so that ≥ 0 means "passes with this much slack":
`min(FS/FS_req − 1, 1 − s/s_allow, …)`. It is comparable across criteria in sign
but not in units — use `flip` for the binary answer and `margin_delta` for the
size of the effect.

Monte Carlo adds `p_fail` and `margin_p05/p50/p95` per criterion. Morris adds
`mu` (direction), `mu_star` (magnitude) and `sigma` (non-linearity or
interaction) per parameter, plus a `rank`.

## Bringing your own data

1. Prepare a DEM as a plain CSV matrix (one value per block, `ny` rows).
2. Export boreholes as intervals with a material key from the library. Adding a
   new material means adding an entry to `materials.LIBRARY` with its validity
   domain — that is the intended extension, not editing the verifiers.
3. If you do not have climate fields, either supply them or accept the constant
   default, which is flagged in `provenance.climate`.
4. Run `blockworld study` with `spin_years: 0` for geotechnical questions, and
   with a positive `spin_years` only when ecological criteria are in play.

## Plugging in an external solver

```python
from blockworld import solvers

def plaxis_qu(profile: dict) -> float:
    # profile: c, phi, gamma, B, D, shape, L, gravity
    return run_plaxis(profile)

solvers.register_property("bearing_qu", plaxis_qu)
```

The registry is global for the process; `Study` records which properties and
criteria were overridden in `meta.solvers`, so a result can be traced back to the
solver that produced it. To replace a whole criterion, register a callable with
the signature `(world, region, foundation) -> Verdict`.

## Pitfalls

- **Site alignment.** Criteria are evaluated at the site recorded in
  `meta.foundation_site` / `meta.landing_site`. An intervention applied elsewhere
  will not move the verdict, by design. Pin the site explicitly when the
  treatment is local.
- **Spin-up cost.** `spin_years > 0` re-runs vegetation growth for the baseline
  and again for every intervention. Keep it small (10–30) unless you need an
  ecological equilibrium.
- **Ecology plus Monte Carlo** is expensive: sampling climate parameters without
  re-running the vegetation model gives an answer about the current vegetation,
  not about a shifted climate.
- **Units are not enforced.** Everything is metres, kPa, kg and years; passing
  feet or psi will not raise, it will just be wrong.
