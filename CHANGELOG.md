# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-11-18

Robot in the loop: a measured-data path from sensors to decisions.

### Added

- **`blockworld.robot`** — a closed loop that scans, assimilates and decides:
  - `devices.ScanDevice` protocol with a registry (`register_device`) and four
    implementations: `SimLiDAR`, `SimCPT`, `SimProbe`, `ReplayDevice`.
  - `link.Transport` protocol with a registry (`register_transport`) and three
    implementations: in-process loopback, JSONL recorder and JSONL replayer.
  - `assimilate.Assimilator`: inverse-variance fusion with a 4σ outlier gate,
    single-block updates for point measurements, neighbourhood blending for
    strength, and **material reclassification** when a scan contradicts the log.
  - `agent.Supervisor`: an explicit six-rule policy returning
    `rescan / treat / abort / escalate / certify` with reasons and confidence,
    gated on evidence count and subsurface coverage.
  - `session.RobotSession`: scan → assimilate → evaluate → decide, with per-tick
    JSONL events, treatment selection by re-evaluation, optional actuation
    (`act=True`) that also updates the truth world, and before/after model error.
  - `robot.demo`: a reproducible scenario in which a soft clay layer was logged as
    stiff; the prior passes, the scan flips the verdict, the robot prescribes
    ground replacement and re-certifies after the fix.
- **CLI**: `blockworld robot --out DIR [--ticks N] [--transport jsonl:path] [--no-act]`.
- **`docs/ROBOT.md`**: loop diagram, swap points, decision table, hardware
  adapter example and stated limits.
- 26 new tests (95 total): device determinism, registry swaps, JSONL round-trip,
  outlier rejection, reclassification, every policy branch, and an end-to-end
  scenario asserting that the verdict flips and the model error falls.

### Changed

- `api.evaluate(world, criteria, sites=...)` is now a module-level function; the
  `Study` method delegates to it so the robot loop and a study share one path.
- Version bumped to 0.2.0.

## [0.1.0] — 2026-11-15

First public release.

### Added

- **Site data contract** (`src/blockworld/site.py`, `src/blockworld/io.py`):
  `SiteSpec` / `GridSpec` built from a DEM, boreholes and climate fields, with
  `site.json` + CSV on-disk formats, validation with actionable errors, and an
  explicit `synthesize()` generator whose output is labelled `synthetic` in
  provenance. Real data can no longer be silently replaced by made-up data.
- **Layered geology** (`geology.py`): ordinary kriging of layer-boundary depths
  from boreholes, nearest-borehole facies extrapolation, and a per-block
  `LayeredColumn` that supports depth-interval material replacement while
  conserving total thickness.
- **Material library and constitutive domains** (`materials.py`): 13 materials
  (soils, rock, lunar/martian/asteroid regolith, cemented regolith), each with a
  validity domain. Parameter sets outside the domain, or a constitutive model the
  material does not support, raise instead of producing a plausible-looking
  number.
- **Transactional edits** (`edits.py`): `set_material`, `replace_component`,
  `set_elevation`, `set_water_table`, `set_precip`, `set_pft`, `set_gravity`,
  each validated, logged and reversible.
- **Feasibility criteria** (`verifiers.py`): Vesic/Terzaghi bearing capacity with
  shape and depth factors, elastic settlement, infinite-slope stability with
  pore pressure, lander footpad bearing with dynamic amplification, penetration
  and peak deceleration, and ecological sustainability. Every verdict carries a
  normalised `margin`.
- **Public study API** (`api.py`): `Study`, `Criteria`, `Intervention`,
  `StudyResult` with `passed()` and `flips()`; interventions are serialisable
  op lists; site selection policies are explicit functions, not hidden defaults.
- **Uncertainty** (`uncertainty.py`): uniform/triangular/normal/lognormal
  distributions, Monte Carlo with per-criterion probability of failure and
  margin quantiles, and Morris elementary-effects screening with `mu*`/`sigma`.
- **Solver registry** (`solvers.py`): `register_property` for a single formula
  (e.g. an external bearing-capacity solver) and `register_criterion` for a
  whole external verifier; the active registry is recorded in results.
- **Process models** (`hydrology.py`, `vegetation.py`, `animals.py`): Budyko
  evapotranspiration with Hargreaves PET, D8 routing, topographic-wetness water
  table, cohort competition producing biomes, and an energy-constrained
  individual-based animal model.
- **Reports** (`report.py`): a self-contained interactive HTML sandbox with six
  layers, verdict cards, edit log and sensitivity table, plus a six-panel
  before/after PNG.
- **CLI** (`cli.py`): `blockworld demo`, `blockworld study`, `blockworld template`.
- **Tests**: 69 tests, including validation against closed-form bearing factors,
  Budyko bounds, allometric scaling, IO round-trips and refusal behaviour.

### Notes

- Parameter values for the ecological and hydrological processes are
  order-of-magnitude settings, not field calibrations; see
  `docs/PROVENANCE.md`. Use the package for relative comparisons.
- Editable installation requires network access for `setuptools`; tests and the
  CLI can also run with `PYTHONPATH=src`.
