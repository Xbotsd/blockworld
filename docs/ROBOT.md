# Robot in the loop

A static site model answers "what would happen if". This module answers the next
question: **what should the robot do next, given what it has measured so far.**

```
      truth world (what is really there)
            │
            ▼
   ┌─────────────────┐   measurements    ┌──────────────────┐
   │  devices        │ ────────────────► │  assimilator     │
   │  lidar/cpt/probe│   σ, depth, id    │  inverse-variance│
   └─────────────────┘                   │  + outlier gate  │
            ▲                            └────────┬─────────┘
            │                                     ▼
   ┌─────────────────┐                   ┌──────────────────┐
   │  transport      │                   │  model world     │
   │  loopback/jsonl │                   │  (the twin)      │
   └─────────────────┘                   └────────┬─────────┘
                                                  ▼
                                         ┌──────────────────┐
                                         │  verifiers       │
                                         │  margins, pass   │
                                         └────────┬─────────┘
                                                  ▼
   ┌─────────────────┐                   ┌──────────────────┐
   │  actuator       │ ◄──── treat ───── │  supervisor      │
   │  (your robot)   │                   │  rules + evidence│
   └─────────────────┘                   └──────────────────┘
```

## Three swap points

| you want to change | replace | where |
|---|---|---|
| the sensor payload (LiDAR → GPR → CPT → your SDK) | `ScanDevice.scan()` | `robot/devices.py`, `register_device` |
| how data arrives (in-process → serial → TCP → recorded stream) | `Transport.send/recv` | `robot/link.py`, `register_transport` |
| the decision rules or thresholds | `Supervisor` / `PolicyConfig` | `robot/agent.py` |

Nothing else moves: the session, the assimilator and the verifiers are unchanged.

## The loop, one tick at a time

```python
from blockworld import Criteria
from blockworld.robot import RobotSession, Supervisor, make_device

session = RobotSession(
    model=model_world,      # the twin: what we currently believe
    truth=truth_world,      # reality: only the sensors may look at it
    criteria=Criteria(purposes=("infrastructure",)),
    region=foundation_region,
    devices=[make_device("lidar"), make_device("cpt", per_scan=3),
             make_device("probe")],
    treatments=(gravel_replacement,),
    act=True,               # False = advise only
    seed=3,
)
report = session.run(ticks=12)
report.final.action         # certify / treat / abort / escalate / rescan
```

Each tick: scan → assimilate → evaluate → decide. Every tick emits two JSONL
events (`scan`, `decision`) through the transport, so a run can be replayed or
audited line by line.

## Decision rules

The supervisor is a rule table, not a model. In order:

| # | condition | action |
|---|---|---|
| 1 | effective measurements < `min_evidence` or subsurface coverage < `min_coverage` | `rescan` |
| 2 | a supplied `p_fail` exceeds `p_fail_limit` | `treat` if a treatment exists, else `abort` |
| 3 | a criterion fails and a treatment exists | `treat` (the best one is chosen by re-evaluating each candidate) |
| 4 | a criterion fails and nothing helps | `abort` |
| 5 | margins pass but stay below `certify_margin`, and repeated rescans do not move them | `escalate` (human review) |
| 6 | margins pass and are above `certify_margin` | `certify` |

Defaults: `certify_margin=0.30`, `min_evidence=12`, `min_coverage=0.90`,
`p_fail_limit=0.50`. Coverage counts **subsurface** evidence (strength, water
level) — LiDAR alone cannot certify a bearing capacity.

## Assimilation rules

* Inverse-variance blend: `w = σ_prior² / (σ_prior² + σ_meas²)`.
* Outlier gate: `|x − prior| / √(σ_prior² + σ_meas²) > 4` → rejected, logged, and
  the model is left alone. Sensor faults do not silently enter the twin.
* Point measurements update **one block** (elevation, water level). Strength
  measurements spread over `influence_m` with a Gaussian weight, because a logged
  unit is laterally continuous.
* If a scan reports a material the block does not contain, the layer at that
  depth is **reclassified** instead of having its parameters nudged. This is how
  a mis-logged borehole gets corrected.

## The shipped scenario

`blockworld robot` reproduces a specific failure mode: an early survey recorded a
soft clay layer as stiff clay (`clay_soft → clay_stiff`). Under that prior, a
1.5 × 1.5 m footing carrying 400 kN **passes** (margin +0.50). Reality disagrees
(margin −1.54).

```
$ blockworld robot --out out_robot --ticks 12

  t0  测量 11 采纳 11 覆盖率 44%  infrastructure=-1.54  → rescan
  t1  测量 13 采纳 13 覆盖率 67%  infrastructure=-1.54  → rescan
  t2  测量 12 采纳 12 覆盖率 78%  infrastructure=-1.54  → rescan
  t3  测量 12 采纳 12 覆盖率 100% infrastructure=-1.54  → treat
  t4  测量 12 采纳  9 覆盖率 100% infrastructure=+0.38  → certify
  t5  测量 12 采纳  9 覆盖率 100% infrastructure=+0.38  → certify

最终判断：certify
模型误差（扫描范围内）先验：c MAE 16.5 kPa, 材料错判 50%
                      扫描后：c MAE 2.67 kPa, 材料错判 5.6%
```

Three things are visible at once: the model is corrected by measurement, the
verdict flips, and the robot decides *what to do about it* rather than just
reporting a number.

## Connecting real hardware

A device adapter is ~15 lines:

```python
from blockworld.robot import Measurement, register_device

class MyCPT:
    name = "cpt-live"

    def __init__(self, port):
        self.link = open_serial(port)          # your transport

    def scan(self, truth, rng):                # truth is unused on hardware
        frame = self.link.readline()           # {"depth": 1.2, "qc": 1.8, "fs": 0.03}
        return [Measurement("material_c", x=self.x, y=self.y,
                            value=qc_to_c(frame["qc"]), sigma=2.0,
                            depth=frame["depth"], material="clay_soft",
                            device=self.name)]

register_device("cpt-live", MyCPT)
```

The same applies to the link: implement `send`/`recv` over TCP or MQTT and
register it as `tcp:`; the session keeps working and the JSONL remains auditable.

## Limits

* The twin only learns what the payload measures; unmeasured blocks keep their
  prior. Coverage is reported for exactly this reason.
* Actuation is modelled as "the intervention is applied to both the twin and the
  ground". Real execution delay, partial application and rework are out of scope.
* `escalate` means the data cannot separate pass from fail at the chosen
  thresholds. It is a first-class outcome, not a failure of the loop.
* No sensor calibration model, no timing/synchronisation model, no SLAM: pose is
  assumed known.
