"""闭环会话：扫描 → 同化 → 评估 → 判断，可重复、可录制。

一次 tick 做四件事：

    scan()      从终端取回测量（真值世界 + 噪声）
    assimilate  逆方差融合进模型，异常值被挡下
    evaluate    用当前模型重跑判据
    decide      规则型自主判断：认证 / 复扫 / 处理 / 中止

`transport` 决定事件流去哪里：进程内回环、JSONL 录制、或回放（等价于换一条
链路重跑同一套判断）。模型与真值是两个世界——这正是数字孪生的设置：
模型只有通过测量才能靠近真值。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from ..api import (Intervention, choose_foundation_site, choose_landing_site,
                   evaluate)
from ..verifiers import Verdict
from ..world import Region, World
from .agent import Supervisor
from .assimilate import Assimilator
from .devices import ScanDevice
from .link import Loopback, Transport
from .types import Decision, Measurement, SessionReport


@dataclass
class RobotSession:
    model: World
    truth: World
    criteria: object
    devices: list[ScanDevice]
    policy: Supervisor = field(default_factory=Supervisor)
    seed: int = 0
    region: Region | None = None
    transport: Transport | None = None
    treatments: tuple[Intervention, ...] = ()
    max_ticks: int = 8
    act: bool = False            # True：判断为 treat 时直接在执行层落地，并复核

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.assimilator = Assimilator(
            self.model, influence_m=float(self.model.config.cell) * 1.5
        )
        self.transport = self.transport or Loopback()
        self.transport.open()
        self.sites = {
            "foundation": self.criteria.foundation_site or
            choose_foundation_site(self.model),
            "landing": self.criteria.landing_site or
            choose_landing_site(self.model),
        }
        # 扫描范围默认与选区一致：机器人在要建的地方扫描，而不是全世界
        for dev in self.devices:
            if getattr(dev, "region", None) is None and self.region is not None:
                try:
                    dev.region = self.region
                except AttributeError:
                    pass
        self.scope = self.model.region_tiles(self.region) if self.region else self.model.tiles
        self.scope_idx = {self.model.idx(t.ix, t.iy) for t in self.scope}
        self.measured: set[tuple[int, int]] = set()
        self.measured_sub: set[tuple[int, int]] = set()   # 有地下证据（强度/水位）的区块
        self.evidence = 0
        self.measurements = 0
        self.ticks: list[dict] = []
        self._error_before_act: dict | None = None
        self._last_margins: dict | None = None
        self._treated: set[tuple[int, int]] = set()
        self.before = self._snapshot()
        self.before_error = self._model_error()

    # -- 扫描 -----------------------------------------------------------
    def scan(self) -> list[Measurement]:
        out: list[Measurement] = []
        for dev in self.devices:
            got = dev.scan(self.truth, self.rng)
            for m in got:
                out.append(Measurement(
                    kind=m.kind, x=m.x, y=m.y, value=m.value, sigma=m.sigma,
                    device=m.device, depth=m.depth, material=m.material,
                    t=float(len(self.ticks)), meta=m.meta,
                ))
        self.measurements += len(out)
        return out

    # -- 覆盖度 ---------------------------------------------------------
    SUBSURFACE = {"material_c", "material_phi", "water_table"}

    def coverage(self) -> float:
        """有**地下证据**的区块比例：单纯测高程不能支撑承载/沉降结论。"""
        return len(self.measured_sub) / max(len(self.scope_idx), 1)

    def coverage_any(self) -> float:
        return len(self.measured) / max(len(self.scope_idx), 1)

    # -- 单步 -----------------------------------------------------------
    def tick(self) -> dict:
        k = len(self.ticks)
        ms = self.scan()
        records = self.assimilator.apply_all(ms)

        for m, rec in zip(ms, records):
            if not rec.accepted:
                continue
            t = self._tile_near(m.x, m.y)
            if t is not None:
                self.measured.add((t.ix, t.iy))
                if m.kind in self.SUBSURFACE:
                    self.measured_sub.add((t.ix, t.iy))

        self.evidence = self.assimilator.accepted
        verdicts = self.evaluate()
        margins = {v.key: round(v.margin, 4) for v in verdicts}
        stagnant = margins == self._last_margins
        self._last_margins = margins
        decision = self._decide(verdicts, k, stagnant)

        self.transport.send({
            "tick": k, "type": "scan",
            "measurements": [m.to_dict() for m in ms],
            "accepted": sum(1 for r in records if r.accepted),
            "rejected": sum(1 for r in records if not r.accepted),
            "coverage": round(self.coverage(), 3),
            "coverage_any": round(self.coverage_any(), 3),
        })
        self.transport.send({"tick": k, "type": "decision",
                             **decision.to_dict()})

        rec = {
            "tick": k,
            "measurements": len(ms),
            "accepted": sum(1 for r in records if r.accepted),
            "rejected": sum(1 for r in records if not r.accepted),
            "coverage": round(self.coverage(), 3),
            "coverage_any": round(self.coverage_any(), 3),
            "evidence": self.evidence,
            "verdicts": [
                {"key": v.key, "passed": v.passed, "margin": round(v.margin, 3)}
                for v in verdicts
            ],
            "decision": decision.to_dict(),
        }
        self.ticks.append(rec)
        return rec

    def evaluate(self) -> list[Verdict]:
        return evaluate(self.model, self.criteria, sites=self.sites)

    # -- 决策 -----------------------------------------------------------
    def _decide(self, verdicts: list[Verdict], tick: int,
                stagnant: bool = False) -> Decision:
        dec = self.policy.decide(
            verdicts,
            coverage=self.coverage(),
            evidence=self.evidence,
            has_treatments=bool(self.treatments),
            stagnant=stagnant,
            tick=tick,
        )
        if dec.action == "treat":
            best = self.best_treatment(verdicts)
            if best is None:
                return Decision("abort",
                                f"{dec.reason}；但候选方案都无法改善判据",
                                dec.confidence, {"tried": [t.key for t in self.treatments]},
                                tick)
            key, delta, detail = best
            payload = {**dec.payload, "treatment": key,
                       "margin_delta": round(delta, 3), "verdicts": detail}
            reason = f"{dec.reason} → 选择「{key}」（裕度 +{delta:.2f}）"
            if self.act:
                iv = next(t for t in self.treatments if t.key == key)
                if self._error_before_act is None:
                    self._error_before_act = self._model_error()
                top_before = {(t.ix, t.iy): t.col.layers[0][1].key
                              for t in self.model.tiles}
                iv.apply(self.model)
                # 执行层真的动了地面：真值世界同步改变，否则下一轮传感器
                # 会读到未处理的地层，把刚做的处理又"同化"回去
                if self.truth is not None:
                    try:
                        iv.apply(self.truth)
                    except Exception:
                        pass
                for (ix, iy), key0 in top_before.items():
                    if self.model.tile(ix, iy).col.layers[0][1].key != key0:
                        self._treated.add((ix, iy))
                self.treatments = tuple(t for t in self.treatments if t.key != key)
                after = {v.key: round(v.margin, 3) for v in self.evaluate()}
                payload["applied"] = True
                payload["verdicts_after_act"] = after
                reason += f"；已执行，复核裕度 {after}"
            return Decision("treat", reason, dec.confidence, payload, tick)
        return dec

    def best_treatment(self, verdicts: list[Verdict]):
        """在当前（已同化的）模型上逐个试算候选方案，取裕度改善最大者。"""
        base = {v.key: v.margin for v in verdicts}
        best = None
        for iv in self.treatments:
            probe = copy.deepcopy(self.model)
            try:
                iv.apply(probe)
            except Exception:                     # 方案越界，跳过
                continue
            after = {v.key: v for v in evaluate(probe, self.criteria, sites=self.sites)}
            delta = min(after[k].margin - base.get(k, 0.0) for k in after)
            if delta <= 0:
                continue
            detail = {k: round(v.margin, 3) for k, v in after.items()}
            if best is None or delta > best[1]:
                best = (iv.key, delta, detail)
        return best

    # -- 整轮 -----------------------------------------------------------
    def run(self, ticks: int | None = None) -> SessionReport:
        limit = ticks or self.max_ticks
        last = None
        stable = 0
        while len(self.ticks) < limit:
            rec = self.tick()
            action = rec["decision"]["action"]
            if action in ("certify", "abort", "escalate"):
                stable = stable + 1 if action == last else 1
                last = action
                if stable >= 2:
                    break
            else:
                stable, last = 0, None

        self.transport.close()
        after = self._snapshot()
        final = Decision(**(self.ticks[-1]["decision"])) if self.ticks else None
        return SessionReport(
            ticks=self.ticks,
            final=final,
            coverage=self.coverage(),
            evidence=self.evidence,
            measurements=self.measurements,
            assimilated=self.assimilator.accepted,
            rejected=self.assimilator.rejected,
            before=self.before,
            after=after,
            meta={
                "seed": self.seed,
                "devices": [d.name for d in self.devices],
                "scope_tiles": len(self.scope_idx),
                "measured_tiles": len(self.measured),
                "model_error_before": self.before_error,
                # 处理过的区块被排除在误差统计外：模型与自然真值的偏离只对
                # 未改造的地面有意义
                "model_error_after": self._model_error(),
                "model_error_at_decision": self._error_before_act,
                "acted": bool(self._error_before_act is not None),
                "policy": self.policy.cfg.to_dict(),
            },
        )

    # -- 内部 -----------------------------------------------------------
    def _tile_near(self, x: float, y: float):
        for t in self.model.tiles:
            if abs(t.x - x) <= self.model.config.cell / 2 and \
               abs(t.y - y) <= self.model.config.cell / 2:
                return t
        return None

    def _snapshot(self) -> dict:
        vs = self.evaluate()
        return {v.key: {"passed": v.passed, "margin": round(v.margin, 3)}
                for v in vs}

    def _model_error(self) -> dict:
        """模型与真值的差距（只统计本次扫描范围，全球误差会被稀释）。

        高程 RMSE、按深度比对的地层参数误差、材料错判率。
        """
        if self.truth is None:
            return {}
        scope = [t for t in (self.scope or self.model.tiles)
                 if (t.ix, t.iy) not in self._treated]
        dz, dc, dphi, mismatch, n = [], [], [], 0, 0
        depths = (0.5, 1.5, 3.0, 6.0)
        for a in scope:
            b = self.truth.tiles[self.model.idx(a.ix, a.iy)]
            dz.append(abs(a.elev - b.elev))
            for d in depths:
                ma, mb = a.col.material_at(d), b.col.material_at(d)
                dc.append(abs(ma.c - mb.c))
                dphi.append(abs(ma.phi - mb.phi))
                mismatch += 0 if ma.key == mb.key else 1
                n += 1
        rmse = (sum(x * x for x in dz) / max(len(dz), 1)) ** 0.5
        return {
            "scope_tiles": len(scope),
            "elev_rmse_m": round(rmse, 3),
            "cohesion_mae_kpa": round(sum(dc) / max(len(dc), 1), 3),
            "phi_mae_deg": round(sum(dphi) / max(len(dphi), 1), 3),
            "material_mismatch": round(mismatch / max(n, 1), 4),
        }


def replay(transport_spec: str, *, model: World, truth: World, criteria,
           devices: list[ScanDevice], **kwargs) -> RobotSession:
    """用回放链路重建一个会话的输入流（链路替换的用法示例）。"""
    from .link import make_transport

    return RobotSession(model=model, truth=truth, criteria=criteria,
                        devices=devices, transport=make_transport(transport_spec),
                        **kwargs)
