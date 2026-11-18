"""机器人在环：扫描 → 同化 → 自主判断。

三个可替换点，对应三类真实部署差异：

    devices.ScanDevice     载荷（LiDAR / CPT / 探针 / 真实 SDK）
    link.Transport         链路（进程内 / JSONL 录制回放 / TCP）
    agent.Supervisor       判断策略（阈值与规则）

最小用法::

    from blockworld.robot import RobotSession, Supervisor, make_device

    session = RobotSession(
        model=model_world, truth=truth_world, criteria=criteria,
        devices=[make_device("lidar"), make_device("cpt")],
        treatments=(gravel_replacement,),
    )
    report = session.run(ticks=8)
    report.final.action        # certify / rescan / treat / abort
"""

from .agent import PolicyConfig, Supervisor, rules
from .assimilate import Assimilator
from .devices import (DEVICES, ReplayDevice, ScanDevice, SimCPT, SimLiDAR,
                      SimProbe, device_table, make_device, register_device)
from .link import (JsonlReplay, JsonlWriter, Loopback, Transport,
                   make_transport, register_transport)
from .session import RobotSession, replay
from .types import Assimilated, Decision, Measurement, SessionReport

__all__ = [
    "RobotSession", "replay", "SessionReport",
    "Supervisor", "PolicyConfig", "rules",
    "Assimilator",
    "Measurement", "Assimilated", "Decision",
    "ScanDevice", "SimLiDAR", "SimCPT", "SimProbe", "ReplayDevice",
    "make_device", "register_device", "device_table", "DEVICES",
    "Transport", "Loopback", "JsonlWriter", "JsonlReplay",
    "make_transport", "register_transport",
]
