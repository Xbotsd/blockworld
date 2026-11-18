"""BLOCKWORLD —— 区块化地质—生态沙盘（可复用库）。

解决的问题：**在给定场地模型下，某个干预措施会不会改变可行性判据，
以及这个结论对不确定输入有多稳；接上机器人后，实测数据能不能改变这个结论。**

公开接口（见 ``blockworld.api``）：

    SiteSpec / GridSpec        场地数据（DEM、钻孔、气候场），真实数据走这里
    build_from(spec)           数据 → World
    Intervention               可序列化的干预（换填、降水、植被替换、改重力……）
    Criteria                   要验证哪些目的、阈值多少
    Study(...).run()           基线 + 每个干预 + 判据 + 变化量
    monte_carlo / morris       失效概率与全局敏感性

闭环（见 ``blockworld.robot``）：

    RobotSession               扫描 → 同化 → 评估 → 判断，逐 tick 可审计
    Supervisor                 规则型决策：认证/处理/复扫/中止/转人工

离线顾问（见 ``blockworld.agent``）：

    Advisor                    本地模型只解释与建议，不参与判定
    Facts.from_report(report)  把一次闭环压成事实包

其余模块（geology/hydrology/vegetation/animals/verifiers）是可替换的引擎实现，
扩展点见 ``blockworld.solvers``。
"""

from . import (agent, animals, api, edits, geology, hydrology, io, materials,
               pft, report, robot, sensitivity, solvers, uncertainty,
               vegetation, verifiers)
from .api import Criteria, Intervention, Study, StudyResult
from .site import GridSpec, SiteSpec, build_from, synthesize
from .world import Cohort, Region, Tile, World, WorldConfig, build

__version__ = "0.3.1"

__all__ = [
    "SiteSpec", "GridSpec", "build_from", "synthesize",
    "Intervention", "Study", "StudyResult", "Criteria",
    "World", "WorldConfig", "Tile", "Region", "Cohort", "build",
    "monte_carlo", "morris",
    "materials", "pft", "geology", "hydrology", "vegetation", "animals",
    "edits", "verifiers", "sensitivity", "solvers", "uncertainty", "io",
    "report", "api", "robot", "agent", "__version__",
]

from .uncertainty import Parameter, Distribution, monte_carlo, morris  # noqa: E402
