"""
knowledge.py —— MAPE-K 环路中的 **Knowledge（知识库）** 组件。

职责：
1. 承载被管系统的领域常量与 SLO 定义（SystemConfig）；
2. 定义环路各组件之间流转的统一数据结构（Observation / ScaleDecision）；
3. 为 Analyze / Plan 组件提供可读写的历史时序与在线模型状态（KnowledgeBase）。

设计约束：
- 本文件处于依赖树的最底层，不 import 任何本项目模块，避免循环依赖；
- 所有数据结构均为可 pickle 的纯数据类，便于 main_exp.py 使用进程池并行实验。
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# 1. 系统与被控对象常量
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SystemConfig:
    """被管系统（K8s Deployment + Service）的仿真参数与 SLO 定义。

    所有控制器共享同一份配置，保证实验的可比性。
    """

    # --- 时间尺度 ---
    step_seconds: float = 5.0            # 一个控制周期对应的真实时长（秒）
    startup_steps: int = 2               # 新副本从创建到可服务所需周期数（冷启动 / 镜像拉取）
    shutdown_steps: int = 1              # 副本从决定缩容到完全退出所需周期数

    # --- 容量模型 ---
    pod_capacity_rps: float = 60.0       # 单副本可持续吞吐（req/s），超出即排队
    service_time_ms: float = 8.0         # 单请求 CPU 服务时间
    base_latency_ms: float = 12.0        # 空载固定开销（网络 / 序列化 / GC）

    # --- 副本边界 ---
    min_replicas: int = 1
    max_replicas: int = 40
    initial_replicas: int = 3

    # --- SLO ---
    slo_latency_ms: float = 300.0        # p95 时延目标
    slo_violation_tolerance: float = 0.0 # 允许的违反比例（0 = 严格）

    # --- 排队模型数值安全 ---
    rho_clamp: float = 0.995             # 利用率上限，避免 1/(1-rho) 发散
    max_latency_ms: float = 4000.0       # 时延饱和上限（视为超时熔断）
    noise_cv: float = 0.06               # 观测噪声变异系数

    # --- 成本模型 ---
    pod_cost_per_step: float = 1.0       # 单副本单周期资源成本（pod-second 归一）
    scale_cost_per_change: float = 2.0   # 单次扩缩容折算成本（抑制抖动）

    # --- 控制器公共约束 ---
    cooldown_steps: int = 2              # 两次扩缩容之间的最小间隔（防抖）

    @property
    def steps_per_minute(self) -> int:
        return max(1, int(round(60.0 / self.step_seconds)))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = SystemConfig()


# --------------------------------------------------------------------------- #
# 2. 环路数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """Monitor 每个控制周期采集的一次系统观测（M -> A 的输入）。"""

    t: int                      # 周期序号
    arrival_rps: float          # 到达速率（真实负载，控制器仅能看到含噪观测）
    observed_rps: float         # 含噪观测值（控制器实际可见）
    replicas: int               # 期望副本数（spec.replicas）
    ready_replicas: int         # 已就绪可服务副本数
    cpu_util: float             # 平均 CPU 利用率（0~100），HPA 的目标指标
    p95_latency_ms: float       # p95 时延
    error_rate: float           # 错误率（超时请求占比 0~1）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def violates_slo(self) -> bool:
        return self.p95_latency_ms > DEFAULT_CONFIG.slo_latency_ms


@dataclass
class ScaleDecision:
    """Plan 组件的输出（P -> E 的输入）。"""

    target_replicas: int
    reason: str = ""
    forecast_rps: float = 0.0                       # 该决策所依据的负载预测值
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ControlStep:
    """一次完整 MAPE-K 环路的落库记录，供 Evaluator 与可视化消费。"""

    t: int
    arrival_rps: float
    observed_rps: float
    forecast_rps: float = 0.0          # 无预测能力的控制器填 NaN
    replicas: int = 0
    ready_replicas: int = 0
    cpu_util: float = 0.0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0
    decision_reason: str = ""
    pod_cost: float = 0.0
    scale_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 3. 知识库
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    """MAPE-K 的共享知识库。

    提供三类能力：
    - **时序记忆**：滚动窗口内的观测历史，支持按字段抽取序列；
    - **模型暂存**：Analyze / Plan 的在线模型状态（权重、平滑量、自适应 margin）；
    - **SLO 反馈**：滚动违反率、残差标准差等供自适应算法读取的统计量。
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, window: int = 600):
        self.config = config
        self.window = window
        self._history: "deque[Observation]" = deque(maxlen=window)
        self._model_state: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []

    # ---------------- 时序记忆 ----------------
    def push(self, obs: Observation) -> None:
        self._history.append(obs)

    @property
    def history(self) -> List[Observation]:
        return list(self._history)

    def series(self, field_name: str, n: Optional[int] = None) -> List[float]:
        """抽取指定字段的时序（最近的 n 个点）。"""
        items = list(self._history)
        if n is not None:
            items = items[-n:]
        return [float(getattr(o, field_name)) for o in items]

    def last(self) -> Optional[Observation]:
        return self._history[-1] if self._history else None

    def __len__(self) -> int:
        return len(self._history)

    # ---------------- 模型状态 ----------------
    def put(self, key: str, value: Any) -> None:
        self._model_state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._model_state.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        return self._model_state.pop(key, default)

    def update_state(self, **kwargs: Any) -> None:
        self._model_state.update(kwargs)

    def clear(self) -> None:
        """清空知识库（reset 场景使用，保持对象身份不变）。"""
        self._history.clear()
        self._model_state.clear()
        self._events.clear()

    # ---------------- 事件日志 ----------------
    def log_event(self, kind: str, **payload: Any) -> None:
        self._events.append({"kind": kind, **payload})

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    # ---------------- 统计反馈 ----------------
    def rolling_violation_rate(self, window: int = 30) -> float:
        """最近 window 个周期的 SLO 违反率。"""
        items = list(self._history)[-window:]
        if not items:
            return 0.0
        hit = sum(1 for o in items if o.p95_latency_ms > self.config.slo_latency_ms)
        return hit / len(items)

    def rolling_mean(self, field_name: str, window: int = 10) -> float:
        vals = self.series(field_name, n=window)
        return sum(vals) / len(vals) if vals else 0.0

    def residual_std(self, window: int = 30, default: float = 1.0) -> float:
        """最近预测残差的标准差，用于构造预测区间上界。"""
        res = [r for r in self.get("residual_buffer", [])][-window:]
        if len(res) < 2:
            return default
        mu = sum(res) / len(res)
        var = sum((x - mu) ** 2 for x in res) / (len(res) - 1)
        return math.sqrt(max(var, 1e-9))

    def push_residual(self, residual: float, buffer_size: int = 100) -> None:
        buf = self._model_state.setdefault("residual_buffer", deque(maxlen=buffer_size))
        buf.append(float(residual))

    # ---------------- 持久化 ----------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "model_state": {k: v for k, v in self._model_state.items()
                            if k != "residual_buffer"},
            "events": self._events,
            "history_len": len(self._history),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2, default=str)


def summarize_series(values: List[float]) -> Dict[str, float]:
    """对任意序列给出常用统计量，供评估报告使用。"""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "min": 0.0}

    def _q(q: float) -> float:
        s = sorted(values)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return float(s[idx])

    return {
        "mean": sum(values) / len(values),
        "p50": _q(0.50),
        "p95": _q(0.95),
        "p99": _q(0.99),
        "max": max(values),
        "min": min(values),
    }
