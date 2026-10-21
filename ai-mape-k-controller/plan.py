"""
plan.py —— MAPE-K 的 **Plan（P）** 环节：把分析结论转化为副本数决策。

提供四类规划器，与控制器的映射关系：
- `StaticPlanner`        -> Static（按峰值静态配置，不做自适应）
- `HPAPlanner`           -> HPA（复刻 K8s 原生 HPA 的比值算法与稳定窗口策略）
- `RulePlanner`          -> NoAI-MAPE（阈值规则 + 固定安全系数）
- `AdaptivePlanner`      -> AI-MAPE（分位数预测 + 在线自适应安全边际 + 迟滞防抖）

设计要点：
- 规划器不直接操作被管系统，仅返回 `ScaleDecision`，由 execute.py 统一执行；
- 所有在线状态存放于 KnowledgeBase，保证可持久化与可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from analyze import ForecastResult
from knowledge import (DEFAULT_CONFIG, KnowledgeBase, Observation,
                       ScaleDecision, SystemConfig)


class BasePlanner:
    """规划器接口。"""

    name = "base"

    def plan(self, obs: Observation, forecast: ForecastResult,
             knowledge: KnowledgeBase,
             config: SystemConfig = DEFAULT_CONFIG) -> ScaleDecision:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# 1. 静态规划
# --------------------------------------------------------------------------- #
class StaticPlanner(BasePlanner):
    """不做自适应：始终返回预配置的副本数（工程上的“按峰值预留”）。"""

    name = "static"

    def __init__(self, replicas: int = 8):
        self.replicas = replicas

    def plan(self, obs: Observation, forecast: ForecastResult,
             knowledge: KnowledgeBase,
             config: SystemConfig = DEFAULT_CONFIG) -> ScaleDecision:
        return ScaleDecision(
            target_replicas=int(self.clamp(self.replicas, config)),
            reason="static-provisioned",
            forecast_rps=obs.observed_rps,
            meta={"provisioned": self.replicas},
        )

    @staticmethod
    def clamp(n: int, config: SystemConfig) -> int:
        return int(max(config.min_replicas, min(config.max_replicas, n)))


# --------------------------------------------------------------------------- #
# 2. K8s 原生 HPA 算法
# --------------------------------------------------------------------------- #
@dataclass
class HPAConfig:
    """HPA 关键可调参数（与 kube-controller-manager 语义对齐）。"""

    target_cpu_util: float = 60.0        # targetCPUUtilizationPercentage
    tolerance: float = 0.1               # --horizontal-pod-autoscaler-tolerance
    up_stabilization: int = 0            # scaleUp 稳定窗口（周期数），0=立即
    down_stabilization: int = 12         # scaleDown 稳定窗口，默认 5min / 5s = 60 step
    cpu_sync_period: int = 1             # 指标采样周期（周期数）


class HPAPlanner(BasePlanner):
    """复刻 Kubernetes HPA v2 的控制律。

        desired = ceil(current × observed / target)
    并施加：容差死区、缩容稳定窗口（取窗口内最小推荐值）、副本上下界。
    该控制律是**纯反应式**的——只在指标已经恶化后才动作，因此天然滞后于突发流量。
    """

    name = "hpa"

    def __init__(self, hpa: Optional[HPAConfig] = None):
        self.hpa = hpa or HPAConfig()
        self._recommendations: list = []

    def plan(self, obs: Observation, forecast: ForecastResult,
             knowledge: KnowledgeBase,
             config: SystemConfig = DEFAULT_CONFIG) -> ScaleDecision:
        h = self.hpa
        current = max(1, obs.ready_replicas)
        metric = max(0.0, obs.cpu_util)

        ratio = metric / max(h.target_cpu_util, 1e-6)
        desired = current
        if ratio > 1.0 + h.tolerance or ratio < 1.0 - h.tolerance:
            desired = int(math.ceil(current * ratio))
        else:
            desired = current

        desired = int(max(config.min_replicas, min(config.max_replicas, desired)))
        self._recommendations.append(desired)

        if desired > current:
            # 扩容：取窗口内最大推荐值（快速响应）
            window = self._recommendations[-max(1, h.up_stabilization):]
            target = max(window)
            reason = "hpa-scale-up"
        elif desired < current:
            # 缩容：取窗口内最小推荐值（保守，防止过早缩容）
            window = self._recommendations[-max(1, h.down_stabilization):]
            target = min(window)
            reason = "hpa-scale-down(stabilized)"
        else:
            target = current
            reason = "hpa-within-tolerance"

        return ScaleDecision(
            target_replicas=int(target),
            reason=reason,
            forecast_rps=obs.observed_rps,
            meta={"cpu_util": metric, "ratio": ratio, "raw_desired": desired},
        )

    def reset(self) -> None:
        self._recommendations = []


# --------------------------------------------------------------------------- #
# 3. NoAI-MAPE：规则规划
# --------------------------------------------------------------------------- #
class RulePlanner(BasePlanner):
    """基于阈值规则与固定安全系数的规划器（无在线学习）。

    容量估算：needed = ceil(pred × (1 + margin) / pod_capacity)
    再叠加：变化死区（deadzone）与冷却周期（cooldown）以抑制抖动。
    """

    name = "rule"

    def __init__(self, margin: float = 0.15, deadzone: float = 0.15,
                 cooldown_steps: int = 2):
        self.margin = margin
        self.deadzone = deadzone
        self.cooldown_steps = cooldown_steps
        self._last_change_t: int = -10 ** 9
        self._last_target: Optional[int] = None

    def plan(self, obs: Observation, forecast: ForecastResult,
             knowledge: KnowledgeBase,
             config: SystemConfig = DEFAULT_CONFIG) -> ScaleDecision:
        load = forecast.point if math.isfinite(forecast.point) else obs.observed_rps
        needed = int(math.ceil(load * (1.0 + self.margin) / config.pod_capacity_rps))
        needed = int(max(config.min_replicas, min(config.max_replicas, needed)))

        current = obs.replicas
        reason = "rule-hold"

        if self._last_target is not None and current != self._last_target:
            # 期望值已被外部改变，直接跟随
            target = needed
            reason = "rule-resync"
        elif abs(needed - current) / max(current, 1) < self.deadzone:
            target = current
            reason = "rule-deadzone"
        elif obs.t - self._last_change_t < self.cooldown_steps:
            target = current
            reason = "rule-cooldown"
        else:
            target = needed
            reason = "rule-scale-up" if needed > current else "rule-scale-down"

        if target != current:
            self._last_change_t = obs.t
        self._last_target = target

        return ScaleDecision(
            target_replicas=int(target),
            reason=reason,
            forecast_rps=float(load),
            meta={"needed": needed, "margin": self.margin},
        )

    def reset(self) -> None:
        self._last_change_t = -10 ** 9
        self._last_target = None


# --------------------------------------------------------------------------- #
# 4. AI-MAPE：自适应规划
# --------------------------------------------------------------------------- #
class AdaptivePlanner(BasePlanner):
    """具备在线学习能力的规划器。

    三项关键机制：
    1. **分位数容量估算**：用预测分布上界（point + z·σ）而非点估计，
       对突发流量具备天然冗余；
    2. **提前量补偿**：预测视界 = 副本冷启动周期数 + 1，用“副本就绪那一刻”
       的预测负载做决策，消除冷启动期间的容量缺口；
    3. **在线自适应安全边际**：以滚动 SLO 违反率为反馈信号，乘性更新安全边际
       与目标利用率——违反上升即加冗余，长期安全则回收冗余以压低成本。
       另配非对称迟滞：扩容立即生效，缩容需连续满足 hold 周期，抑制抖动。
    """

    name = "adaptive"

    def __init__(self, margin_init: float = 1.05, margin_min: float = 0.95,
                 margin_max: float = 1.35, target_util_init: float = 0.92,
                 target_util_min: float = 0.70, target_util_max: float = 0.97,
                 hold_steps: int = 4, gain: float = 1.05, decay: float = 0.985,
                 sample_every: int = 15):
        self.margin = margin_init
        self.margin_min, self.margin_max = margin_min, margin_max
        self.target_util = target_util_init
        self.tu_min, self.tu_max = target_util_min, target_util_max
        self.hold_steps = hold_steps
        self.gain, self.decay = gain, decay
        self.sample_every = sample_every
        self._down_streak: int = 0
        self._pending_down: Optional[int] = None
        self._margin_trace: list = []
        self._vr_samples: list = []          # 违反率采样，用于识别“加冗余无效”
        self._tick = 0

    # ---------------- 边际收益检测 ----------------
    def _is_stagnant(self) -> bool:
        """冗余已封顶但违反率不再改善 —— 说明容量缺口是结构性的
        （冷启动延迟 / 突发不可预知），继续加码无收益。"""
        s = self._vr_samples
        if len(s) < 3:
            return False
        return s[-1] >= min(s[:-1]) - 0.005

    # ---------------- 自适应更新 ----------------
    def _adapt(self, knowledge: KnowledgeBase, obs: Observation) -> None:
        vr = knowledge.rolling_violation_rate(window=15)
        util = knowledge.rolling_mean("cpu_util", window=10) / 100.0

        self._tick += 1
        if self._tick % self.sample_every == 0:
            self._vr_samples.append(vr)
            if len(self._vr_samples) > 8:
                self._vr_samples.pop(0)

        if vr > 0.01:                       # 违反偏多：增加冗余
            if self.margin >= self.margin_max * 0.98 and self._is_stagnant():
                # 冗余已封顶仍无改善：停止加码，缓慢回收以止损（避免无谓的资源浪费）
                self.margin = max(self.margin_min, self.margin * (2.0 - self.decay))
                self.target_util = min(self.tu_max, self.target_util * 1.01)
            else:
                self.margin = min(self.margin_max, self.margin * self.gain)
                self.target_util = max(self.tu_min, self.target_util * (2.0 - self.gain))
        elif vr < 0.015 and util < 0.85:    # 长期安全且有余量：回收冗余
            self.margin = max(self.margin_min, self.margin * self.decay)
            self.target_util = min(self.tu_max, self.target_util * (1.0 + (1.0 - self.decay)))
        self._margin_trace.append(self.margin)
        knowledge.put("adaptive_margin", self.margin)
        knowledge.put("adaptive_target_util", self.target_util)

    # ---------------- 规划 ----------------
    def plan(self, obs: Observation, forecast: ForecastResult,
             knowledge: KnowledgeBase,
             config: SystemConfig = DEFAULT_CONFIG) -> ScaleDecision:
        self._adapt(knowledge, obs)

        load = forecast.upper if math.isfinite(forecast.upper) else obs.observed_rps
        load = max(load, obs.observed_rps)          # 不低于当前观测（防止缩容过激）
        eff_capacity = config.pod_capacity_rps * self.target_util
        needed = int(math.ceil(load * self.margin / max(eff_capacity, 1e-6)))

        # ---- 反应式兜底（safeguard）----
        # 预测器不可能预知所有突发；当 SLO 已逼近或 CPU 高位时绕过预测，
        # 直接按当前观测以更保守的单副本容量补齐，避免突发期的容量缺口。
        emergency = False
        if obs.cpu_util > 90.0 or obs.p95_latency_ms > 0.70 * config.slo_latency_ms:
            guard = int(math.ceil(obs.observed_rps /
                                  max(config.pod_capacity_rps * 0.88, 1e-6)))
            if guard > needed:
                needed, emergency = guard, True

        needed = int(max(config.min_replicas, min(config.max_replicas, needed)))

        current = obs.replicas
        if needed > current:
            self._down_streak = 0
            self._pending_down = None
            target = needed
            reason = "ai-scale-up(emergency)" if emergency else "ai-scale-up(preactive)"
        elif needed < current:
            # 非对称迟滞：缩容需连续 hold_steps 个周期一致
            if self._pending_down == needed:
                self._down_streak += 1
            else:
                self._pending_down, self._down_streak = needed, 1
            if self._down_streak >= self.hold_steps:
                target, reason = needed, "ai-scale-down(hysteresis-satisfied)"
                self._down_streak = 0
                self._pending_down = None
            else:
                target, reason = current, "ai-hold(hysteresis)"
        else:
            self._down_streak = 0
            self._pending_down = None
            target, reason = current, "ai-steady"

        return ScaleDecision(
            target_replicas=int(target),
            reason=reason,
            forecast_rps=float(load),
            meta={
                "needed": needed,
                "margin": round(self.margin, 4),
                "target_util": round(self.target_util, 4),
                "sigma": round(forecast.sigma, 3),
                "emergency": emergency,
            },
        )

    def reset(self) -> None:
        self._down_streak = 0
        self._pending_down = None
        self._margin_trace = []
        self._vr_samples = []
        self._tick = 0


def build_planner(kind: str, **kwargs) -> BasePlanner:
    """规划器工厂。kind ∈ {static, hpa, rule, adaptive}"""
    table = {
        "static": StaticPlanner,
        "hpa": HPAPlanner,
        "rule": RulePlanner,
        "adaptive": AdaptivePlanner,
    }
    if kind not in table:
        raise ValueError(f"未知规划器类型 '{kind}'，可选：{list(table)}")
    if kind == "hpa" and "hpa" in kwargs:
        cfg = kwargs.pop("hpa")
        return HPAPlanner(cfg)
    return table[kind](**kwargs)
