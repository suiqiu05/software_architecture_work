"""
monitor.py —— MAPE-K 的 **Monitor（M）** 环节 + 被管系统（Plant）仿真模型。

包含两部分：
1. `ApplicationPlant`：被管对象（K8s Deployment + Service）的动力学模型，
   负责维护副本状态、按排队论计算时延/CPU/错误率，并模拟副本冷启动与退出延迟；
2. `Monitor`：每周期从 Plant 采样，叠加观测噪声后产出 `Observation` 写入知识库。

时延模型：采用 Whitt(1993) 对 M/M/N 队列的等待时间近似——
    Wq ≈ (1/μ_total) · ρ^(√(2(N+1))−1) / (1−ρ)
该近似同时刻画了两件事：利用率升高时延指数恶化；副本数增大时同利用率下排队更短。
这正是不同控制器拉开差距的物理来源。
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from knowledge import DEFAULT_CONFIG, KnowledgeBase, Observation, SystemConfig

_P95_QUANTILE = -math.log(0.05)  # ≈2.996，指数分布尾部的 p95 系数


# --------------------------------------------------------------------------- #
# 被管系统
# --------------------------------------------------------------------------- #
class ApplicationPlant:
    """被管应用：接受期望副本数，产出真实服务质量指标。"""

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, seed: int = 0):
        self.cfg = config
        self.rng = random.Random(seed)
        self.desired_replicas: int = config.initial_replicas
        self.ready_replicas: int = config.initial_replicas
        self.step: int = 0
        # (生效周期, 副本增量)：模拟冷启动与优雅退出延迟
        self._pending: "Deque[Tuple[int, int]]" = deque()
        self.scale_events: List[Dict[str, object]] = []

    # ---------------- 控制面 ----------------
    def apply_replicas(self, target: int, reason: str = "") -> int:
        """设置 spec.replicas；返回实际产生的增量（0 表示无变更）。"""
        target = int(max(self.cfg.min_replicas, min(self.cfg.max_replicas, target)))
        delta = target - self.desired_replicas
        if delta == 0:
            return 0
        self.desired_replicas = target
        delay = self.cfg.startup_steps if delta > 0 else self.cfg.shutdown_steps
        self._pending.append((self.step + delay, delta))
        self.scale_events.append({
            "t": self.step, "from": self.desired_replicas - delta,
            "to": target, "delta": delta, "reason": reason,
            "effective_at": self.step + delay,
        })
        return delta

    def advance(self, t: Optional[int] = None) -> None:
        """推进一个周期，结算到期的副本变更。"""
        self.step = (self.step + 1) if t is None else t
        while self._pending and self._pending[0][0] <= self.step:
            _, delta = self._pending.popleft()
            self.ready_replicas = int(max(
                self.cfg.min_replicas,
                min(self.cfg.max_replicas, self.ready_replicas + delta)
            ))

    # ---------------- 数据面 ----------------
    def serve(self, arrival_rps: float) -> Dict[str, float]:
        """给定到达速率，计算本周期的服务质量指标。"""
        cfg = self.cfg
        n = max(1.0, float(self.ready_replicas))
        capacity = n * cfg.pod_capacity_rps
        rho = min(arrival_rps / max(capacity, 1e-6), cfg.rho_clamp)

        # Whitt 近似的 M/M/N 平均等待时间（秒）
        exponent = math.sqrt(2.0 * (n + 1.0)) - 1.0
        wait_mean_s = (1.0 / max(capacity, 1e-6)) * \
                      (rho ** exponent) / max(1.0 - rho, 1e-6)

        p95_ms = cfg.base_latency_ms + cfg.service_time_ms + wait_mean_s * 1000.0 * _P95_QUANTILE
        # 观测抖动
        p95_ms *= 1.0 + self.rng.gauss(0.0, cfg.noise_cv * 0.5)
        p95_ms = float(max(cfg.base_latency_ms, min(p95_ms, cfg.max_latency_ms)))

        # 超时型错误：以 2s 超时为界做 logistic 上升
        timeout_ms = 2000.0
        error_rate = 1.0 / (1.0 + math.exp(-(p95_ms - timeout_ms * 0.8) /
                                           max(timeout_ms * 0.15, 1e-6)))

        cpu_util = 100.0 * min(1.0, arrival_rps / max(capacity, 1e-6))
        cpu_util = float(max(0.0, min(100.0, cpu_util + self.rng.gauss(0.0, 1.2))))

        return {
            "p95_latency_ms": p95_ms,
            "cpu_util": cpu_util,
            "error_rate": float(min(1.0, max(0.0, error_rate))),
            "utilization": float(rho),
            "capacity_rps": capacity,
        }

    # ---------------- 状态 ----------------
    def snapshot(self) -> Dict[str, float]:
        return {
            "desired_replicas": self.desired_replicas,
            "ready_replicas": self.ready_replicas,
            "capacity_rps": self.ready_replicas * self.cfg.pod_capacity_rps,
        }


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #
class Monitor:
    """周期性采样：Plant -> Observation -> KnowledgeBase。"""

    def __init__(self, plant: ApplicationPlant, knowledge: Optional[KnowledgeBase] = None,
                 config: SystemConfig = DEFAULT_CONFIG, seed: int = 0):
        self.plant = plant
        # 注意：KnowledgeBase 定义了 __len__，此处不可用 `or` 做默认值判断
        # （空知识库会被判为 falsy 而丢失外部注入的实例）。
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase(config=config)
        self.cfg = config
        self.rng = random.Random(seed)
        self._observations: List[Observation] = []

    def sample(self, t: int, arrival_rps: float) -> Observation:
        """推进被管系统一个周期并采集一次观测。"""
        self.plant.advance(t)
        quality = self.plant.serve(arrival_rps)

        noise = self.rng.gauss(0.0, self.cfg.noise_cv)
        observed_rps = max(0.0, arrival_rps * (1.0 + noise))

        obs = Observation(
            t=t,
            arrival_rps=float(arrival_rps),
            observed_rps=float(observed_rps),
            replicas=self.plant.desired_replicas,
            ready_replicas=self.plant.ready_replicas,
            cpu_util=float(quality["cpu_util"]),
            p95_latency_ms=float(quality["p95_latency_ms"]),
            error_rate=float(quality["error_rate"]),
        )
        self._observations.append(obs)
        self.knowledge.push(obs)
        return obs

    # ---------------- 窗口统计（供 HPA 等反应式控制器使用） ----------------
    def window_stat(self, field_name: str, window: int, stat: str = "mean") -> float:
        vals = self.knowledge.series(field_name, n=window)
        if not vals:
            return 0.0
        if stat == "mean":
            return sum(vals) / len(vals)
        if stat == "max":
            return max(vals)
        if stat == "min":
            return min(vals)
        raise ValueError(f"不支持的窗口统计：{stat}")

    @property
    def observations(self) -> List[Observation]:
        return list(self._observations)
