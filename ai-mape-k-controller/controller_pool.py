"""
controller_pool.py —— 控制器池（新增）：封装四组待对比的控制器实例。

四组控制器的能力矩阵：

| 控制器        | Monitor | Analyze            | Plan                 | 预测视界 | 在线学习 |
|---------------|---------|--------------------|----------------------|----------|----------|
| Static        | ✔       | ✗（无预测）        | 固定预留              | —        | ✗        |
| HPA           | ✔       | ✗（无预测）        | 原生比值控制律        | —        | ✗        |
| NoAI-MAPE     | ✔       | 启发式平滑+趋势    | 阈值规则 + 固定余量   | 1        | ✗        |
| AI-MAPE       | ✔       | 在线集成预测        | 分位数 + 自适应余量   | 冷启动+1 | ✔        |

统一接口：
    controller.observe(obs)  # 回灌真实值（学习 + 预测误差对齐）
    controller.decide(obs)   # 产出 ScaleDecision

预测误差评估：控制器内部维护 (target_step, predicted) 队列，待真实值到达后
自动对齐产出 (actual, predicted) 样本，供 Evaluator 计算 RMSE / MAPE。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from analyze import (BaseAnalyzer, EnsembleAnalyzer, ForecastResult,
                     HeuristicAnalyzer, NoForecastAnalyzer, forecast_report)
from knowledge import (DEFAULT_CONFIG, KnowledgeBase, Observation,
                       ScaleDecision, SystemConfig)
from plan import (AdaptivePlanner, BasePlanner, HPAPlanner, RulePlanner,
                  StaticPlanner)


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #
class BaseController:
    """控制器基类：组合 Analyzer + Planner，完成 A→P 两段决策。"""

    key: str = "base"
    label: str = "Base"
    has_forecast: bool = False

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG,
                 analyzer: Optional[BaseAnalyzer] = None,
                 planner: Optional[BasePlanner] = None,
                 horizon: int = 1,
                 knowledge: Optional[KnowledgeBase] = None,
                 eval_horizon: int = 1):
        self.cfg = config
        self.analyzer = analyzer or NoForecastAnalyzer()
        self.planner = planner or StaticPlanner(config.initial_replicas)
        self.horizon = horizon
        # 决策视界之外，额外产出 1 步预测用于**同口径**比较不同控制器的预测能力
        # （决策视界越长，预测误差天然越大，直接跨视界比较 RMSE 并不公平）
        self.eval_horizon = eval_horizon
        self.knowledge = (knowledge if knowledge is not None
                          else KnowledgeBase(config=config))
        self.has_forecast = self.analyzer.supports_forecast
        self._pending: "Deque[Tuple[int, float]]" = deque()
        self._eval_pending: "Deque[Tuple[int, float]]" = deque()
        self._pairs: List[Tuple[float, float]] = []     # (actual, predicted) @horizon
        self._eval_pairs: List[Tuple[float, float]] = []  # (actual, predicted) @eval_horizon
        self._forecast_trace: List[Optional[float]] = []  # 每周期输出的预测值
        self.last_forecast: Optional[ForecastResult] = None

    # ---------------- 学习 / 误差对齐 ----------------
    def observe(self, obs: Observation) -> None:
        """把真实到达速率回灌给分析器，并结算到期的预测误差样本。"""
        for queue, sink in ((self._pending, self._pairs),
                            (self._eval_pending, self._eval_pairs)):
            while queue and queue[0][0] <= obs.t:
                target_t, predicted = queue.popleft()
                if target_t == obs.t:
                    sink.append((obs.observed_rps, predicted))
        self.analyzer.observe(obs.observed_rps,
                              self.last_forecast.point if self.last_forecast else float("nan"),
                              self.knowledge)

    # ---------------- 决策 ----------------
    def decide(self, obs: Observation) -> ScaleDecision:
        # 注：观测写入知识库由 Monitor 负责（二者共享同一 KnowledgeBase 实例），
        # 此处只读取，避免同一周期重复入栈。
        forecast = self.analyzer.predict(self.knowledge, self.horizon, self.cfg)
        self.last_forecast = forecast
        point_is_valid = not math.isnan(forecast.point)
        self._forecast_trace.append(float(forecast.point) if point_is_valid else None)
        if self.has_forecast and point_is_valid:
            self._pending.append((obs.t + self.horizon, float(forecast.point)))
            if self.horizon == self.eval_horizon:
                self._eval_pending.append((obs.t + self.horizon, float(forecast.point)))
            else:
                f_eval = self.analyzer.predict(self.knowledge, self.eval_horizon, self.cfg)
                if not math.isnan(f_eval.point):
                    self._eval_pending.append((obs.t + self.eval_horizon,
                                               float(f_eval.point)))

        decision = self.planner.plan(obs, forecast, self.knowledge, self.cfg)
        decision.forecast_rps = float(forecast.point) if forecast.point == forecast.point else float("nan")
        return decision

    # ---------------- 输出 ----------------
    @staticmethod
    def _empty_errors() -> Dict[str, float]:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"),
                "smape": float("nan"), "bias": float("nan"), "n": 0.0}

    @property
    def forecast_errors(self) -> Dict[str, float]:
        """基于已对齐样本计算预测误差画像（RMSE / MAPE / ...），视界 = 决策视界。"""
        if not self._pairs:
            return self._empty_errors()
        actual = [p[0] for p in self._pairs]
        pred = [p[1] for p in self._pairs]
        return forecast_report(actual, pred)

    @property
    def forecast_errors_h1(self) -> Dict[str, float]:
        """统一视界（h=1）下的预测误差，用于跨控制器公平比较。"""
        if not self._eval_pairs:
            return self._empty_errors()
        actual = [p[0] for p in self._eval_pairs]
        pred = [p[1] for p in self._eval_pairs]
        return forecast_report(actual, pred)

    @property
    def forecast_pairs(self) -> List[Tuple[float, float]]:
        return list(self._pairs)

    @property
    def forecast_pairs_h1(self) -> List[Tuple[float, float]]:
        return list(self._eval_pairs)

    @property
    def forecast_trace(self) -> List[Optional[float]]:
        return list(self._forecast_trace)

    def reset(self) -> None:
        self.analyzer.reset()
        self.planner.reset()
        # 清空而非重建：Monitor 持有同一 KnowledgeBase 实例，重建会断开共享
        self.knowledge.clear()
        self._pending.clear()
        self._eval_pending.clear()
        self._pairs.clear()
        self._eval_pairs.clear()
        self._forecast_trace.clear()
        self.last_forecast = None

    def describe(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "analyzer": self.analyzer.name,
            "planner": self.planner.name,
            "horizon": self.horizon,
            "has_forecast": self.has_forecast,
        }


# --------------------------------------------------------------------------- #
# 1. Static：静态副本
# --------------------------------------------------------------------------- #
class StaticController(BaseController):
    """按峰值预配置副本数，全程不调整。代表“过度预留换稳定”的基线策略。"""

    key, label = "static", "Static (fixed replicas)"

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG, replicas: int = 8):
        super().__init__(config=config, analyzer=NoForecastAnalyzer(),
                         planner=StaticPlanner(replicas), horizon=1)
        self.replicas = replicas


# --------------------------------------------------------------------------- #
# 2. HPA：K8s 原生水平扩缩容
# --------------------------------------------------------------------------- #
class HPAController(BaseController):
    """K8s 原生 HPA：基于 CPU 利用率的比值控制律，纯反应式，无预测。"""

    key, label = "hpa", "K8s HPA (reactive CPU)"

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG,
                 target_cpu_util: float = 60.0, down_stabilization: Optional[int] = None):
        from plan import HPAConfig
        steps_per_5min = max(1, int(round(300.0 / config.step_seconds)))
        cfg_hpa = HPAConfig(
            target_cpu_util=target_cpu_util,
            down_stabilization=down_stabilization or steps_per_5min,
        )
        super().__init__(config=config, analyzer=NoForecastAnalyzer(),
                         planner=HPAPlanner(cfg_hpa), horizon=1)
        self.hpa_cfg = cfg_hpa


# --------------------------------------------------------------------------- #
# 3. NoAI-MAPE：规则式自适应
# --------------------------------------------------------------------------- #
class NoAIMapeController(BaseController):
    """传统 MAPE-K：启发式分析 + 规则规划，参数固定、无在线学习。"""

    key, label = "noai_mape", "NoAI-MAPE (rule-based)"

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG,
                 margin: float = 0.15, horizon: int = 1):
        super().__init__(config=config,
                         analyzer=HeuristicAnalyzer(),
                         planner=RulePlanner(margin=margin),
                         horizon=horizon)


# --------------------------------------------------------------------------- #
# 4. AI-MAPE：在线学习自适应
# --------------------------------------------------------------------------- #
class AIMapeController(BaseController):
    """AI-MAPE：在线集成预测 + 自适应安全边际 + 提前量补偿。

    预测视界取 `startup_steps + 1`，即用“新副本就绪那一刻”的预测负载决策，
    从原理上消除了冷启动窗口内的容量缺口。
    """

    key, label = "ai_mape", "AI-MAPE (learning-based)"

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG,
                 horizon: Optional[int] = None):
        h = horizon or (config.startup_steps + 1)
        super().__init__(config=config,
                         analyzer=EnsembleAnalyzer(),
                         planner=AdaptivePlanner(),
                         horizon=h)

    def apply_hints(self, hints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """把外部（LLM 顾问 / 人工）给出的初始参数建议注入规划器。

        这是大模型介入控制环的**唯一合法入口**：只调整低频、有界的安全参数，
        不改变在线学习算法本身，也不参与逐周期决策。
        """
        applied: Dict[str, Any] = {}
        if not hints:
            return applied

        # 边界一律取自 AdaptivePlanner.param_spec()：hints 可能来自 LLM、人工或
        # Web 页面，不能假定上游已经钳制过。此前这里对 margin / target_util 直接
        # 赋值，一旦上游放行越界值就会污染控制环（且不会被 _adapt 纠正下限之外的量）。
        spec = AdaptivePlanner.param_spec()
        planner = self.planner
        if isinstance(planner, AdaptivePlanner):
            for key in ("margin", "target_util"):
                if key not in hints:
                    continue
                try:
                    raw = float(hints[key])
                except (TypeError, ValueError):
                    continue
                lo, hi = spec[key]["min"], spec[key]["max"]
                v = max(lo, min(hi, raw))
                setattr(planner, key, v)
                applied[key] = v
                if abs(v - raw) > 1e-9:
                    applied[key + "_clamped_from"] = raw
        if "horizon" in hints:
            try:
                h = int(float(hints["horizon"]))
            except (TypeError, ValueError):
                h = self.horizon
            self.horizon = int(max(spec["horizon"]["min"],
                                   min(spec["horizon"]["max"], h)))
            applied["horizon"] = self.horizon
        return applied


# --------------------------------------------------------------------------- #
# 控制器池
# --------------------------------------------------------------------------- #
def build_controller_pool(config: SystemConfig = DEFAULT_CONFIG,
                          static_replicas: Optional[int] = None,
                          keys: Optional[Sequence[str]] = None
                          ) -> Dict[str, BaseController]:
    """构建待对比的控制器集合。

    `static_replicas` 未指定时，按“峰值负载 / 单副本容量 × 1.15”估算，
    模拟工程上常见的按峰值静态预留。
    """
    replicas = static_replicas if static_replicas is not None else 12
    pool: Dict[str, BaseController] = {
        "static": StaticController(config, replicas=replicas),
        "hpa": HPAController(config),
        "noai_mape": NoAIMapeController(config),
        "ai_mape": AIMapeController(config),
    }
    if keys:
        pool = {k: v for k, v in pool.items() if k in keys}
    return pool


DEFAULT_CONTROLLER_KEYS: Sequence[str] = ("static", "hpa", "noai_mape", "ai_mape")


def describe_pool(pool: Dict[str, BaseController]) -> List[Dict[str, object]]:
    return [c.describe() for c in pool.values()]
