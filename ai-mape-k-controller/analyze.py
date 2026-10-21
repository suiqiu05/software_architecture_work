"""
analyze.py —— MAPE-K 的 **Analyze（A）** 环节：负载预测 + 预测误差度量。

提供两类分析器：
1. `HeuristicAnalyzer`（NoAI-MAPE 使用）：固定参数的指数平滑 + 线性趋势外推，
   不含任何在线学习，代表“传统自适应中间件”的分析能力；
2. `EnsembleAnalyzer`（AI-MAPE 使用）：由 4 个基预测器构成的在线集成（Hedge /
   指数加权多数算法），权重按各预测器的滚动损失指数更新，并输出预测残差分布
   以构造分位数上界，代表“具备在线学习能力的 AI 分析器”。

同时提供 `compute_rmse` / `compute_mape` / `forecast_report` 等误差度量工具，
供 Evaluator 与可视化模块直接调用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from knowledge import KnowledgeBase, SystemConfig, DEFAULT_CONFIG

EPS = 1e-6


# --------------------------------------------------------------------------- #
# 误差度量（新增 RMSE / MAPE）
# --------------------------------------------------------------------------- #
def compute_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """均方根误差（对大偏差敏感，用于衡量突发预测能力）。"""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def compute_mape(y_true: Sequence[float], y_pred: Sequence[float],
                 floor: float = 1.0) -> float:
    """平均绝对百分比误差（%）。

    为避免低负载区间的除零放大，对真实值施加下限 `floor`（默认 1 rps）。
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(a), floor)
    return float(np.mean(np.abs(a - b) / denom) * 100.0)


def compute_smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """对称 MAPE（%），对真实值接近 0 的情形更稳健。"""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0:
        return float("nan")
    denom = (np.abs(a) + np.abs(b)) / 2.0
    denom = np.maximum(denom, EPS)
    return float(np.mean(np.abs(a - b) / denom) * 100.0)


def compute_bias(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """平均偏差（正=低估，负=高估），用于判断预测是否系统性偏乐观。"""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(a - b)) if a.size else float("nan")


def forecast_report(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """一次给出完整预测误差画像。"""
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "smape": compute_smape(y_true, y_pred),
        "bias": compute_bias(y_true, y_pred),
        "n": float(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# 预测结果
# --------------------------------------------------------------------------- #
@dataclass
class ForecastResult:
    horizon: int
    point: float                       # 点预测（h 步之后）
    upper: float                       # 分位数上界（点预测 + z·σ）
    sigma: float = 0.0                 # 残差标准差
    method: str = ""
    weights: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 基预测器
# --------------------------------------------------------------------------- #
class _LastValue:
    name = "last"

    def __init__(self):
        self._x = 0.0

    def update(self, x: float) -> None:
        self._x = x

    def forecast(self, h: int) -> float:
        return self._x


class _EWMA:
    name = "ewma"

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.level: Optional[float] = None

    def update(self, x: float) -> None:
        self.level = x if self.level is None else self.alpha * x + (1 - self.alpha) * self.level

    def forecast(self, h: int) -> float:
        return self.level if self.level is not None else 0.0


class _HoltLinear:
    """Holt 双指数平滑：水平 + 趋势。"""
    name = "holt"

    def __init__(self, alpha: float = 0.45, beta: float = 0.15):
        self.alpha, self.beta = alpha, beta
        self.level: Optional[float] = None
        self.trend: float = 0.0

    def update(self, x: float) -> None:
        if self.level is None:
            self.level, self.trend = x, 0.0
            return
        prev = self.level
        self.level = self.alpha * x + (1 - self.alpha) * (prev + self.trend)
        self.trend = self.beta * (self.level - prev) + (1 - self.beta) * self.trend

    def forecast(self, h: int) -> float:
        if self.level is None:
            return 0.0
        # 趋势项做阻尼，避免长周期外推发散
        damp = sum(0.85 ** i for i in range(1, h + 1))
        return self.level + self.trend * damp


class _FourierRLS:
    """傅里叶季节特征 + 递推最小二乘（带遗忘因子）的在线线性回归。

    特征向量：常数项、当前水平、短时动量、K 组正/余弦季节基。
    由于特征不依赖未来滞后值，可直接外推 h 步。
    """
    name = "fourier_rls"

    def __init__(self, period: float = 240.0, n_harmonics: int = 3,
                 forget: float = 0.98, ridge: float = 10.0,
                 n_features: Optional[int] = None):
        self.period = period
        self.n_harmonics = n_harmonics
        self.forget = forget
        self.t = 0
        # 特征维数：常数 1 + 水平 1 + 动量 1 + 2·谐波数
        n_feat = n_features or (3 + 2 * n_harmonics)
        self.P = np.eye(n_feat) * ridge
        self.w = np.zeros(n_feat)
        self._x = 0.0
        self._prev = 0.0

    # 目标为 x_{t+1}，因此在 update 时用“上一时刻的特征”解释当前观测
    def _feat(self, t: int, x: float, prev: float) -> np.ndarray:
        f = [1.0, x, x - prev]
        for k in range(1, self.n_harmonics + 1):
            f.append(math.sin(2 * math.pi * k * t / self.period))
            f.append(math.cos(2 * math.pi * k * t / self.period))
        return np.asarray(f, dtype=float)

    def update(self, x: float) -> None:
        if self.t > 0:
            phi = self._feat(self.t - 1, self._prev_x, self._prev_prev)
            # RLS 更新
            denom = self.forget + float(phi @ self.P @ phi)
            k_gain = (self.P @ phi) / denom
            err = x - float(self.w @ phi)
            self.w = self.w + k_gain * err
            self.P = (self.P - np.outer(k_gain, phi @ self.P)) / self.forget
        self._prev_prev, self._prev_x, self._x = self._prev_x, self._x, x
        self.t += 1

    def forecast(self, h: int) -> float:
        if self.t == 0:
            return 0.0
        # 多步递推：水平项用当前观测滚动，季节项按时间推进
        cur, prev = self._x, self._prev_x
        val = cur
        for step in range(1, h + 1):
            t_future = self.t - 1 + step
            phi = self._feat(t_future, cur, prev)
            val = float(self.w @ phi)
            prev, cur = cur, val
        return float(np.clip(val, 0.0, None))

    # 兼容属性名
    @property
    def _prev_x(self) -> float:
        return getattr(self, "_px", 0.0)

    @_prev_x.setter
    def _prev_x(self, v: float) -> None:
        self._px = v

    @property
    def _prev_prev(self) -> float:
        return getattr(self, "_pp", 0.0)

    @_prev_prev.setter
    def _prev_prev(self, v: float) -> None:
        self._pp = v


# --------------------------------------------------------------------------- #
# 分析器
# --------------------------------------------------------------------------- #
class BaseAnalyzer:
    """分析器接口。"""

    name = "base"
    supports_forecast = False

    def predict(self, knowledge: KnowledgeBase, horizon: int,
                config: SystemConfig = DEFAULT_CONFIG) -> ForecastResult:
        raise NotImplementedError

    def observe(self, realized: float, predicted: float,
                knowledge: KnowledgeBase) -> None:
        """把已实现的真实值回灌给分析器（在线学习钩子）。"""
        raise NotImplementedError

    def reset(self) -> None:
        pass


class NoForecastAnalyzer(BaseAnalyzer):
    """无预测能力的占位分析器（Static / HPA 使用）。

    仅维护窗口统计量，输出 NaN 预测值，便于在对比图中标注“该组无预测”。
    """

    name = "none"
    supports_forecast = False

    def predict(self, knowledge: KnowledgeBase, horizon: int,
                config: SystemConfig = DEFAULT_CONFIG) -> ForecastResult:
        recent = knowledge.series("observed_rps", n=5) or [0.0]
        cur = recent[-1]
        return ForecastResult(horizon=horizon, point=float("nan"),
                              upper=float("nan"), sigma=0.0, method="none",
                              detail={"current": float(cur)})

    def observe(self, realized: float, predicted: float,
                knowledge: KnowledgeBase) -> None:
        return None


class HeuristicAnalyzer(BaseAnalyzer):
    """NoAI-MAPE 的分析器：EWMA 水平 + 固定窗口线性趋势，参数恒定不学习。"""

    name = "heuristic"
    supports_forecast = True

    def __init__(self, alpha: float = 0.25, trend_window: int = 5):
        self.alpha = alpha
        self.trend_window = trend_window
        self.level: Optional[float] = None

    def predict(self, knowledge: KnowledgeBase, horizon: int,
                config: SystemConfig = DEFAULT_CONFIG) -> ForecastResult:
        series = knowledge.series("observed_rps")
        if not series:
            return ForecastResult(horizon, 0.0, 0.0, 0.0, "heuristic")
        x = series[-1]
        self.level = x if self.level is None else self.alpha * x + (1 - self.alpha) * self.level

        if len(series) >= self.trend_window:
            slope = (series[-1] - series[-self.trend_window]) / (self.trend_window - 1)
        else:
            slope = 0.0
        point = self.level + slope * horizon
        # 启发式安全上界：水平 + 20%
        return ForecastResult(
            horizon=horizon,
            point=float(max(0.0, point)),
            upper=float(max(0.0, max(point, self.level * 1.2))),
            sigma=0.0,
            method="heuristic",
            detail={"level": float(self.level), "slope": float(slope)},
        )

    def observe(self, realized: float, predicted: float,
                knowledge: KnowledgeBase) -> None:
        # 无学习：仅记录残差用于评估
        knowledge.push_residual(realized - predicted)

    def reset(self) -> None:
        self.level = None


class EnsembleAnalyzer(BaseAnalyzer):
    """AI-MAPE 的分析器：多基预测器在线集成（指数加权多数 / Hedge）。

    权重更新：w_i ∝ exp(−η · L_i)，L_i 为归一化滚动累计损失。
    这使集成能在“平稳期信任 EWMA / Holt”、“突发期信任 Last / 季节回归”之间切换。
    """

    name = "ensemble_ai"
    supports_forecast = True

    def __init__(self, eta: float = 2.5, horizon_default: int = 3,
                 z: float = 1.5, loss_decay: float = 0.92, max_uplift: float = 0.5):
        self.eta = eta
        self.horizon_default = horizon_default
        self.z = z               # 上分位数系数（1.5 ≈ 93% 分位，抑制突发漏判）
        self.max_uplift = max_uplift  # 上界相对点预测的最大放大比例（防止过度冗余）
        self.loss_decay = loss_decay
        self.models = [_LastValue(), _EWMA(0.35), _HoltLinear(), _FourierRLS()]
        self._n = len(self.models)
        self._cum_loss = np.zeros(self._n, dtype=float)
        self._weights = np.ones(self._n, dtype=float) / self._n
        self._history: List[float] = []
        self._last_pred: Optional[float] = None
        self._warmup = 3

    # ---------------- 预测 ----------------
    def predict(self, knowledge: KnowledgeBase, horizon: int,
                config: SystemConfig = DEFAULT_CONFIG) -> ForecastResult:
        preds = np.array([m.forecast(horizon) for m in self.models], dtype=float)
        if len(self._history) < self._warmup:
            # 冷启动：退化为“最近观测”，避免未收敛模型给出离谱值
            series = knowledge.series("observed_rps", n=1)
            point = float(series[-1]) if series else 0.0
            preds = np.full(self._n, point)
        else:
            point = float(self._weights @ preds)

        # 残差分布 -> 分位数上界
        sigma = float(np.sqrt(float(self._weights @ (preds - point) ** 2)))  # 分歧度
        hist_sigma = knowledge.residual_std(window=30, default=0.0)
        sigma = float(max(sigma, hist_sigma, 0.0))
        # 上界既要覆盖突发，也要防止残差 std 过大时无限放大导致过度预留
        upper = min(point + self.z * sigma, point * (1.0 + self.max_uplift))

        return ForecastResult(
            horizon=horizon,
            point=float(max(0.0, point)),
            upper=float(max(0.0, upper)),
            sigma=sigma,
            method="ensemble_ai",
            weights={m.name: float(w) for m, w in zip(self.models, self._weights)},
            detail={m.name: float(p) for m, p in zip(self.models, preds)},
        )

    # ---------------- 在线学习 ----------------
    def observe(self, realized: float, predicted: float,
                knowledge: KnowledgeBase) -> None:
        # 关键：先取各基模型“更新前”的一步预测（即对当前时刻的预测），
        # 与刚到达的真实值比对后再更新模型，保证损失评估无信息泄露。
        prior_preds = np.array([m.forecast(1) for m in self.models], dtype=float)

        for m in self.models:
            m.update(float(realized))
        self._history.append(float(realized))

        if len(self._history) <= self._warmup:
            return

        scale = max(abs(realized), 10.0)
        losses = np.abs(prior_preds - realized) / scale
        self._cum_loss = self.loss_decay * self._cum_loss + losses
        logits = -self.eta * (self._cum_loss - self._cum_loss.min())
        exp_w = np.exp(np.clip(logits, -50, 50))
        self._weights = exp_w / exp_w.sum()

        knowledge.push_residual(realized - predicted)
        knowledge.put("ensemble_weights",
                      {m.name: float(w) for m, w in zip(self.models, self._weights)})

    def reset(self) -> None:
        self.models = [_LastValue(), _EWMA(0.35), _HoltLinear(), _FourierRLS()]
        self._cum_loss = np.zeros(self._n, dtype=float)
        self._weights = np.ones(self._n, dtype=float) / self._n
        self._history = []
        self._last_pred = None


def build_analyzer(kind: str, **kwargs) -> BaseAnalyzer:
    """分析器工厂。kind ∈ {none, heuristic, ensemble}"""
    table = {
        "none": NoForecastAnalyzer,
        "heuristic": HeuristicAnalyzer,
        "ensemble": EnsembleAnalyzer,
    }
    if kind not in table:
        raise ValueError(f"未知分析器类型 '{kind}'，可选：{list(table)}")
    return table[kind](**kwargs)
