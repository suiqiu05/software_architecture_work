"""
visual_compare.py —— 四组控制器的对比可视化（升级版）。

产出两张图：
1. `comparison.png`：四组曲线叠加对比
   - (a) 回放流量（四组共享同一输入，验证实验可比性）
   - (b) p95 时延（叠加 SLO 阈值线，直观观察违反区间）
   - (c) 副本数轨迹（观察响应速度与抖动）
   - (d) 累计成本（资源开销增长曲线）
   - (e) **预测误差子图**：真实负载 vs NoAI/AI 预测曲线 + 残差柱状图
2. `forecast_error.png`：预测误差专项对比（RMSE / MAPE 分组柱状图 + 残差时序）

使用 `matplotlib` 的 Agg 后端，可在无显示环境（服务器 / CI）直接出图。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from evaluator import Evaluator
from knowledge import ControlStep, SystemConfig

# 默认输出目录由 __file__ 推导，保证换机器 / 换工作目录后仍落在项目内
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "results")

# 四组固定配色（浅色主题下对比清晰）
COLORS: Dict[str, str] = {
    "static": "#9E9E9E",
    "hpa": "#1F77B4",
    "noai_mape": "#FF7F0E",
    "ai_mape": "#2CA02C",
}
LABELS: Dict[str, str] = {
    "static": "Static",
    "hpa": "K8s HPA",
    "noai_mape": "NoAI-MAPE",
    "ai_mape": "AI-MAPE",
}
ORDER: Sequence[str] = ("static", "hpa", "noai_mape", "ai_mape")


def _style_axes(ax, title: str, ylabel: str, xlabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.7)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _trace(run, field_name: str) -> np.ndarray:
    return np.asarray([getattr(s, field_name) for s in run.steps], dtype=float)


# --------------------------------------------------------------------------- #
# 主对比图
# --------------------------------------------------------------------------- #
def plot_comparison(evaluator: Evaluator, scenario_values: Optional[Sequence[float]] = None,
                    outdir: str = DEFAULT_OUTDIR, filename: str = "comparison.png",
                    config: Optional[SystemConfig] = None,
                    title: str = "Four-Controller Comparison under Identical Replayed Traffic",
                    scenario: Optional[object] = None,
                    ) -> str:
    os.makedirs(outdir, exist_ok=True)
    runs = evaluator.runs
    keys = [k for k in ORDER if k in runs] or list(runs.keys())
    cfg = config or evaluator.cfg
    if scenario is not None and scenario_values is None:
        scenario_values = getattr(scenario, "values", None)

    fig, axes = plt.subplots(5, 1, figsize=(15, 17), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.25, 1.1, 1.0, 1.35]})
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)

    n_steps = max((r.n_steps for r in runs.values()), default=0)
    t = np.arange(n_steps)
    t_min = t * (cfg.step_seconds / 60.0) if cfg else t

    # ---------- (a) 流量 ----------
    ax = axes[0]
    if scenario_values is not None:
        v = np.asarray(scenario_values, dtype=float)
        ax.plot(t_min[:len(v)], v, color="#37474F", linewidth=1.2)
        ax.fill_between(t_min[:len(v)], 0, v, color="#37474F", alpha=0.12)

        # 混合形态分段标注：说明该段施加的是哪一类负载特征
        phases = getattr(scenario, "phases", None) or []
        if phases:
            y_max = float(np.max(v))
            ax.set_ylim(0, y_max * 1.28)
            for ph in phases:
                s, e = int(ph["start"]), int(ph["end"])
                if e <= s or s >= len(v):
                    continue
                ax.axvspan(s * cfg.step_seconds / 60.0, e * cfg.step_seconds / 60.0,
                           color="#90A4AE", alpha=0.07, linewidth=0)
                ax.text((s + e) / 2.0 * cfg.step_seconds / 60.0, y_max * 1.16,
                        str(ph["kind"]), ha="center", va="top", fontsize=7.5,
                        color="#455A64", style="italic")
    _style_axes(ax, "(a) Replayed Arrival Rate (identical input for all controllers)",
                "requests / s")
    ax.legend(["arrival rate"], fontsize=8, loc="upper right")

    # ---------- (b) p95 时延 ----------
    ax = axes[1]
    for k in keys:
        lat = _trace(runs[k], "p95_latency_ms")
        ax.plot(t_min[:len(lat)], lat, color=COLORS.get(k), linewidth=1.1,
                label=LABELS.get(k, k), alpha=0.95)
    ax.axhline(cfg.slo_latency_ms, color="#B00020", linestyle="--", linewidth=1.2,
               label=f"SLO p95 = {cfg.slo_latency_ms:.0f} ms")
    ax.set_yscale("log")
    _style_axes(ax, "(b) p95 Latency (log scale) vs SLO", "p95 latency (ms)")
    ax.legend(fontsize=8, ncol=5, loc="upper left")

    # ---------- (c) 副本数 ----------
    ax = axes[2]
    for k in keys:
        r = _trace(runs[k], "replicas")
        ax.step(t_min[:len(r)], r, color=COLORS.get(k), linewidth=1.2,
                where="post", label=LABELS.get(k, k), alpha=0.95)
    _style_axes(ax, "(c) Replica Count Trajectory", "replicas")
    ax.legend(fontsize=8, ncol=4, loc="upper left")

    # ---------- (d) 累计成本 ----------
    ax = axes[3]
    for k in keys:
        cost = np.cumsum(_trace(runs[k], "pod_cost") + _trace(runs[k], "scale_cost"))
        ax.plot(t_min[:len(cost)], cost, color=COLORS.get(k), linewidth=1.3,
                label=LABELS.get(k, k), alpha=0.95)
    _style_axes(ax, "(d) Cumulative Cost (pod-steps + scaling penalty)",
                "cumulative cost")
    ax.legend(fontsize=8, ncol=4, loc="upper left")

    # ---------- (e) 预测误差子图 ----------
    ax = axes[4]
    if scenario_values is not None:
        v = np.asarray(scenario_values, dtype=float)
        ax.plot(t_min[:len(v)], v, color="#37474F", linewidth=1.4, label="actual arrival", zorder=3)
    for k in keys:
        fc = [s.forecast_rps for s in runs[k].steps]
        if not any(np.isfinite(np.asarray(fc, dtype=float))):
            continue
        ax.plot(t_min[:len(fc)], fc, color=COLORS.get(k), linewidth=1.0,
                linestyle="--", alpha=0.95, label=f"{LABELS.get(k, k)} forecast", zorder=2)
    _style_axes(ax, "(e) Forecast vs Actual (dashed = prediction)",
                "requests / s", "time (minutes)")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    if scenario_values is not None:
        # 截断 y 轴：NoAI 的趋势外推在突发上升沿会给出极端高估，保留可读性
        ax.set_ylim(0, float(np.max(np.asarray(scenario_values, dtype=float))) * 1.30)

    # 残差柱状图（右下角 inset）
    try:
        ins = ax.inset_axes([0.62, 0.12, 0.36, 0.34])
        for k in keys:
            pairs = runs[k].forecast_pairs
            if not pairs:
                continue
            res = np.asarray([a - p for a, p in pairs], dtype=float)
            ins.bar(np.arange(len(res)), res, width=1.0, alpha=0.65,
                    color=COLORS.get(k), label=LABELS.get(k, k))
        ins.axhline(0.0, color="#333333", linewidth=0.8)
        ins.set_title("residual (actual - predicted)", fontsize=8)
        ins.tick_params(labelsize=6)
        ins.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ins.legend(fontsize=6, ncol=2)
    except Exception:  # inset 失败不影响主图
        pass

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 预测误差专项图
# --------------------------------------------------------------------------- #
def plot_forecast_errors(evaluator: Evaluator, outdir: str = DEFAULT_OUTDIR,
                         filename: str = "forecast_error.png") -> str:
    os.makedirs(outdir, exist_ok=True)
    runs = evaluator.runs
    keys = [k for k in ORDER if k in runs] or list(runs.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("Forecast Accuracy: NoAI-MAPE vs AI-MAPE",
                 fontsize=13, fontweight="bold", y=1.0)

    # ---- RMSE / MAPE 柱状图（统一 h=1 视界，跨控制器公平） ----
    ax = axes[0]
    metric_keys = ["rmse", "mape"]
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(keys), 1)
    for i, k in enumerate(keys):
        errs = runs[k].forecast_errors_h1 or runs[k].forecast_errors or {}
        vals = []
        for m in metric_keys:
            v = errs.get(m, float("nan"))
            vals.append(0.0 if v != v else v)  # NaN -> 0
        ax.bar(x + i * width, vals, width=width, color=COLORS.get(k),
               label=LABELS.get(k, k), alpha=0.9)
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels(["RMSE (rps)", "MAPE (%)"], fontsize=9)
    ax.set_title("(a) Error metrics @ h=1, same horizon (lower is better)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.28, linestyle="--", linewidth=0.7)
    ax.legend(fontsize=8)
    ax.set_xlabel("Static / HPA have no forecasting capability "
                  "(plotted as 0)", fontsize=7, color="#666666", style="italic")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ---- 残差分布 ----
    ax = axes[1]
    data, labels, colors = [], [], []
    for k in keys:
        pairs = runs[k].forecast_pairs_h1 or runs[k].forecast_pairs
        if not pairs:
            continue
        data.append([a - p for a, p in pairs])
        labels.append(LABELS.get(k, k))
        colors.append(COLORS.get(k))
    if data:
        # matplotlib >= 3.9 使用 tick_labels；旧版本使用 labels
        try:
            bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.5,
                            showfliers=False, medianprops={"color": "#222222"})
        except TypeError:
            bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5,
                            showfliers=False, medianprops={"color": "#222222"})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_title("(b) Residual distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("actual - predicted (rps)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.28, linestyle="--", linewidth=0.7)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ---- 残差绝对值时序（滚动均值） ----
    ax = axes[2]
    for k in keys:
        pairs = runs[k].forecast_pairs_h1 or runs[k].forecast_pairs
        if not pairs:
            continue
        res = np.abs(np.asarray([a - p for a, p in pairs], dtype=float))
        win = max(5, len(res) // 40)
        kernel = np.ones(win) / win
        smooth = np.convolve(res, kernel, mode="valid")
        ax.plot(np.arange(len(smooth)), smooth, color=COLORS.get(k),
                linewidth=1.2, label=LABELS.get(k, k), alpha=0.95)
    ax.set_title("(c) |residual| rolling mean over time", fontsize=11, fontweight="bold")
    ax.set_ylabel("|actual - predicted| (rps)", fontsize=9)
    ax.set_xlabel("forecast sample index", fontsize=9)
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.7)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(evaluator: Evaluator, scenario_values: Optional[Sequence[float]] = None,
             outdir: str = DEFAULT_OUTDIR,
             config: Optional[SystemConfig] = None,
             scenario: Optional[object] = None) -> Dict[str, str]:
    """一次性产出全部对比图。"""
    return {
        "comparison": plot_comparison(evaluator, scenario_values, outdir=outdir,
                                      config=config, scenario=scenario),
        "forecast_error": plot_forecast_errors(evaluator, outdir=outdir),
    }
