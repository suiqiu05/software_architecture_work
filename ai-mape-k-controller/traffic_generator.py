"""
traffic_generator.py —— 流量时序生成器（重构版）。

重构要点：
1. **预生成**：一次性生成完整的负载时序（而非在线随机产生），保证四组控制器
   面对**完全相同**的流量输入，实验具备可比性与可复现性；
2. **多形态**：内置 steady / diurnal / burst / random_walk / staircase / ramp /
   oscillation / spike 八种基元形态，并支持拼接为混合形态曲线；
3. **可回放**：`TrafficReplay` 提供确定性逐周期取值，并支持 CSV 落盘与重载，
   使 main_exp.py 能“先落盘、后回放”，跨进程共享同一份流量。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# 形态注册表
# --------------------------------------------------------------------------- #
TRAFFIC_KINDS: Sequence[str] = (
    "steady",       # 平稳
    "diurnal",      # 日周期（正弦 + 次谐波）
    "burst",        # 周期性突发脉冲
    "random_walk",  # 随机游走（慢漂移）
    "staircase",    # 阶梯上升/下降
    "ramp",         # 线性爬坡
    "oscillation",  # 高频振荡
    "spike",        # 孤立尖峰
)


@dataclass
class TrafficScenario:
    """一段可回放的负载时序。"""

    name: str
    values: np.ndarray                 # 单位：req/s，长度 = steps
    step_seconds: float = 5.0
    seed: int = 42
    phases: List[Dict[str, object]] = field(default_factory=list)  # 混合形态的分段说明

    # ---------------- 基本属性 ----------------
    @property
    def steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration_min(self) -> float:
        return self.steps * self.step_seconds / 60.0

    def at(self, t: int) -> float:
        """安全取值：越界时返回边界值（保证长跑不崩）。"""
        if t < 0:
            return float(self.values[0])
        if t >= self.steps:
            return float(self.values[-1])
        return float(self.values[t])

    def stats(self) -> Dict[str, float]:
        v = self.values
        return {
            "mean": float(np.mean(v)),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(np.max(v)),
            "min": float(np.min(v)),
            "std": float(np.std(v)),
        }

    # ---------------- 序列化 ----------------
    def save_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "arrival_rps"])
            for i, v in enumerate(self.values):
                w.writerow([i, f"{float(v):.4f}"])
        return path

    @staticmethod
    def load_csv(path: str, name: Optional[str] = None,
                 step_seconds: float = 5.0) -> "TrafficScenario":
        steps: List[float] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(float(row["arrival_rps"]))
        base = name or os.path.splitext(os.path.basename(path))[0]
        return TrafficScenario(name=base, values=np.asarray(steps, dtype=float),
                               step_seconds=step_seconds, seed=-1)


class TrafficReplay:
    """流量回放器：把预生成时序按控制周期逐步吐出，并保证可复现。"""

    def __init__(self, scenario: TrafficScenario):
        self.scenario = scenario
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def next(self) -> float:
        v = self.scenario.at(self._cursor)
        self._cursor += 1
        return v

    def __iter__(self):
        for v in self.scenario.values:
            yield float(v)

    def __len__(self) -> int:
        return self.scenario.steps


# --------------------------------------------------------------------------- #
# 基元形态生成器
# --------------------------------------------------------------------------- #
def _smooth_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """一阶平滑噪声，避免逐点白噪导致控制器频繁抖动。"""
    raw = rng.normal(0.0, sigma, size=n)
    k = np.exp(-np.arange(6) / 2.0)
    k /= k.sum()
    return np.convolve(raw, k, mode="same")


def gen_steady(n: int, rng: np.random.Generator, base: float = 180.0) -> np.ndarray:
    return np.clip(base + _smooth_noise(rng, n, base * 0.03), 1.0, None)


def gen_diurnal(n: int, rng: np.random.Generator, base: float = 200.0,
                amp: float = 150.0, period: float = 240.0) -> np.ndarray:
    t = np.arange(n)
    wave = (
        amp * np.sin(2 * np.pi * t / period)
        + 0.35 * amp * np.sin(2 * np.pi * t / (period / 3.0) + 1.1)  # 次谐波：午间小高峰
    )
    return np.clip(base + wave + _smooth_noise(rng, n, base * 0.04), 1.0, None)


def gen_burst(n: int, rng: np.random.Generator, base: float = 150.0,
              peak: float = 620.0) -> np.ndarray:
    """底噪背景上叠加随机到达的突发脉冲（脉冲宽度与间隔均随机）。"""
    v = np.full(n, base, dtype=float) + _smooth_noise(rng, n, base * 0.05)
    t = 0
    while t < n:
        t += int(rng.integers(28, 55))
        if t >= n:
            break
        width = int(rng.integers(4, 11))
        height = peak * float(rng.uniform(0.65, 1.15))
        shape = np.hanning(width + 2)[1:-1]  # 平滑的起落沿
        end = min(n, t + width)
        v[t:end] += height * shape[: end - t]
    return np.clip(v, 1.0, None)


def gen_random_walk(n: int, rng: np.random.Generator, base: float = 260.0,
                    step_sigma: float = 18.0) -> np.ndarray:
    inc = rng.normal(0.0, step_sigma, size=n)
    walk = base + np.cumsum(inc)
    walk = np.clip(walk, 60.0, 700.0)
    # 轻微回归基线，避免长期漂移到边界后失去区分度
    drift = (walk - base) * 0.004
    walk = walk - np.cumsum(drift)
    return np.clip(walk + _smooth_noise(rng, n, 6.0), 1.0, None)


def gen_staircase(n: int, rng: np.random.Generator, base: float = 120.0,
                  tread: int = 55) -> np.ndarray:
    levels = rng.uniform(0.55, 1.9, size=max(2, n // tread + 1))
    v = np.repeat(levels, tread)[:n] * base
    return np.clip(v + _smooth_noise(rng, n, 5.0), 1.0, None)


def gen_ramp(n: int, rng: np.random.Generator, base: float = 120.0,
             peak: float = 640.0) -> np.ndarray:
    t = np.arange(n)
    up = np.linspace(base, peak, n // 2 + 1)
    down = np.linspace(peak, base, n - len(up) + 1)[1:]
    v = np.concatenate([up, down])[:n]
    return np.clip(v + _smooth_noise(rng, n, 6.0), 1.0, None)


def gen_oscillation(n: int, rng: np.random.Generator, base: float = 300.0,
                    amp: float = 170.0, period: float = 40.0) -> np.ndarray:
    t = np.arange(n)
    v = base + amp * np.sin(2 * np.pi * t / period)
    return np.clip(v + _smooth_noise(rng, n, 8.0), 1.0, None)


def gen_spike(n: int, rng: np.random.Generator, base: float = 170.0,
              peak: float = 780.0) -> np.ndarray:
    v = np.full(n, base, dtype=float) + _smooth_noise(rng, n, base * 0.05)
    for _ in range(max(2, n // 180)):
        pos = int(rng.integers(0, n))
        width = int(rng.integers(2, 5))
        end = min(n, pos + width)
        shape = np.hanning(width + 2)[1:-1]
        v[pos:end] += (peak - base) * shape[: end - pos]
    return np.clip(v, 1.0, None)


_GENERATORS = {
    "steady": gen_steady,
    "diurnal": gen_diurnal,
    "burst": gen_burst,
    "random_walk": gen_random_walk,
    "staircase": gen_staircase,
    "ramp": gen_ramp,
    "oscillation": gen_oscillation,
    "spike": gen_spike,
}


def generate_traffic(kind: str, steps: int = 720, seed: int = 42,
                     step_seconds: float = 5.0) -> TrafficScenario:
    """生成单一形态的流量时序。"""
    if kind not in _GENERATORS:
        raise ValueError(f"未知流量形态 '{kind}'，可选：{list(_GENERATORS)}")
    rng = np.random.default_rng(seed)
    values = _GENERATORS[kind](steps, rng)
    return TrafficScenario(name=kind, values=np.asarray(values, dtype=float),
                           step_seconds=step_seconds, seed=seed)


# --------------------------------------------------------------------------- #
# 混合形态（主实验默认使用）
# --------------------------------------------------------------------------- #
DEFAULT_COMPOSITE_ORDER = ("diurnal", "burst", "random_walk",
                           "staircase", "oscillation", "spike")


def build_composite_profile(steps: int = 720, seed: int = 42, step_seconds: float = 5.0,
                            order: Sequence[str] = DEFAULT_COMPOSITE_ORDER,
                            blend: int = 12) -> TrafficScenario:
    """把多种形态拼接为一段连续曲线。

    为保证形态切换处不出现非物理跳变，在相邻段边界做 `blend` 个周期的线性过渡。
    随机种子在各段之间派生（seed + 段序号），保证整体可复现。
    """
    seg = max(1, steps // len(order))
    parts: List[np.ndarray] = []
    for i, kind in enumerate(order):
        n_seg = seg if i < len(order) - 1 else (steps - seg * (len(order) - 1))
        if n_seg <= 0:
            continue
        parts.append(_GENERATORS[kind](n_seg, np.random.default_rng(seed + i)))

    values = np.concatenate(parts).astype(float)

    # 边界平滑过渡
    boundaries = [seg * (i + 1) for i in range(len(order) - 1)]
    for b in boundaries:
        if b + blend >= len(values) or b - blend <= 0:
            continue
        left, right = values[b - 1], values[b + blend]
        ramp = np.linspace(0.0, 1.0, blend)
        values[b:b + blend] = (1 - ramp) * values[b:b + blend] + ramp * right
        values[b - blend:b] = values[b - blend:b] * 0.85 + (left + (right - left) * 0.15) * 0.15

    values = np.clip(values, 10.0, None)
    phases = [
        {"kind": kind, "start": i * seg,
         "end": min(steps, (i + 1) * seg)}
        for i, kind in enumerate(order)
    ]
    return TrafficScenario(name="composite", values=values,
                           step_seconds=step_seconds, seed=seed, phases=phases)


def make_scenario(profile: str = "composite", steps: int = 720, seed: int = 42,
                  step_seconds: float = 5.0,
                  csv_path: Optional[str] = None) -> TrafficScenario:
    """统一入口：支持内置形态、混合形态，或从 CSV 载入已落盘的流量。"""
    if csv_path:
        return TrafficScenario.load_csv(csv_path, name=profile, step_seconds=step_seconds)
    if profile == "composite":
        return build_composite_profile(steps=steps, seed=seed, step_seconds=step_seconds)
    return generate_traffic(profile, steps=steps, seed=seed, step_seconds=step_seconds)


if __name__ == "__main__":
    sc = make_scenario("composite", steps=720, seed=7)
    print(f"scenario={sc.name} steps={sc.steps} duration={sc.duration_min:.1f}min")
    for k, v in sc.stats().items():
        print(f"  {k:>5s}: {v:8.1f} rps")
