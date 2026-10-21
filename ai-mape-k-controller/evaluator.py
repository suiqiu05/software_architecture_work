"""
evaluator.py —— 实验评估器（升级版）：同时采集并对比四组控制器的质量指标。

采集维度（每个控制周期一条 `ControlStep`）：
- **服务质量**：p95 时延分布、SLO 违反率、错误率；
- **资源效率**：副本轨迹、累计 pod-step 成本、扩缩容抖动成本；
- **预测能力**（新增）：RMSE / MAE / MAPE / sMAPE / 偏差，仅在具备预测能力的
  控制器（NoAI-MAPE / AI-MAPE）上有定义。

产出：
- `summary()`：逐控制器的指标字典；
- `print_table()`：控制台对比表；
- `to_csv()` / `to_json()`：落盘，供 visual_compare.py 与后续分析消费。
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from analyze import forecast_report
import jsonutil
from knowledge import DEFAULT_CONFIG, ControlStep, SystemConfig, summarize_series


@dataclass
class ControllerRun:
    """单组控制器的一次完整运行记录。"""

    key: str
    label: str
    steps: List[ControlStep] = field(default_factory=list)
    forecast_pairs: List[Tuple[float, float]] = field(default_factory=list)
    forecast_errors: Dict[str, float] = field(default_factory=dict)
    forecast_errors_h1: Dict[str, float] = field(default_factory=dict)
    forecast_pairs_h1: List[Tuple[float, float]] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)
    scale_events: List[Dict[str, object]] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)


class Evaluator:
    """四组控制器的统一指标采集与汇总。"""

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg = config
        self.runs: Dict[str, ControllerRun] = {}

    # ---------------- 采集 ----------------
    def begin_run(self, key: str, label: str, meta: Optional[Dict[str, object]] = None) -> None:
        self.runs[key] = ControllerRun(key=key, label=label, meta=meta or {})

    def record(self, key: str, step: ControlStep) -> None:
        self.runs[key].steps.append(step)

    def attach_forecast(self, key: str,
                        pairs: Sequence[Tuple[float, float]],
                        pairs_h1: Optional[Sequence[Tuple[float, float]]] = None,
                        ) -> None:
        """登记预测-真值样本对，并即时计算预测误差指标。

        pairs     : 决策视界下的样本（反映控制器“实际用上的预测”质量）
        pairs_h1  : 统一 1 步视界下的样本（用于跨控制器公平比较预测能力）
        """
        run = self.runs[key]
        run.forecast_pairs = list(pairs)
        run.forecast_errors = self._errors_of(pairs)
        run.forecast_pairs_h1 = list(pairs_h1 or [])
        run.forecast_errors_h1 = self._errors_of(pairs_h1 or [])

    @staticmethod
    def _errors_of(pairs: Sequence[Tuple[float, float]]) -> Dict[str, float]:
        if not pairs:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"),
                    "smape": float("nan"), "bias": float("nan"), "n": 0.0}
        actual = [p[0] for p in pairs]
        pred = [p[1] for p in pairs]
        return forecast_report(actual, pred)

    def attach_scale_events(self, key: str, events: Sequence[Dict[str, object]]) -> None:
        self.runs[key].scale_events = list(events)

    # ---------------- 汇总 ----------------
    def _summarize(self, run: ControllerRun) -> Dict[str, float]:
        steps = run.steps
        if not steps:
            return {}

        lat = [s.p95_latency_ms for s in steps]
        reps = [float(s.replicas) for s in steps]
        ready = [float(s.ready_replicas) for s in steps]
        arrival = [s.arrival_rps for s in steps]

        n = len(steps)
        slo = self.cfg.slo_latency_ms
        violations = sum(1 for x in lat if x > slo)
        viol_rate = violations / n

        pod_cost = sum(s.pod_cost for s in steps)
        scale_cost = sum(s.scale_cost for s in steps)
        total_cost = pod_cost + scale_cost

        total_requests = sum(arrival) * self.cfg.step_seconds      # 请求总数
        cost_per_1k = total_cost / max(total_requests / 1000.0, 1e-6)

        # 扩缩容抖动
        changes = 0
        ups = downs = 0
        prev = steps[0].replicas
        for s in steps[1:]:
            if s.replicas != prev:
                changes += 1
                ups += int(s.replicas > prev)
                downs += int(s.replicas < prev)
                prev = s.replicas
        minutes = max(n * self.cfg.step_seconds / 60.0, 1e-6)

        lat_stat = summarize_series(lat)

        metrics = {
            "steps": float(n),
            "slo_violation_rate_pct": viol_rate * 100.0,
            "slo_violations": float(violations),
            "mean_p95_ms": lat_stat["mean"],
            "p95_of_p95_ms": lat_stat["p95"],
            "p99_p95_ms": lat_stat["p99"],
            "max_p95_ms": lat_stat["max"],
            "mean_error_rate": sum(s.error_rate for s in steps) / n,
            "mean_replicas": sum(reps) / n,
            "max_replicas": max(reps),
            "mean_ready_replicas": sum(ready) / n,
            "pod_cost": pod_cost,
            "scale_cost": scale_cost,
            "total_cost": total_cost,
            "cost_per_1k_requests": cost_per_1k,
            "scale_changes": float(changes),
            "scale_ups": float(ups),
            "scale_downs": float(downs),
            "flapping_per_min": changes / minutes,
            "mean_arrival_rps": sum(arrival) / n,
        }

        # 预测能力（无预测能力的控制器为 NaN）
        fe = run.forecast_errors or {}
        fe1 = run.forecast_errors_h1 or {}
        metrics.update({
            "forecast_horizon": float(run.meta.get("horizon", 0) or 0),
            "forecast_rmse": float(fe.get("rmse", float("nan"))),
            "forecast_mape_pct": float(fe.get("mape", float("nan"))),
            "forecast_bias": float(fe.get("bias", float("nan"))),
            "forecast_samples": float(fe.get("n", 0.0)),
            # 统一 1 步视界：跨控制器可比的预测精度
            "rmse_h1": float(fe1.get("rmse", float("nan"))),
            "mape_h1_pct": float(fe1.get("mape", float("nan"))),
            "mae_h1": float(fe1.get("mae", float("nan"))),
        })
        return metrics

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {k: self._summarize(r) for k, r in self.runs.items()}
        # 跨组归一化：以最低成本组为基准计算相对成本与综合分
        valid_costs = [v["cost_per_1k_requests"] for v in out.values()
                       if not math.isnan(v.get("cost_per_1k_requests", float("nan")))]
        base_cost = min(valid_costs) if valid_costs else 1.0
        for v in out.values():
            v["cost_index"] = v["cost_per_1k_requests"] / max(base_cost, 1e-6)
            # 综合分（越低越好）：违反率为主，成本为次
            v["composite_score"] = v["slo_violation_rate_pct"] + 0.5 * v["cost_per_1k_requests"]
        return out

    # ---------------- 输出 ----------------
    def rows(self) -> List[Dict[str, object]]:
        s = self.summary()
        rows = []
        for key, run in self.runs.items():
            row: Dict[str, object] = {"controller": key, "label": run.label}
            row.update({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in s[key].items()})
            rows.append(row)
        return rows

    def print_table(self) -> None:
        rows = self.rows()
        if not rows:
            print("(无评估结果)")
            return
        cols = ["controller", "slo_violation_rate_pct", "mean_p95_ms", "p95_of_p95_ms",
                "mean_replicas", "total_cost", "cost_per_1k_requests",
                "scale_changes", "flapping_per_min",
                "forecast_horizon", "forecast_rmse", "forecast_mape_pct",
                "rmse_h1", "mape_h1_pct", "composite_score"]
        widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols}
        line = "  ".join(c.ljust(widths[c]) for c in cols)
        print("\n" + "=" * len(line))
        print(line)
        print("-" * len(line))
        for r in rows:
            print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
        print("=" * len(line) + "\n")

    def to_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        rows = self.rows()
        if not rows:
            return path
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "config": self.cfg.as_dict(),
            "summary": self.summary(),
            "runs": {
                k: {
                    "label": r.label,
                    "meta": r.meta,
                    "trace": [s.to_dict() for s in r.steps],
                    "forecast_pairs": r.forecast_pairs,
                    "forecast_errors": r.forecast_errors,
                    "forecast_errors_h1": r.forecast_errors_h1,
                    "scale_events": r.scale_events,
                }
                for k, r in self.runs.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            # 用 jsonutil.dump：把 NaN/Inf 转成 null，保证输出是**合法** JSON
            # （否则 JS/严格解析器会整段失败）
            jsonutil.dump(payload, f, indent=2, default=str)
        return path

    def to_traces_csv(self, outdir: str) -> List[str]:
        """逐控制器导出逐周期轨迹，供绘图与细粒度分析。"""
        os.makedirs(outdir, exist_ok=True)
        paths = []
        for key, run in self.runs.items():
            p = os.path.join(outdir, f"trace_{key}.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(list(ControlStep(0, 0.0, 0.0).to_dict().keys()))
                for s in run.steps:
                    w.writerow(list(s.to_dict().values()))
            paths.append(p)
        return paths
