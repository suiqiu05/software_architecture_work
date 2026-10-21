"""
main_exp.py —— 实验总调度。

实验设计（控制变量法）：
1. **固定流量**：先用 `traffic_generator` 预生成一段多形态负载时序并落盘，
   四组控制器回放**完全相同**的流量与随机噪声，排除输入差异；
2. **并行执行**：四组控制器各自拥有独立的 Plant / Monitor / Knowledge 实例，
   通过进程池并行仿真，互不干扰；
3. **统一评估**：`Evaluator` 同时采集四组的服务质量、资源成本与预测误差；
4. **自动出图**：`visual_compare` 产出叠加对比图与预测误差专项图。

用法示例：
    python main_exp.py                                  # 默认：复合形态 720 步（60 分钟）
    python main_exp.py --profile burst --steps 600      # 单一形态
    python main_exp.py --seed 7 --outdir results/run7   # 换随机种子
    python main_exp.py --sequential                     # 关闭并行（便于调试）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# 启动预检（可移植性基石）：
#   - 注入 sys.path，保证任意工作目录 / 进程池子进程都能 import 本地模块；
#   - 检查 numpy / matplotlib，缺失时给出修复指引而非裸 traceback。
# 必须在 import 任何本地模块之前执行。
# --------------------------------------------------------------------------- #
from bootstrap import HERE, ensure_ready  # noqa: E402

ensure_ready(entry="main_exp.py")

DEFAULT_OUTDIR = os.path.join(HERE, "results")

import numpy as np  # noqa: E402

from analyze import forecast_report
from controller_pool import (DEFAULT_CONTROLLER_KEYS, build_controller_pool,
                             describe_pool)
from evaluator import Evaluator
from execute import SimulatedExecutor
from knowledge import ControlStep, SystemConfig
from llm_advisor import (ENV_KEY, PROVIDER_PRESETS, LLMAdvisor, mask_key)
from monitor import ApplicationPlant, Monitor
from traffic_generator import TRAFFIC_KINDS, make_scenario
from visual_compare import plot_all


# --------------------------------------------------------------------------- #
# 单组实验（供进程池调用，必须可 pickle 的模块级函数）
# --------------------------------------------------------------------------- #
def run_single_controller(key: str, traffic: Sequence[float], cfg_dict: Dict,
                          seed: int = 42, static_replicas: Optional[int] = None,
                          hints: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, object]:
    """在给定流量上完整回放一个控制器，返回轨迹与统计样本。

    hints: LLM 顾问（或人工）给出的初始参数建议，仅对具备学习能力的控制器生效。
    """
    cfg = SystemConfig(**cfg_dict)
    controller = build_controller_pool(cfg, static_replicas=static_replicas)[key]
    applied = {}
    if hints and hasattr(controller, "apply_hints"):
        applied = controller.apply_hints(hints)

    plant = ApplicationPlant(cfg, seed=seed)
    monitor = Monitor(plant, knowledge=controller.knowledge, config=cfg, seed=seed + 1)
    executor = SimulatedExecutor(plant, cfg)

    steps: List[Dict[str, object]] = []
    values = np.asarray(traffic, dtype=float)

    for t in range(values.shape[0]):
        arrival = float(values[t])

        # --- M: 采样 ---
        obs = monitor.sample(t, arrival)
        # --- A: 学习 + 预测误差对齐 ---
        controller.observe(obs)
        # --- A + P: 分析并规划 ---
        decision = controller.decide(obs)
        # --- E: 执行 ---
        res = executor.apply(decision, t, obs.replicas)

        steps.append(ControlStep(
            t=t,
            arrival_rps=arrival,
            observed_rps=obs.observed_rps,
            forecast_rps=decision.forecast_rps,
            replicas=obs.replicas,
            ready_replicas=obs.ready_replicas,
            cpu_util=obs.cpu_util,
            p95_latency_ms=obs.p95_latency_ms,
            error_rate=obs.error_rate,
            decision_reason=res.reason,
            pod_cost=res.pod_cost,
            scale_cost=res.scale_cost,
        ).to_dict())

    meta = controller.describe()
    if applied:
        meta["llm_hints_applied"] = applied

    return {
        "key": key,
        "label": controller.label,
        "meta": meta,
        "steps": steps,
        "forecast_pairs": controller.forecast_pairs,
        "forecast_errors": controller.forecast_errors,
        "forecast_pairs_h1": controller.forecast_pairs_h1,
        "forecast_errors_h1": controller.forecast_errors_h1,
        "scale_events": plant.scale_events,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-MAPE-K 自适应控制器对比实验（Static / HPA / NoAI-MAPE / AI-MAPE）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--profile", default="composite",
                   help=f"流量形态：composite 或 {list(TRAFFIC_KINDS)}")
    p.add_argument("--traffic-csv", default=None, help="从 CSV 载入已生成流量（优先于 --profile）")
    p.add_argument("--steps", type=int, default=720, help="仿真周期数")
    p.add_argument("--step-seconds", type=float, default=5.0, help="单控制周期时长（秒）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（流量与噪声）")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="结果输出目录")
    p.add_argument("--workers", type=int, default=4, help="并行进程数")
    p.add_argument("--sequential", action="store_true", help="串行执行（便于调试）")
    p.add_argument("--static-replicas", type=int, default=None,
                   help="Static 组副本数；默认按流量峰值×1.15/单副本容量估算")
    p.add_argument("--slo", type=float, default=300.0, help="p95 时延 SLO（毫秒）")
    p.add_argument("--pod-capacity", type=float, default=60.0, help="单副本可持续吞吐（req/s）")
    # ---- 综合评分口径（0-100，越高越好）----
    p.add_argument("--slo-budget", type=float, default=5.0,
                   help="SLO 违反率预算（%%）：0%% 得满分，达到该值质量分归零")
    p.add_argument("--w-quality", type=float, default=0.7,
                   help="综合得分中质量分（SLO 达标度）的权重")
    p.add_argument("--w-cost", type=float, default=0.3,
                   help="综合得分中效率分（相对成本）的权重")
    p.add_argument("--no-plot", action="store_true", help="跳过绘图")
    p.add_argument("--quiet", action="store_true", help="减少控制台输出")
    # 注意：此处**故意不提供** --api-key 参数——命令行参数会进入 shell history
    # 与进程列表，属于明文泄露。Key 请通过环境变量 AI_MAPE_LLM_API_KEY 或 Web 页面输入。
    p.add_argument("--llm", action="store_true",
                   help="启用大模型顾问（参数建议 + 结论报告）")
    p.add_argument("--llm-provider", default=None,
                   help=f"服务商预设：{list(PROVIDER_PRESETS)}")
    p.add_argument("--llm-base-url", default=None, help="OpenAI 兼容端点 Base URL")
    p.add_argument("--llm-model", default=None, help="模型名称")
    return p.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    t_start = time.time()

    # ---------------- 1. 预生成并落盘流量 ----------------
    scenario = make_scenario(profile=args.profile, steps=args.steps, seed=args.seed,
                             step_seconds=args.step_seconds, csv_path=args.traffic_csv)
    os.makedirs(args.outdir, exist_ok=True)
    traffic_csv = scenario.save_csv(os.path.join(args.outdir, "traffic.csv"))
    values = scenario.values

    # ---------------- 2. 实验配置 ----------------
    cfg = SystemConfig(
        step_seconds=args.step_seconds,
        slo_latency_ms=args.slo,
        pod_capacity_rps=args.pod_capacity,
    )
    static_replicas = args.static_replicas
    if static_replicas is None:
        peak = float(np.percentile(values, 99))
        static_replicas = int(math.ceil(peak * 1.15 / cfg.pod_capacity_rps))
        static_replicas = int(max(cfg.min_replicas, min(cfg.max_replicas, static_replicas)))

    if not args.quiet:
        print("\n" + "=" * 78)
        print("  AI-MAPE-K Controller Experiment")
        print("=" * 78)
        print(f"  traffic      : {scenario.name}  steps={scenario.steps}  "
              f"duration={scenario.duration_min:.1f} min  seed={args.seed}")
        st = scenario.stats()
        print(f"                 mean={st['mean']:.0f}  p95={st['p95']:.0f}  "
              f"max={st['max']:.0f} rps")
        print(f"  config       : pod_capacity={cfg.pod_capacity_rps} rps, "
              f"SLO p95={cfg.slo_latency_ms} ms, step={cfg.step_seconds}s")
        print(f"  static group : {static_replicas} replicas (provisioned by peak×1.15)")
        print(f"  traffic file : {traffic_csv}")

    # ---------------- 3. 可选：大模型顾问给出初始参数建议 ----------------
    advisor: Optional[LLMAdvisor] = None
    hints: Optional[Dict[str, Any]] = None
    if args.llm:
        advisor = LLMAdvisor.from_env()
        if args.llm_provider and args.llm_provider in PROVIDER_PRESETS:
            preset = PROVIDER_PRESETS[args.llm_provider]
            advisor.cfg.base_url = args.llm_base_url or preset["base_url"]
            advisor.cfg.model = args.llm_model or preset["model"]
        else:
            if args.llm_base_url:
                advisor.cfg.base_url = args.llm_base_url
            if args.llm_model:
                advisor.cfg.model = args.llm_model

        if not advisor.cfg.configured:
            if not args.quiet:
                print(f"  [llm] 未检测到环境变量 {ENV_KEY}，跳过顾问环节"
                      f"（CLI 不接受明文 Key，请用 Web 页面录入）")
            advisor = None
        else:
            ok, sug = advisor.suggest_params(scenario.stats(), cfg.as_dict(), args.profile)
            if ok:
                hints = sug
                if not args.quiet:
                    print(f"  [llm] 参数建议 (model={advisor.cfg.model}, "
                          f"key={mask_key(advisor.cfg.api_key)}):")
                    print(f"        margin={sug['margin']:.3f}, "
                          f"target_util={sug['target_util']:.3f}, "
                          f"horizon={sug['horizon']}")
                    if sug.get("rationale"):
                        print(f"        rationale: {sug['rationale']}")
            else:
                if not args.quiet:
                    print(f"  [llm] 参数建议失败，沿用默认参数：{sug.get('error', '')[:120]}")

    # ---------------- 4. 并行运行四组控制器 ----------------
    cfg_dict = cfg.as_dict()
    pool = build_controller_pool(cfg, static_replicas=static_replicas)
    keys = [k for k in DEFAULT_CONTROLLER_KEYS if k in pool]

    if not args.quiet:
        print("\n  controllers  :")
        for d in describe_pool(pool):
            print(f"    - {d['label']:<26s} analyzer={d['analyzer']:<12s} "
                  f"planner={d['planner']:<10s} horizon={d['horizon']}")

    results: Dict[str, Dict[str, object]] = {}
    if args.sequential or args.workers <= 1:
        for k in keys:
            results[k] = run_single_controller(k, values, cfg_dict, args.seed,
                                               static_replicas, hints)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(keys))) as ex:
            futures = {
                ex.submit(run_single_controller, k, values, cfg_dict,
                          args.seed, static_replicas, hints): k
                for k in keys
            }
            for fut in as_completed(futures):
                k = futures[fut]
                results[k] = fut.result()

    # ---------------- 5. 汇总评估 ----------------
    ev = Evaluator(cfg, slo_budget_pct=args.slo_budget,
                   w_quality=args.w_quality, w_cost=args.w_cost)
    for k in keys:
        r = results[k]
        ev.begin_run(k, str(r["label"]), meta=dict(r["meta"]))  # type: ignore[arg-type]
        for s in r["steps"]:  # type: ignore[union-attr]
            ev.record(k, ControlStep(**s))  # type: ignore[arg-type]
        ev.attach_forecast(k, r["forecast_pairs"],      # type: ignore[arg-type]
                           r["forecast_pairs_h1"])      # type: ignore[arg-type]
        ev.attach_scale_events(k, r["scale_events"])  # type: ignore[arg-type]

    summary = ev.summary()

    if not args.quiet:
        ev.print_table()

    # ---------------- 6. 落盘 ----------------
    summary_csv = ev.to_csv(os.path.join(args.outdir, "comparison.csv"))
    summary_json = ev.to_json(os.path.join(args.outdir, "comparison.json"))
    traces = ev.to_traces_csv(args.outdir)

    paths: Dict[str, str] = {
        "traffic": traffic_csv,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "traces": ";".join(traces),
    }

    # ---------------- 7. 出图 ----------------
    if not args.no_plot:
        try:
            paths.update(plot_all(ev, scenario_values=values,
                                  outdir=args.outdir, config=cfg, scenario=scenario))
        except Exception as exc:  # 绘图失败不应中断实验结论
            print(f"  [warn] 绘图失败：{exc}")

    # ---------------- 8. 可选：大模型结论报告 ----------------
    if advisor is not None and advisor.cfg.configured:
        ok, report = advisor.explain_results(summary, cfg.as_dict(), args.profile)
        if ok:
            rp = os.path.join(args.outdir, "report.md")
            with open(rp, "w", encoding="utf-8") as f:
                f.write(report if report.endswith("\n") else report + "\n")
            paths["report"] = rp
            if not args.quiet:
                print("\n  [llm] 结论报告：")
                for line in report.strip().splitlines():
                    print("    " + line)
        elif not args.quiet:
            print(f"  [llm] 报告生成失败：{str(report)[:160]}")

    elapsed = time.time() - t_start

    if not args.quiet:
        print(f"  elapsed      : {elapsed:.2f}s")
        print("  artifacts    :")
        for name, p in paths.items():
            if name == "traces":
                print(f"    - traces   : {len(traces)} files under {args.outdir}")
            else:
                print(f"    - {name:<12s}: {p}")
        _print_conclusion(summary, args.slo_budget, args.w_quality, args.w_cost)

    return {"evaluator": ev, "summary": summary, "paths": paths,
            "scenario": scenario, "config": cfg}


def _print_conclusion(summary: Dict[str, Dict[str, float]],
                      slo_budget_pct: float = 5.0,
                      w_quality: float = 0.7, w_cost: float = 0.3) -> None:
    """自动生成结论摘要：AI-MAPE 相对各组基线的改进幅度 + 综合排名。"""
    if "ai_mape" not in summary:
        return
    ai = summary["ai_mape"]
    ai_v, ai_c = ai["slo_violation_rate_pct"], ai["cost_per_1k_requests"]
    print("\n  conclusion   :  (AI-MAPE vs baselines; negative = AI-MAPE better)")
    for key in ("static", "hpa", "noai_mape"):
        if key not in summary:
            continue
        base = summary[key]
        bv, bc = base["slo_violation_rate_pct"], base["cost_per_1k_requests"]
        dv = ai_v - bv                       # 正 = AI 违反更多（更差）
        rel_c = (ai_c - bc) / bc * 100.0 if bc > 1e-9 else 0.0
        print(f"    vs {key:<10s}: SLO violation {bv:6.2f}% -> {ai_v:6.2f}% "
              f"({dv:+6.2f} pp) | cost/1k req {bc:6.2f} -> {ai_c:6.2f} ({rel_c:+6.1f}%)")

    # 预测精度：统一在 h=1 视界下比较，避免“视界越长误差越大”的口径偏差
    ai_rmse = ai.get("rmse_h1", float("nan"))
    ai_mape = ai.get("mape_h1_pct", float("nan"))
    print(f"    AI-MAPE   forecast @h=1: RMSE={ai_rmse:.2f} rps, MAPE={ai_mape:.2f}% "
          f"(decision horizon h={ai.get('forecast_horizon', float('nan')):.0f})")
    if "noai_mape" in summary:
        nb = summary["noai_mape"]
        n_rmse, n_mape = nb.get("rmse_h1", float("nan")), nb.get("mape_h1_pct", float("nan"))
        print(f"    NoAI-MAPE forecast @h=1: RMSE={n_rmse:.2f} rps, MAPE={n_mape:.2f}%")
        if n_rmse == n_rmse and ai_rmse == ai_rmse and n_rmse > 1e-9:
            print(f"    -> same-horizon RMSE reduction: "
                  f"{(n_rmse - ai_rmse) / n_rmse * 100:.1f}%")

    # ---- 综合得分（0-100，越高越好）：质量 + 效率的显式加权 ----
    print(f"\n  composite    :  (0-100, higher is better; "
          f"quality {w_quality:.2f} / efficiency {w_cost:.2f}, "
          f"SLO budget {slo_budget_pct:.1f}%)")
    order = sorted(summary, key=lambda k: -summary[k].get("composite_score", float("-inf")))
    for k in order:
        m = summary[k]
        mark = " [pareto]" if m.get("pareto_optimal", 0) >= 0.5 else ""
        print(f"    #{int(m.get('rank', 0))} {k:<10s} score {m['composite_score']:6.2f}  "
              f"(quality {m['quality_score']:5.1f} / efficiency {m['efficiency_score']:5.1f}, "
              f"cost {m['cost_per_1k_requests']:6.2f}{mark})")

    pareto = [k for k in order if summary[k].get("pareto_optimal", 0) >= 0.5]
    ok = [k for k in order if summary[k].get("slo_compliant", 0) >= 0.5]
    if len(pareto) == len(order):
        print(f"    pareto front : all {len(order)} controllers are mutually non-dominated "
              f"(i.e. they trace a quality-cost trade-off curve)")
    else:
        dominated = [k for k in order if k not in pareto]
        print(f"    pareto front : {', '.join(pareto)}   | dominated: {', '.join(dominated)}")
    print(f"    budget-pass  : {', '.join(ok) if ok else '(none)'}"
          f"  —— SLO 预算 {slo_budget_pct:.1f}% 内共 {len(ok)}/{len(order)} 组达标")
    if ok:
        cheapest = min(ok, key=lambda k: summary[k]["cost_per_1k_requests"])
        print(f"    best-in-budget: {cheapest} "
              f"(达标组中成本最低 = {summary[cheapest]['cost_per_1k_requests']:.2f} / 1k req)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
