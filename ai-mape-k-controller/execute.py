"""
execute.py —— MAPE-K 的 **Execute（E）** 环节：把规划结果施加到被管系统。

提供两种执行器：
1. `SimulatedExecutor`：实验默认使用，把副本变更施加给 `ApplicationPlant`，
   统一处理边界钳制、冷却期与成本记账；
2. `KubernetesExecutor`：面向真实集群的适配器（默认 `dry_run=True`），
   通过 `kubectl scale` 下发变更，配合 k8s/ 目录下的清单即可实机验证。

二者实现同一接口 `apply(decision) -> ExecutionResult`，可在实验中无缝替换。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from knowledge import DEFAULT_CONFIG, ScaleDecision, SystemConfig


@dataclass
class ExecutionResult:
    applied: bool
    target: int
    delta: int
    pod_cost: float = 0.0      # 本周期资源成本（按期望副本计）
    scale_cost: float = 0.0    # 本周期扩缩容抖动成本
    reason: str = ""
    raw: Dict[str, object] = field(default_factory=dict)


class BaseExecutor:
    """执行器接口。"""

    name = "base"

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.cfg = config
        self.last_change_t: int = -10 ** 9
        self.last_target: Optional[int] = None
        self.history: List[ExecutionResult] = []

    def _within_cooldown(self, t: int) -> bool:
        return (t - self.last_change_t) < self.cfg.cooldown_steps

    def clamp(self, n: int) -> int:
        return int(max(self.cfg.min_replicas, min(self.cfg.max_replicas, int(n))))

    def apply(self, decision: ScaleDecision, t: int, current: int) -> ExecutionResult:
        raise NotImplementedError

    def reset(self) -> None:
        self.last_change_t = -10 ** 9
        self.last_target = None
        self.history = []


# --------------------------------------------------------------------------- #
# 仿真执行器
# --------------------------------------------------------------------------- #
class SimulatedExecutor(BaseExecutor):
    """把决策施加到仿真 Plant 上。"""

    name = "simulated"

    def __init__(self, plant, config: SystemConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self.plant = plant

    def apply(self, decision: ScaleDecision, t: int, current: int) -> ExecutionResult:
        target = self.clamp(decision.target_replicas)
        delta = target - current
        reason = decision.reason

        if delta == 0:
            res = ExecutionResult(False, target, 0,
                                  pod_cost=target * self.cfg.pod_cost_per_step,
                                  scale_cost=0.0, reason=reason or "no-op")
        elif self._within_cooldown(t):
            # 冷却期内拒绝变更（真实控制器同样受 --scale-down-delay 等限制）
            res = ExecutionResult(False, current, 0,
                                  pod_cost=current * self.cfg.pod_cost_per_step,
                                  scale_cost=0.0, reason="blocked-by-cooldown")
        else:
            self.plant.apply_replicas(target, reason=reason)
            self.last_change_t = t
            res = ExecutionResult(
                True, target, delta,
                pod_cost=target * self.cfg.pod_cost_per_step,
                scale_cost=abs(delta) * self.cfg.scale_cost_per_change,
                reason=reason,
            )
        self.last_target = res.target
        self.history.append(res)
        return res


# --------------------------------------------------------------------------- #
# 真实集群执行器
# --------------------------------------------------------------------------- #
class KubernetesExecutor(BaseExecutor):
    """面向真实 K8s 集群的执行器。

    依赖：kubectl 已配置 kubeconfig，且 k8s/rbac.yaml 已授予
    deployments/get + deployments/scale 权限。

    默认 `dry_run=True`：只打印将下发的命令，不产生副作用。
    """

    name = "kubernetes"

    def __init__(self, deployment: str = "demo-app", namespace: str = "default",
                 config: SystemConfig = DEFAULT_CONFIG, dry_run: bool = True,
                 kubectl: str = "kubectl"):
        super().__init__(config)
        self.deployment = deployment
        self.namespace = namespace
        self.dry_run = dry_run
        self.kubectl = kubectl
        self.commands: List[str] = []

    def _build_cmd(self, target: int) -> List[str]:
        return [self.kubectl, "-n", self.namespace, "scale",
                f"deployment/{self.deployment}", f"--replicas={target}"]

    def apply(self, decision: ScaleDecision, t: int, current: int) -> ExecutionResult:
        target = self.clamp(decision.target_replicas)
        delta = target - current
        cmd = self._build_cmd(target)
        self.commands.append(" ".join(cmd))

        if delta == 0:
            res = ExecutionResult(False, target, 0,
                                  pod_cost=target * self.cfg.pod_cost_per_step,
                                  reason=decision.reason or "no-op", raw={"cmd": " ".join(cmd)})
        elif self._within_cooldown(t):
            res = ExecutionResult(False, current, 0,
                                  pod_cost=current * self.cfg.pod_cost_per_step,
                                  reason="blocked-by-cooldown", raw={"cmd": " ".join(cmd)})
        else:
            if self.dry_run:
                out, ok = f"[dry-run] {' '.join(cmd)}", True
            else:
                if shutil.which(self.kubectl) is None:
                    raise RuntimeError(f"未找到 kubectl：{self.kubectl}")
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                out, ok = (proc.stdout or proc.stderr).strip(), proc.returncode == 0
            self.last_change_t = t
            res = ExecutionResult(
                ok, target if ok else current, delta if ok else 0,
                pod_cost=(target if ok else current) * self.cfg.pod_cost_per_step,
                scale_cost=abs(delta) * self.cfg.scale_cost_per_change if ok else 0.0,
                reason=decision.reason, raw={"cmd": " ".join(cmd), "output": out},
            )
        self.last_target = res.target
        self.history.append(res)
        return res
