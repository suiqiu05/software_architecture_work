"""
llm_advisor.py —— 可选的大模型策略顾问（LLM-as-Advisor）。

定位说明（重要）：
    本项目的 "AI-MAPE" 默认使用**本地在线学习算法**（Hedge 集成预测 + 自适应安全
    边际），运行在毫秒级控制环内，不依赖任何外部服务，也不需要 API Key。
    大模型的价值不在于替控制环做实时决策（网络时延与不稳定性会破坏控制律），
    而在于三个**低频、高语义**的环节：
      1) 实验前：根据流量形态画像推荐初始控制参数（margin / target_util / 视界）；
      2) 实验后：把指标摘要翻译成可执行的运维结论报告；
      3) 运行期（可选，默认关闭）：对持续恶化的区间做归因诊断。

凭据安全约定：
    - API Key 只从**运行时**获取：环境变量 `AI_MAPE_LLM_API_KEY` 或 Web 页面提交；
    - 绝不写入源码、不写入结果文件、不打印完整 Key（日志仅保留前 4 后 4 位）；
    - 默认仅驻留内存；除非用户显式选择"写入环境变量"，否则不落盘。
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

ENV_KEY = "AI_MAPE_LLM_API_KEY"
ENV_BASE_URL = "AI_MAPE_LLM_BASE_URL"
ENV_MODEL = "AI_MAPE_LLM_MODEL"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# 常见 OpenAI 兼容端点预设（用户可在页面下拉选择，也可自定义）
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "DeepSeek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "Moonshot (Kimi)": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "阿里云百炼(兼容)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "智谱 GLM": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    "本地 Ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
}


def mask_key(key: Optional[str]) -> str:
    """Key 脱敏展示：仅保留前 4 与后 4 位。"""
    if not key:
        return ""
    k = str(key)
    if len(k) <= 8:
        return "*" * len(k)
    return f"{k[:4]}{'*' * (len(k) - 8)}{k[-4:]}"


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 25.0
    temperature: float = 0.2
    enabled: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def safe_dict(self) -> Dict[str, Any]:
        """可安全展示/落盘的配置（Key 已脱敏）。"""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "configured": self.configured,
            "api_key_masked": mask_key(self.api_key),
            "timeout": self.timeout,
            "enabled": self.enabled,
        }


class LLMAdvisor:
    """OpenAI 兼容接口的轻量客户端 + 面向本实验的三类顾问能力。"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.cfg = config or LLMConfig()
        self.last_error: Optional[str] = None
        self.calls: int = 0
        self.tokens_hint: int = 0

    # ---------------- 凭据注入 ----------------
    @classmethod
    def from_env(cls) -> "LLMAdvisor":
        """从环境变量读取凭据（key 为空时仅是未配置，不报错）。"""
        return cls(LLMConfig(
            api_key=os.environ.get(ENV_KEY, ""),
            base_url=os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
            model=os.environ.get(ENV_MODEL, DEFAULT_MODEL),
        ))

    def set_credentials(self, api_key: str, base_url: Optional[str] = None,
                        model: Optional[str] = None,
                        export_env: bool = False) -> None:
        """运行时注入凭据。

        export_env=True 时同步写入当前进程环境变量，使并行子进程同样可见；
        仅影响进程内存，不写任何文件。
        """
        self.cfg.api_key = (api_key or "").strip()
        if base_url:
            self.cfg.base_url = base_url.strip().rstrip("/")
        if model:
            self.cfg.model = model.strip()
        if export_env:
            os.environ[ENV_KEY] = self.cfg.api_key
            os.environ[ENV_BASE_URL] = self.cfg.base_url
            os.environ[ENV_MODEL] = self.cfg.model

    def clear_credentials(self) -> None:
        self.cfg.api_key = ""
        os.environ.pop(ENV_KEY, None)

    # ---------------- 底层调用 ----------------
    def chat(self, messages: Sequence[Dict[str, str]],
             max_tokens: int = 800, temperature: Optional[float] = None) -> Tuple[bool, str]:
        """调用 /chat/completions。失败返回 (False, 原因)，不抛异常。"""
        if not self.cfg.configured:
            return False, "未配置 API Key（LLM 功能处于关闭状态）"
        url = f"{self.cfg.base_url}/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": list(messages),
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg.api_key}"},
            method="POST",
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout, context=ctx) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
            self.calls += 1
            choices = body.get("choices") or []
            if not choices:
                return False, f"响应缺少 choices：{str(body)[:200]}"
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return False, "响应内容为空"
            usage = body.get("usage") or {}
            self.tokens_hint += int(usage.get("total_tokens", 0) or 0)
            return True, content
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            self.last_error = f"HTTP {e.code}: {detail}"
            return False, self.last_error
        except urllib.error.URLError as e:
            self.last_error = f"网络不可达：{e.reason}"
            return False, self.last_error
        except Exception as e:  # noqa: BLE001 - UI 需要稳定的失败原因
            self.last_error = f"{type(e).__name__}: {e}"
            return False, self.last_error

    def test_connection(self) -> Tuple[bool, str]:
        """连通性自检：一次极短对话。"""
        ok, msg = self.chat(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=8, temperature=0.0,
        )
        return (True, f"连接成功 · model={self.cfg.model}") if ok else (False, msg)

    # ---------------- 1) 实验前：参数建议 ----------------
    def suggest_params(self, traffic_stats: Dict[str, float], config: Dict[str, float],
                       profile: str = "composite") -> Tuple[bool, Dict[str, Any]]:
        """根据流量画像推荐 AdaptivePlanner 的初始参数。"""
        prompt = f"""你是自适应扩容控制专家。下面是即将回放的负载画像与被管系统参数。

流量形态: {profile}
统计(rps): mean={traffic_stats.get('mean', 0):.0f}, p50={traffic_stats.get('p50', 0):.0f}, \
p95={traffic_stats.get('p95', 0):.0f}, p99={traffic_stats.get('p99', 0):.0f}, \
max={traffic_stats.get('max', 0):.0f}, std={traffic_stats.get('std', 0):.0f}
系统: 单副本可持续吞吐={config.get('pod_capacity_rps', 60):.0f} rps, \
SLO p95={config.get('slo_latency_ms', 300):.0f} ms, 控制周期={config.get('step_seconds', 5):.0f} s, \
副本冷启动={config.get('startup_steps', 2)} 个周期, 副本上限={config.get('max_replicas', 40)}

请给出 AI-MAPE 自适应规划器的初始参数，只输出 JSON，不要解释：
{{"margin": <float 0.95~1.35, 负载侧安全系数>, \
"target_util": <float 0.70~0.97, 单副本目标利用率>, \
"horizon": <int 1~5, 预测视界(控制周期数)>, \
"rationale": "<不超过40字的原因>"}}"""
        ok, text = self.chat([{"role": "user", "content": prompt}], max_tokens=300)
        if not ok:
            return False, {"error": text}
        parsed = _extract_json(text)
        if not isinstance(parsed, dict):
            return False, {"error": "模型未返回可解析的 JSON", "raw": text[:300]}
        return True, _sanitize_params(parsed)

    # ---------------- 2) 实验后：结论报告 ----------------
    def explain_results(self, summary: Dict[str, Dict[str, float]],
                        config: Dict[str, float], profile: str = "composite"
                        ) -> Tuple[bool, str]:
        """把四组指标翻译成运维结论（Markdown）。"""
        lines = []
        for k, v in summary.items():
            lines.append(
                f"- {k}: SLO违反率={v.get('slo_violation_rate_pct', float('nan')):.2f}%, "
                f"p95均值={v.get('mean_p95_ms', float('nan')):.0f}ms, "
                f"p95的p95={v.get('p95_of_p95_ms', float('nan')):.0f}ms, "
                f"平均副本={v.get('mean_replicas', float('nan')):.2f}, "
                f"成本/千请求={v.get('cost_per_1k_requests', float('nan')):.2f}, "
                f"调整次数={v.get('scale_changes', 0):.0f}, "
                f"预测RMSE@h1={v.get('rmse_h1', float('nan'))}"
            )
        table = "\n".join(lines)
        prompt = f"""你是 SRE 顾问。以下是在同一段回放流量（形态={profile}，SLO p95={config.get('slo_latency_ms', 300):.0f}ms）\
上对比四种扩容控制器的实验结果：

{table}

请输出一份不超过 300 字的中文结论报告（Markdown），包含：
1) 哪种策略综合最优、依据是什么；
2) 各策略的失效模式（为什么会违反 SLO 或浪费资源）；
3) 两条可落地的调参建议。
只输出报告正文，不要复述表格。"""
        return self.chat([{"role": "user", "content": prompt}], max_tokens=900)

    # ---------------- 3) 运行期：低频归因诊断 ----------------
    def diagnose(self, window_desc: str) -> Tuple[bool, str]:
        """对一段持续违反 SLO 的区间做归因（默认关闭，由调用方决定是否触发）。"""
        prompt = f"""以下是自适应扩容系统在某时间窗的运行状态：
{window_desc}
请用不超过 80 字说明最可能的违反原因，并给出一条处置建议。只输出结论。"""
        return self.chat([{"role": "user", "content": prompt}], max_tokens=200)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> Optional[Any]:
    """从模型回复中提取 JSON（容忍 ```json 代码围栏与前后噪声）。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start:end + 1] if start >= 0 and end > start else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _clamp(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(hi, v))


def _sanitize_params(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """把模型建议约束到安全区间，避免异常输出破坏控制环。"""
    margin = _clamp(parsed.get("margin"), 0.95, 1.35, 1.05)
    target_util = _clamp(parsed.get("target_util"), 0.70, 0.97, 0.92)
    try:
        horizon = int(float(parsed.get("horizon", 3)))
    except (TypeError, ValueError):
        horizon = 3
    horizon = max(1, min(5, horizon))
    return {
        "margin": margin,
        "target_util": target_util,
        "horizon": horizon,
        "rationale": str(parsed.get("rationale", ""))[:200],
    }
