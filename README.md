# software_architecture_work

软件体系结构课程作业仓库。当前包含一个子项目：

- **`ai-mape-k-controller/`**：基于 MAPE-K 控制环的 Kubernetes 弹性伸缩控制器对比实验平台。

---

## ai-mape-k-controller

用一套可复现的仿真实验，横向对比四类控制器在**同一段流量**下的表现：

| 控制器 | Monitor | Analyze | Plan | 在线学习 |
|---|---|---|---|---|
| `Static` | ✔ | ✗ 无预测 | 按峰值固定预留 | ✗ |
| `HPA` | ✔ | ✗ 无预测 | K8s 原生比值控制律 + 稳定窗口 | ✗ |
| `NoAI-MAPE` | ✔ | 指数平滑 + 线性趋势外推 | 阈值规则 + 固定安全余量 | ✗ |
| `AI-MAPE` | ✔ | 在线集成预测（Hedge，4 个基预测器） | 分位数上界 + 自适应安全边际 + 迟滞防抖 | ✔ |

> 说明：`AI-MAPE` 的 "AI" 指**运行在控制环内部的本地在线学习算法**（毫秒级、不依赖外部服务、无需 API Key）。
> 大模型（`llm_advisor.py`）只作为低频顾问，负责实验前推荐参数与实验后生成结论报告，不参与实时决策。

### 目录与模块

| 文件 | 作用 |
|---|---|
| `main_exp.py` | 实验总调度：预生成流量 → 进程池并行跑四组 → 统一评估 → 自动出图 |
| `knowledge.py` | MAPE-K 的 **K**：SLO/系统常量、观测与决策数据结构、历史时序与在线模型状态 |
| `monitor.py` | **M** 环节 + 被管系统仿真（M/M/N 排队近似算 p95 时延、CPU、错误率） |
| `analyze.py` | **A** 环节：启发式分析器 vs 在线集成分析器，以及 RMSE / MAPE 等误差度量 |
| `plan.py` | **P** 环节：四类规划器，把分析结论转成副本数决策 |
| `execute.py` | **E** 环节：仿真执行器 / kubectl 真实集群执行器（同一接口） |
| `controller_pool.py` | 四组控制器的封装与统一接口 `observe()` / `decide()` |
| `traffic_generator.py` | 8 种负载形态 + composite 组合 + CSV 流量回放 |
| `evaluator.py` | 采集服务质量（SLO 违反率、p95）、资源成本、预测误差并落盘 |
| `visual_compare.py` | 输出 `comparison.png`（四组叠加对比）与 `forecast_error.png`（预测误差专项） |
| `webui.py` + `web/index.html` | 零第三方依赖的 Web 操作控制台（标准库 `http.server`） |
| `llm_advisor.py` | 可选的大模型顾问（OpenAI 兼容协议，无 Key 自动降级） |
| `bootstrap.py` | 运行环境自举：注入 `sys.path` + 依赖预检，缺失时给出修复指引 |
| `jsonutil.py` | 严格 JSON 序列化：把 `NaN` / `Infinity` 转为 `null`，保证浏览器可解析 |
| `pack.py` / `pack.bat` | 一键打包成不含任何本机路径的 zip |
| `setup_env.*` / `run.*` / `webui.bat` | 一键建虚拟环境、跑实验、开控制台（Windows / Linux 均提供） |
| `k8s/` | `demo-app.yaml`、`hpa.yaml`、`rbac.yaml`：真实集群验证用清单 |

### 快速开始

Windows（双击即可）：

```
setup_env.bat      # 建 .venv 并安装 numpy / matplotlib
run.bat            # 跑默认实验（composite 形态 720 步 ≈ 60 分钟）
webui.bat          # 打开 Web 控制台 http://127.0.0.1:8765
```

Linux / macOS：

```bash
bash setup_env.sh
bash run.sh
python3 webui.py --port 8765
```

手动运行（已有 Python 3.9+ 与依赖时）：

```bash
python main_exp.py                              # 默认：composite 形态 720 步
python main_exp.py --profile burst --steps 600  # 单一形态
python main_exp.py --seed 7 --outdir results/run7
python main_exp.py --sequential                 # 关闭并行，便于调试
python main_exp.py --traffic-csv my.csv         # 回放真实流量
```

常用参数：`--profile`(steady/diurnal/burst/random_walk/staircase/ramp/oscillation/spike/composite)、
`--steps`、`--seed`、`--slo`、`--pod-capacity`、`--workers`、`--outdir`、`--no-plot`、`--llm`；
评分口径：`--slo-budget`（违反率预算，默认 5%）、`--w-quality`（默认 0.7）、`--w-cost`（默认 0.3）。

结果默认输出在脚本所在目录的 `results/`（不污染当前工作目录）：`comparison.csv/json`、
`trace_*.csv`、两张 PNG 与 `comparison` 指标摘要。

### 典型结果（默认参数，720 步）

| 控制器 | SLO 违反率 | 成本/千请求 | 平均副本 | 综合分 | 预测 RMSE@h=1 |
|---|---|---|---|---|---|
| Static | **0.56%** | 10.20 | 12.99 | **75.9** | — |
| K8s HPA | 8.06% | 6.27 | 6.87 | 22.3 | — |
| NoAI-MAPE | 11.25% | **4.66** | 5.42 | 30.0 | 60.03 |
| AI-MAPE | 2.92% | 8.54 | 10.01 | 45.5 | **51.09** |

**综合分口径**（0–100，越高越好，`evaluator.py`）：

```
quality    = 100 × clip(1 − 违反率 / SLO预算, 0, 1)      # SLO预算默认 5%
efficiency = 100 × (最低成本组成本 / 该组成本)            # 相对效率
composite  = 0.7 × quality + 0.3 × efficiency            # 权重可用 --w-quality/--w-cost 调整
```

两个指标先归一化到 0–100 再加权，避免"违反率(%) + 系数×成本/千请求"这种跨量纲裸加和
（成本项的绝对大小取决于 `pod_cost_per_step` 的标定值，换一个成本常数排序就会变）。

读法：

- **AI-MAPE** 用 2.92% 的违反率、成本仅为 Static 的 84%，综合分 45.5，在四个 MAPE/基线里最高；
  同视界（h=1）预测 RMSE 比 NoAI-MAPE 低 14.9%。
- **Static** 综合分最高（75.9），因为它的违反率接近 0；但它用 12.99 个常驻副本换来这个成绩，
  单位请求成本是 NoAI 的 2.19 倍。若按 5% 的 SLO 预算做**硬约束**筛选，仅 Static 与 AI-MAPE 达标，
  其中 AI-MAPE 成本最低（8.54 vs 10.20），是"达标前提下的成本最优解"。
- 四组在（违反率, 成本）平面上**互不支配**，构成一条 trade-off 曲线，因此任何排名都依赖权重选择，
  建议同时看 `--w-quality/--w-cost` 的敏感性，而不是只引用一个总分。

### 可移植性

所有路径均由 `__file__` 推导，代码中不含任何绝对路径；`pack.py` 打包时排除 `.venv`、`results`、
`__pycache__`、`.git`，因此 zip 可直接拷到任意机器：解压 → `setup_env` 装依赖 → `run` 跑实验。

### 安全约定

API Key 只在运行时提供（Web 页面录入或环境变量 `AI_MAPE_LLM_API_KEY`），仅驻留进程内存，
不写入源码、结果文件或日志（展示一律脱敏）。不配置 Key 时全部功能照常可用。
