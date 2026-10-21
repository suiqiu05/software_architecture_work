"""
webui.py —— 简易操作页面（零第三方依赖，基于 Python 标准库 http.server）。

设计目标：
1. 把实验参数、LLM 凭据、运行与结果查看都搬到浏览器里，无需改代码；
2. **API Key 只在页面录入**：提交后仅驻留服务端进程内存（可选同步到进程环境变量，
   便于并行子进程读取），不写入任何源码、结果文件或日志（展示一律脱敏）；
3. 与 main_exp.run_experiment 复用同一条实验流水线，保证 CLI 与页面结果一致。

启动：
    python webui.py --port 8765          # 浏览器打开 http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import threading
import time
import traceback
import uuid
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# 启动预检：必须在 import 本地模块（含 main_exp）之前执行
from bootstrap import ensure_ready  # noqa: E402

ensure_ready(entry="webui.py")

from knowledge import SystemConfig  # noqa: E402
from llm_advisor import (ENV_KEY, PROVIDER_PRESETS, LLMAdvisor, LLMConfig,
                         mask_key)
from traffic_generator import TRAFFIC_KINDS

import jsonutil
from jsonutil import json_safe as _jsonutil_safe

import main_exp  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
RESULTS_DIR = os.path.join(ROOT, "results")

# 全局（进程内）凭据与任务表
_ADVISOR = LLMAdvisor(LLMConfig())
_ADVISOR_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
# 实验串行锁：contextlib.redirect_stdout 是**进程级**的，两个实验并发执行会
# 互相覆盖 stdout，导致日志串台甚至输出丢失；同时并发也会让 CPU 争抢、
# 计时失真。这里用锁排队，语义上等价于"一次只跑一个实验"。
_RUN_LOCK = threading.Lock()

# 访问日志（环形 + 追加落盘）：用于排查"页面明明打开了却连不上接口"一类问题，
# 只记录 方法/路径/状态码/来源 Origin，不含任何凭据。
_ACCESS = deque(maxlen=400)
_ACCESS_LOCK = threading.Lock()
ACCESS_LOG = os.path.join(RESULTS_DIR, "access.log")

def json_safe(obj: Any) -> Any:
    """把对象清洗成严格合法的 JSON 值（实现集中在 jsonutil，避免两份逻辑漂移）。"""
    return _jsonutil_safe(obj)


STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


# --------------------------------------------------------------------------- #
# 任务执行
# --------------------------------------------------------------------------- #
def _run_job(job_id: str, payload: Dict[str, Any]) -> None:
    """排队执行：同一时刻只允许一个实验占用 stdout 与 CPU。"""
    with _RUN_LOCK:
        _run_job_inner(job_id, payload)


def _run_job_inner(job_id: str, payload: Dict[str, Any]) -> None:
    """后台线程执行实验，捕获 stdout 作为页面日志。

    健壮性要点：
    - 记录 started_at / finished_at / elapsed，前端可据此显示"已运行多久"，
      避免任务异常时页面只显示一个永不消失的"运行中"；
    - 捕获 BaseException（含 argparse 的 SystemExit），否则该线程会静默
      死亡而 status 永远停在 running；
    - params 一并回传，页面上能直接看到本次用的 seed/profile，
      从而区分"图没变"到底是缓存还是参数确实相同。
    """
    buf = io.StringIO()
    t_start = time.time()
    params = {
        "profile": str(payload.get("profile", "composite")),
        "steps": int(payload.get("steps", 720)),
        "step_seconds": float(payload.get("step_seconds", 5.0)),
        "seed": int(payload.get("seed", 42)),
        "slo": float(payload.get("slo", 300.0)),
        "pod_capacity": float(payload.get("pod_capacity", 60.0)),
        "workers": int(payload.get("workers", 1)),
        "use_llm": bool(payload.get("use_llm", False)),
    }
    with _JOBS_LOCK:
        _JOBS[job_id].update({"status": "running",
                              "started_at": t_start,
                              "params": params})
    try:
        ns = main_exp.parse_args([])
        ns.profile = str(payload.get("profile", "composite"))
        ns.traffic_csv = None
        ns.steps = int(payload.get("steps", 720))
        ns.step_seconds = float(payload.get("step_seconds", 5.0))
        ns.seed = int(payload.get("seed", 42))
        ns.outdir = os.path.join(RESULTS_DIR, "web")
        ns.workers = int(payload.get("workers", 1))
        ns.sequential = ns.workers <= 1
        ns.static_replicas = payload.get("static_replicas") or None
        if ns.static_replicas is not None:
            ns.static_replicas = int(ns.static_replicas)
        ns.slo = float(payload.get("slo", 300.0))
        ns.pod_capacity = float(payload.get("pod_capacity", 60.0))
        ns.no_plot = False
        ns.quiet = False
        ns.llm = bool(payload.get("use_llm", False))
        ns.llm_provider = payload.get("llm_provider") or None
        ns.llm_base_url = payload.get("llm_base_url") or None
        ns.llm_model = payload.get("llm_model") or None

        with contextlib.redirect_stdout(buf):
            out = main_exp.run_experiment(ns)

        summary = out["summary"]
        paths = {k: _to_url(v) for k, v in out["paths"].items() if k != "traces"}
        report = ""
        rp = os.path.join(ns.outdir, "report.md")
        if os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as f:
                report = f.read()

        t_end = time.time()
        with _JOBS_LOCK:
            _JOBS[job_id].update({
                "status": "done", "summary": summary, "paths": paths,
                "report": report, "log": buf.getvalue(),
                "finished_at": t_end, "elapsed": round(t_end - t_start, 2),
                "scenario": {
                    "name": out["scenario"].name,
                    "steps": out["scenario"].steps,
                    "stats": out["scenario"].stats(),
                },
            })
    except BaseException as exc:  # noqa: BLE001
        # BaseException: argparse 出错会抛 SystemExit，普通 except 兜不住，
        # 那样线程会静默退出，前端就会永远停在"运行中"。
        t_end = time.time()
        with _JOBS_LOCK:
            _JOBS[job_id].update({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-4000:],
                "log": buf.getvalue(),
                "finished_at": t_end, "elapsed": round(t_end - t_start, 2),
            })


def _to_url(abs_path: str) -> str:
    rel = os.path.relpath(abs_path, ROOT).replace("\\", "/")
    return "/" + rel


# --------------------------------------------------------------------------- #
# HTTP 处理
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "AIMapeWebUI/1.0"

    def log_message(self, fmt: str, *args) -> None:  # 静音默认访问日志
        pass

    # ---------------- 工具 ----------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        try:
            self._log_access(code)
        except Exception:  # 日志失败绝不影响响应
            pass
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # 允许任意来源：页面可能经预览面板/代理打开，避免被同源策略拦下
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _log_access(self, code: int) -> None:
        """记录一次请求：方法 / 路径 / 状态码 / Origin。供 /api/access 与落盘查看。"""
        origin = self.headers.get("Origin") or "-"
        rec = (round(time.time(), 3), self.command or "-",
               (self.path or "-")[:120], int(code), origin[:60])
        with _ACCESS_LOCK:
            _ACCESS.append(rec)
            try:
                if os.path.exists(ACCESS_LOG) and os.path.getsize(ACCESS_LOG) > 2 * 1024 * 1024:
                    os.replace(ACCESS_LOG, ACCESS_LOG + ".old")   # 简单滚动，避免无限增长
                with open(ACCESS_LOG, "a", encoding="utf-8") as fh:
                    fh.write("%.3f\t%s\t%s\t%d\t%s\n" % rec)
            except OSError:
                pass

    def _json(self, obj: Any, code: int = 200) -> None:
        payload = json.dumps(json_safe(obj), ensure_ascii=False, default=str)
        self._send(code, payload.encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:  # 预检请求
        self._send(204, b"", "text/plain")

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _static(self, rel_path: str) -> None:
        # URL /results/xxx 直接映射到 results 目录，其余映射到项目根目录
        if rel_path.startswith("results/"):
            base, sub = RESULTS_DIR, rel_path[len("results/"):]
        else:
            base, sub = ROOT, rel_path
        target = os.path.normpath(os.path.join(base, sub))
        if not (target == base or target.startswith(base + os.sep)):
            self._send(403, b"forbidden", "text/plain; charset=utf-8")
            return
        if not os.path.isfile(target):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ext = os.path.splitext(target)[1].lower()
        ctype = STATIC_TYPES.get(ext, "application/octet-stream")
        with open(target, "rb") as f:
            self._send(200, f.read(), ctype)

    # ---------------- 路由 ----------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._static("web/index.html")
            return
        if path.startswith("/api/config"):
            with _ADVISOR_LOCK:
                cfg = _ADVISOR.cfg.safe_dict()
            self._json({
                "llm": cfg,
                "providers": PROVIDER_PRESETS,
                "env_var": ENV_KEY,
                "traffic_kinds": list(TRAFFIC_KINDS),
                "defaults": SystemConfig().as_dict(),
            })
            return
        if path.startswith("/api/access"):
            with _ACCESS_LOCK:
                rows = list(_ACCESS)
            self._json({"log": rows})
            return
        if path.startswith("/api/ping"):
            # 极轻量健康检查：前端判定"服务是否可达"只打这个端点，
            # 与任务查询解耦，避免把任务查询的失败误判成服务掉线。
            self._json({"ok": True, "ts": time.time()})
            return
        if path.startswith("/api/jobs"):
            # 最近 20 个任务的概要（不含体积较大的 log / summary），
            # 页面刷新后据此恢复"正在运行"的轮询，不会丢失界面状态。
            with _JOBS_LOCK:
                rows = sorted(_JOBS.values(),
                              key=lambda j: j.get("started_at") or 0,
                              reverse=True)[:20]
                items = [{k: row.get(k) for k in
                          ("id", "status", "started_at", "finished_at",
                           "elapsed", "params", "error")} for row in rows]
            self._json({"jobs": items})
            return
        if path.startswith("/api/job"):
            qs = parse_qs(parsed.query or "")
            job_id = (qs.get("id") or [""])[0]
            with _JOBS_LOCK:
                row = _JOBS.get(job_id)
                job = dict(row) if row is not None else None
            if job is None:
                # 服务重启会清空内存任务表；返回结构化 lost，
                # 前端可提示"请重新运行"，而不是无休止地等一个不存在的任务。
                self._json({"id": job_id, "status": "lost",
                            "error": "任务记录不存在（服务可能已重启），请重新运行"})
            else:
                self._json(job)
            return
        if path.startswith("/results/") or path.startswith("/web/"):
            self._static(path.lstrip("/"))
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_json()

        if path == "/api/key":
            api_key = str(body.get("api_key", "")).strip()
            base_url = str(body.get("base_url", "")).strip() or None
            model = str(body.get("model", "")).strip() or None
            remember = bool(body.get("remember", False))
            with _ADVISOR_LOCK:
                _ADVISOR.set_credentials(api_key, base_url, model,
                                         export_env=bool(api_key))
            self._json({
                "ok": True,
                "configured": bool(api_key),
                "masked": mask_key(api_key),
                "base_url": _ADVISOR.cfg.base_url,
                "model": _ADVISOR.cfg.model,
                "remember": remember,   # 是否记住由浏览器端 sessionStorage 决定
                "note": "Key 仅驻留服务端进程内存，不写入代码与结果文件",
            })
            return

        if path == "/api/key/test":
            with _ADVISOR_LOCK:
                ok, msg = _ADVISOR.test_connection()
            self._json({"ok": ok, "message": msg,
                        "masked": mask_key(_ADVISOR.cfg.api_key)})
            return

        if path == "/api/key/clear":
            with _ADVISOR_LOCK:
                _ADVISOR.clear_credentials()
            self._json({"ok": True, "configured": False})
            return

        if path == "/api/run":
            job_id = uuid.uuid4().hex[:8]
            with _JOBS_LOCK:
                _JOBS[job_id] = {"id": job_id, "status": "queued"}
            threading.Thread(target=_run_job, args=(job_id, body), daemon=True).start()
            self._json({"job_id": job_id})
            return

        self._json({"error": "unknown endpoint"}, 404)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="AI-MAPE-K 实验 Web 控制台")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = p.parse_args(argv)

    os.makedirs(WEB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(WEB_DIR, "index.html")):
        raise SystemExit(f"缺少页面文件：{os.path.join(WEB_DIR, 'index.html')}")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[webui] 控制台已启动: {url}")
    print(f"[webui] 结果目录: {RESULTS_DIR}")
    print(f"[webui] LLM Key 请在页面录入（或提前设置环境变量 {ENV_KEY}）")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] 已停止")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
