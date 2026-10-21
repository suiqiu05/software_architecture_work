"""严格 JSON 序列化工具。

为什么需要：Python 的 ``json`` 默认把 ``float('nan')`` / ``inf`` 输出成
``NaN`` / ``Infinity`` 字面量，而这不是合法 JSON。JS 的 ``JSON.parse``
遇到 ``NaN`` 会整段失败 —— 在 Web 控制台里表现为"接口 HTTP 200，但响应
不是 JSON"，前端会误判为服务端掉线。

本项目的 Static / HPA 两组控制器没有预测能力，RMSE / MAPE 天然是 NaN，
所以所有对外（HTTP 与落盘）的 JSON 都必须先经过 :func:`json_safe`。
"""
from __future__ import annotations

import json
import math
from typing import Any


def json_safe(obj: Any) -> Any:
    """递归清洗：NaN/Inf → None；numpy 标量 → Python 标量；dict 键 → str。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:                      # numpy 标量（np.float64 / np.int64 等）
            return json_safe(obj.item())
        except (AttributeError, ValueError, TypeError):
            return str(obj)
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    """严格 JSON 文本（不会出现 NaN/Infinity）。"""
    return json.dumps(json_safe(obj), ensure_ascii=False, **kwargs)


def dump(obj: Any, fp: Any, **kwargs: Any) -> None:
    """写文件，语义同 ``json.dump``，但保证输出合法 JSON。"""
    json.dump(json_safe(obj), fp, ensure_ascii=False, **kwargs)
