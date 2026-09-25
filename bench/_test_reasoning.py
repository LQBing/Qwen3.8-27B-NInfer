# -*- coding: utf-8 -*-
"""_test_reasoning.py —— NInfer 各 reasoning_effort 档位的思考/正文分配（对齐 SGLang medium）。

同 SGLang 质量抽查用的 LRU 提示词、max_tokens=1024，便于直接对照。
"""
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://localhost:30000"

PROMPT = (
    "你是一个资深 Python 工程师。请实现一个 LRU 缓存类，要求：\n"
    "1. 支持 get(key) / put(key, value) / peek(key) 三个方法\n"
    "2. 线程安全\n3. 时间复杂度 O(1)\n4. 用类型注解，写完整的 docstring\n"
    "请直接输出代码，不要解释。"
)

EFFORTS = [None, "none", "minimal", "low", "medium", "high"]


def run(effort, max_tokens=1024):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if effort is not None:
        payload["reasoning_effort"] = effort
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=600)
        dt = time.perf_counter() - t0
        if r.status_code >= 400:
            print(f"{str(effort):8s} HTTP {r.status_code}: {r.text[:120]}")
            return
        obj = r.json()
        ch = (obj.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = obj.get("usage") or {}
        print(
            f"effort={str(effort):8s} finish={str(ch.get('finish_reason')):8s} "
            f"content={len(content):5d} reasoning={len(reasoning):5d} "
            f"tokens={usage.get('completion_tokens'):5d} 用时={dt:5.1f}s"
        )
    except Exception as e:  # noqa: BLE001
        print(f"{str(effort):8s} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("== max_tokens=1024（对齐 SGLang 抽样口径）==")
    for e in EFFORTS:
        run(e, 1024)
    print("\n== max_tokens=4096（给足预算）==")
    for e in EFFORTS:
        run(e, 4096)
