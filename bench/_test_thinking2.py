# -*- coding: utf-8 -*-
"""_test_thinking2.py —— 验证 NInfer 的思考预算修复（服务端 --default-thinking-budget + 客户端参数）。"""
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://localhost:30000"

PROMPT = (
    "你是一个资深 Python 工程师。请实现一个 LRU 缓存类，要求：\n"
    "1. 支持 get(key) / put(key, value) / peek(key) 三个方法\n"
    "2. 线程安全\n3. 时间复杂度 O(1)\n4. 用类型注解，写完整的 docstring\n"
    "请直接输出代码，不要解释。"
)

CASES = [
    ("default @1024", 1024, {}),
    ("default @4096", 4096, {}),
    ("reasoning_effort=medium @1024", 1024, {"reasoning_effort": "medium"}),
    ("reasoning_effort=none @1024", 1024, {"reasoning_effort": "none"}),
    ("enable_thinking=false @1024", 1024, {"enable_thinking": False}),
]


def run(name, max_tokens, extra):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    payload.update(extra)
    try:
        r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=600)
        if r.status_code >= 400:
            print(f"{name:34s} HTTP {r.status_code}: {r.text[:150]}")
            return
        obj = r.json()
        ch = (obj.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = obj.get("usage") or {}
        print(
            f"{name:34s} finish={str(ch.get('finish_reason')):10s} "
            f"content={len(content):5d} reasoning={len(reasoning):5d} "
            f"completion_tokens={usage.get('completion_tokens')}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"{name:34s} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    for name, mt, extra in CASES:
        run(name, mt, extra)
