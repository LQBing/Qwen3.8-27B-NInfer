# -*- coding: utf-8 -*-
"""临时：测试 NInfer 的 thinking 控制是否能避免"只思考不产出"。"""
import json
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

VARIANTS = [
    ("default", {}),
    ("enable_thinking=false", {"enable_thinking": False}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
]


def run(name, extra):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    payload.update(extra)
    try:
        r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=300)
        if r.status_code >= 400:
            print(f"{name:52s} HTTP {r.status_code}: {r.text[:160]}")
            return
        obj = r.json()
        ch = (obj.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = obj.get("usage") or {}
        print(
            f"{name:52s} finish={str(ch.get('finish_reason')):10s} "
            f"content_chars={len(content):5d} reasoning_chars={len(reasoning):5d} "
            f"completion_tokens={usage.get('completion_tokens')}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"{name:52s} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    for name, extra in VARIANTS:
        run(name, extra)
