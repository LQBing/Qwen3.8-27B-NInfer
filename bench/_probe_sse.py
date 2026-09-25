# -*- coding: utf-8 -*-
"""临时探针：打印 NInfer/SGLang 的原始 SSE chunk 序列与到达时间，用于定位 TTFT 口径差异。"""
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "用一句话解释什么是 LRU 缓存"

# 自动发现模型名
if not MODEL:
    try:
        MODEL = requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]
    except Exception:
        MODEL = "default"
print(f"[probe] base={BASE} model={MODEL}", file=sys.stderr)

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 30,
    "temperature": 0.0,
    "stream": True,
    "stream_options": {"include_usage": True},
}
t0 = time.perf_counter()
n = 0
with requests.post(f"{BASE}/v1/chat/completions", json=payload, stream=True, timeout=120) as r:
    print(f"[probe] HTTP {r.status_code}", file=sys.stderr)
    for line in r.iter_lines(decode_unicode=True):
        if line is None:
            continue
        s = line.strip()
        if not s:
            continue
        dt = time.perf_counter() - t0
        print(f"{dt:7.3f}s | {s[:220]}")
        n += 1
        if n > 30:
            break
print(f"[probe] total chunks shown={n}", file=sys.stderr)
