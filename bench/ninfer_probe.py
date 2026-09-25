# -*- coding: utf-8 -*-
"""ninfer_probe.py —— 验证 NInfer 优化配置下的并发能力与最长上下文。

用法:
  python ninfer_probe.py --mode longctx --tokens 250000
  python ninfer_probe.py --mode conc --n 3 --scenario short
  python ninfer_probe.py --mode conc --n 3 --scenario long --tokens 32000
"""
import argparse
import json
import statistics
import sys
import threading
import time
import uuid

import requests

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://localhost:30000"


def _discover_model():
    try:
        data = requests.get(f"{BASE}/v1/models", timeout=15).json().get("data") or []
        if data and data[0].get("id"):
            return data[0]["id"]
    except Exception:  # noqa: BLE001
        pass
    return "qwen3.8-27b"


MODEL = _discover_model()

CODE_TASK = (
    "你是一个资深 Python 工程师。请实现一个 LRU 缓存类，要求：\n"
    "1. 支持 get(key) / put(key, value) / peek(key) 三个方法\n"
    "2. 线程安全\n3. 时间复杂度 O(1)\n4. 用类型注解，写完整的 docstring\n"
    "请直接输出代码，不要解释。"
)


def _func(i):
    name = f"process_record_{i:04d}"
    n = (i * 37) % 30 + 8
    lines = [f"def {name}(records, config=None):", '    """Handle the data pipeline for chunk %d."""' % i]
    for j in range(n):
        k = (i * 31 + j * 17) % 1000
        if j % 5 == 0:
            lines.append(f"    # step {j}: normalize the {k}th field of each record")
        elif j % 5 == 1:
            lines.append(f"    acc_{j} = sum(x.get('value', 0) * {k} for x in records)")
        elif j % 5 == 2:
            lines.append(f"    if acc_{j} > {k}:")
            lines.append(f"        acc_{j} -= len(records)")
        elif j % 5 == 3:
            lines.append(f"    buf_{j} = [x for x in records if x.get('id') % {k + 1} == 0]")
        else:
            lines.append(f"    stats_{j} = {{'mean': acc_{j} / max(len(records), 1), 'bucket': {k}}}")
    lines.append(f"    return locals().get('stats_{n - 1}', {{'done': True}})")
    return "\n".join(lines) + "\n"


def build_body(min_chars=6_000_000):
    parts = ["### 项目代码库\n", "```python\n"]
    size = sum(len(p) for p in parts)
    m = 0
    while size < min_chars:
        h = f"\n# ===== module_{m} =====\n"
        parts.append(h); size += len(h)
        for i in range(m * 20, m * 20 + 20):
            fn = _func(i); parts.append(fn); size += len(fn)
        m += 1
    parts.append("```\n")
    return "".join(parts)


def probe_tokens(text):
    body = text + f"\n# salt {uuid.uuid4()}\n"
    r = requests.post(f"{BASE}/v1/chat/completions",
                      json={"model": MODEL, "messages": [{"role": "user", "content": body}],
                            "max_tokens": 1, "temperature": 0.0}, timeout=600)
    return r.json()["usage"]["prompt_tokens"], len(body)


def stream_once(prompt, max_tokens, temp=0.2):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temp, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); ttft = None; out = None; cache_n = None; err = None
    try:
        with requests.post(f"{BASE}/v1/chat/completions", json=payload, stream=True, timeout=1800) as r:
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                s = line.strip()
                if s.startswith("data: "):
                    s = s[6:]
                if s == "[DONE]":
                    break
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ttft is None and (d.get("content") or d.get("reasoning_content")):
                        ttft = time.perf_counter() - t0
                if obj.get("usage"):
                    out = obj["usage"].get("completion_tokens")
                if obj.get("timings"):
                    cache_n = obj["timings"].get("cache_n")
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    te = time.perf_counter()
    gen = (te - t0) - ttft if ttft else 0.0
    return {"ttft": ttft, "total": te - t0, "out": out,
            "decode": (out / gen) if (out and gen > 0) else None, "cache_n": cache_n, "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["longctx", "conc"], required=True)
    ap.add_argument("--tokens", type=int, default=32000)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scenario", choices=["short", "long"], default="short")
    a = ap.parse_args()

    if a.mode == "longctx":
        body = build_body()
        pt, nchar = probe_tokens(body[:120000])
        cpt = nchar / max(pt, 1)
        cut = min(int(a.tokens * cpt * 1.02), len(body))
        prompt = body[:cut]
        print(f"[info] 目标 {a.tokens} tok，chars/token={cpt:.3f}，实际切 {cut} 字符")
        m = stream_once(prompt + f"\n# salt {uuid.uuid4()}\n", 128)
        print(json.dumps(m, ensure_ascii=False, indent=1))
        return

    # conc
    if a.scenario == "short":
        prompts = [CODE_TASK] * a.n
        mt = 512
    else:
        body = build_body()
        pt, nchar = probe_tokens(body[:120000])
        cpt = nchar / max(pt, 1)
        cut = min(int(a.tokens * cpt * 1.02), len(body))
        # 不同起始位置 → 前缀不同（独立会话）
        prompts = [body[i * 200000 // max(a.n - 1, 1):][:cut] for i in range(a.n)] if a.n > 1 else [body[:cut]]
        mt = 256
    start = threading.Event()
    res = [None] * a.n

    def worker(idx):
        start.wait()
        res[idx] = stream_once(prompts[idx] + f"\n# salt {uuid.uuid4()}\n", mt)

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(a.n)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    start.set()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    print(f"[result] concurrency={a.n} scenario={a.scenario} wall={wall:.2f}s")
    for i, r in enumerate(res):
        print(" ", i, json.dumps(r, ensure_ascii=False))
    dec = [r["decode"] for r in res if r and r["decode"]]
    if dec:
        print(f"[summary] decode median={statistics.median(dec):.1f} 最慢={min(dec):.1f}")


if __name__ == "__main__":
    main()
