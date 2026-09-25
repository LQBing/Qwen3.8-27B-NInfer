# -*- coding: utf-8 -*-
"""quality_probe.py —— 三实例"效果/质量"抽查。

同一批 coding 任务分别打到当前服务，保存 content / reasoning / tool_calls
原文与 finish_reason，供人工或离线比对正确性、工具调用与思考链行为。

用法:
    python quality_probe.py --label sglang --out ..\\quality\\sglang.json
"""
import argparse
import json
import os
import sys
import time

import requests

DEFAULT_BASE = "http://localhost:30000"

# ---------------------------------------------------------------- 测试任务
# 覆盖：代码生成 / 缺陷修复 / 工具调用 / 复杂多步任务
PROMPTS = [
    {
        "id": "lru",
        "cat": "code-gen",
        "max_tokens": 1024,
        "prompt": (
            "你是一个资深 Python 工程师。请实现一个 LRU 缓存类，要求：\n"
            "1. 支持 get(key) / put(key, value) / peek(key) 三个方法\n"
            "2. 线程安全\n3. 时间复杂度 O(1)\n4. 用类型注解，写完整的 docstring\n"
            "请直接输出代码，不要解释。"
        ),
    },
    {
        "id": "bugfix",
        "cat": "bugfix",
        "max_tokens": 1024,
        "prompt": (
            "下面这段 Python 有并发缺陷，请指出根因并用最小改动修复，给出修复后的完整代码：\n\n"
            "```python\n"
            "import threading\n\n"
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.n = 0\n"
            "    def incr(self):\n"
            "        cur = self.n\n"
            "        cur += 1\n"
            "        self.n = cur\n"
            "```\n"
            "要求：支持多线程安全自增，并说明为什么原实现不安全。"
        ),
    },
    {
        "id": "toolcall",
        "cat": "tool-use",
        "max_tokens": 512,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City name"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "prompt": "请查询北京和上海现在的天气。请调用 get_weather 工具。",
    },
    {
        "id": "design",
        "cat": "complex",
        "max_tokens": 1536,
        "prompt": (
            "设计并实现一个线程安全的令牌桶限流器（token bucket rate limiter），要求：\n"
            "1. 支持设置容量与补充速率\n2. try_acquire(n) 非阻塞，成功返回 True\n"
            "3. 支持按 key 维度隔离（多租户）\n4. 提供单元测试\n"
            "请给出完整可运行代码。"
        ),
    },
]


def discover_model(base, explicit):
    try:
        data = requests.get(f"{base}/v1/models", timeout=15).json().get("data") or []
        if data and data[0].get("id"):
            return data[0]["id"]
    except Exception as e:  # noqa: BLE001
        print(f"[info] model discovery failed: {e}", file=sys.stderr)
    return explicit or "default"


def run_prompt(base, model, spec):
    """流式请求，累计 content / reasoning_content / tool_calls。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": spec["prompt"]}],
        "max_tokens": spec.get("max_tokens", 1024),
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if spec.get("tools"):
        payload["tools"] = spec["tools"]

    t0 = time.perf_counter()
    content_parts = []
    reasoning_parts = []
    tool_calls_acc = {}
    finish_reason = None
    usage = None
    engine_timings = None
    err = None
    t_first = None
    try:
        with requests.post(f"{base}/v1/chat/completions", json=payload, stream=True, timeout=900) as r:
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
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
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        if t_first is None:
                            t_first = time.perf_counter() - t0
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        if t_first is None:
                            t_first = time.perf_counter() - t0
                        reasoning_parts.append(delta["reasoning_content"])
                    if delta.get("tool_calls"):
                        # 流式下同一 tool_call 会被拆成多个 delta（带 index），需按 index 合并
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            slot = tool_calls_acc.setdefault(
                                idx,
                                {"id": None, "type": "function",
                                 "function": {"name": None, "arguments": ""}},
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("timings"):
                    engine_timings = obj["timings"]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    total_s = time.perf_counter() - t0
    tool_calls = [tool_calls_acc[k] for k in sorted(tool_calls_acc)]

    return {
        "id": spec["id"],
        "cat": spec["cat"],
        "finish_reason": finish_reason,
        "usage": usage,
        "engine_timings": engine_timings,
        "total_s": total_s,
        "ttft_s": t_first,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "error": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    model = discover_model(base, args.model)
    print(f"[info] model={model}", file=sys.stderr)

    results = []
    for spec in PROMPTS:
        print(f"[info] running {spec['id']} ({spec['cat']})...", file=sys.stderr)
        res = run_prompt(base, model, spec)
        results.append(res)
        print(
            f"[info]   finish={res['finish_reason']} total={res['total_s']:.1f}s "
            f"content_chars={len(res['content'])} reasoning_chars={len(res['reasoning'])} "
            f"tool_calls={len(res['tool_calls'])} err={res['error']}",
            file=sys.stderr,
        )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"label": args.label, "model": model, "results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[info] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
