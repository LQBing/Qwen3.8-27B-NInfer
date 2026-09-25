# -*- coding: utf-8 -*-
"""_test_tools.py —— 验证 NInfer 对 opencode 式工具调用的支持（无需 --tool-call-parser）。"""
import json
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://localhost:30000"

TOOLS = [{
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
}]


def stream_chat(messages, tools=None, max_tokens=512, reasoning_effort=None):
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    content, reasoning, calls = [], [], {}
    finish = usage = None
    with requests.post(f"{BASE}/v1/chat/completions", json=payload, stream=True, timeout=600) as r:
        if r.status_code >= 400:
            return {"http": r.status_code, "err": r.text[:300]}
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
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    slot = calls.setdefault(i, {"id": None, "name": None, "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if obj.get("usage"):
                usage = obj["usage"]
    return {
        "finish": finish, "usage": usage,
        "content_len": len("".join(content)), "reasoning_len": len("".join(reasoning)),
        "tool_calls": [calls[k] for k in sorted(calls)],
    }


print("== A. 单轮流式工具调用（reasoning_effort=medium）==")
a = stream_chat([{"role": "user", "content": "请查询北京和上海现在的天气，调用 get_weather。"}], TOOLS, reasoning_effort="medium")
print(json.dumps(a, ensure_ascii=False, indent=1))

print("\n== B. 多轮：回传 tool 结果，期望给出最终文字回答 ==")
call_id = (a["tool_calls"][0]["id"] if a.get("tool_calls") else "call_x")
msgs = [
    {"role": "user", "content": "请查询北京现在的天气，调用 get_weather。"},
    {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"北京"}'},
    }]},
    {"role": "tool", "tool_call_id": call_id, "content": '{"city":"北京","temp_c":5,"cond":"晴"}'},
]
b = stream_chat(msgs, TOOLS, reasoning_effort="medium")
print(json.dumps({k: b[k] for k in ("finish", "content_len", "reasoning_len", "tool_calls")}, ensure_ascii=False, indent=1))

print("\n== C. tool_choice=auto 是否被接受 ==")
payload = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "查北京天气"}],
           "tools": TOOLS, "tool_choice": "auto", "max_tokens": 256, "temperature": 0.2}
r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=300)
print("HTTP", r.status_code, "| finish:", (r.json().get("choices") or [{}])[0].get("finish_reason") if r.status_code == 200 else r.text[:150])
