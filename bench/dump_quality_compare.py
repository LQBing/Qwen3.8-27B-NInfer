# -*- coding: utf-8 -*-
"""dump_quality_compare.py —— 把两份 quality_probe 结果解码为可读的并排 Markdown。

JSON 里 content/reasoning 是含 \\n 的单行巨串，直接 read 会被 2000 字符/行截断。
本脚本把它们还原成真实换行后再输出，供人工/agent 审查。

用法: python bench\\dump_quality_compare.py
输出: quality\\compare-side-by-side.md
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [
    ("official", os.path.join(ROOT, "quality", "ninfer-official.json")),
    ("abliterated", os.path.join(ROOT, "quality", "ninfer-abliterated.json")),
]
OUT = os.path.join(ROOT, "quality", "compare-side-by-side.md")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fence(text, lang=""):
    if not text:
        return "_(empty)_\n"
    return f"```{lang}\n{text}\n```\n"


def main():
    data = {name: load(p) for name, p in PAIRS}
    order = [r["id"] for r in data["official"]["results"]]

    lines = ["# quality_probe 并排对照 — official vs abliterated\n",
             "> 同一 prompt、同一采样参数（temp 0.2）、同一服务配置；仅权重血统不同。\n"]

    for tid in order:
        lines.append(f"\n---\n\n## 任务 `{tid}`\n")
        recs = {}
        for name, _ in PAIRS:
            recs[name] = next((r for r in data[name]["results"] if r["id"] == tid), None)
        # 元信息表
        lines.append("| 指标 | official | abliterated |")
        lines.append("|---|---|---|")
        for key, label in [
            ("finish_reason", "finish_reason"),
            ("total_s", "total_s(s)"),
            ("ttft_s", "ttft_s(s)"),
        ]:
            vals = []
            for name, _ in PAIRS:
                r = recs[name] or {}
                v = r.get(key)
                vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
        for name in ("content", "reasoning"):
            vals = [len((recs[n] or {}).get(name) or "") for n, _ in PAIRS]
            lines.append(f"| {name}_chars | {vals[0]} | {vals[1]} |")
        for name in ("tool_calls",):
            lines.append(
                f"| tool_calls | {len((recs.get('official') or {}).get(name) or [])} | "
                f"{len((recs.get('abliterated') or {}).get(name) or [])} |"
            )

        for name, _ in PAIRS:
            r = recs[name] or {}
            lines.append(f"\n### {name} — content\n")
            lines.append(fence(r.get("content") or ""))
            if r.get("tool_calls"):
                lines.append(f"\n### {name} — tool_calls\n")
                lines.append(fence(json.dumps(r["tool_calls"], ensure_ascii=False, indent=2), "json"))
            lines.append(f"\n### {name} — reasoning\n")
            lines.append(fence(r.get("reasoning") or ""))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[info] wrote {OUT}")


if __name__ == "__main__":
    main()
