# -*- coding: utf-8 -*-
"""summarize.py —— 读取三个实例的 jsonl，输出对比表（Markdown + CSV）。

    python summarize.py --data-dir ..\\data --out ..\\data\\summary.md
"""
import argparse
import json
import os
import statistics

# 加载耗时 / 显存：来自各容器启动日志（见 docs/README.md 的"证据来源"）
META = {
    "sglang": {"engine": "SGLang", "weights": "NVFP4 LMHead4 + DSpark-NVFP4 drafter",
               "load_s": 157, "vram_mib": 30786, "ctx_max": 163840,
               "prompt_style": "reasoning_effort=medium"},
    "ninfer-nvfp4": {"engine": "NInfer", "weights": "qwen3_8_27b_nvfp4.ninfer",
                     "load_s": 122, "vram_mib": 30751, "ctx_max": 240000,
                     "prompt_style": "preserve-thinking (默认开思考)"},
    "ninfer-mtp": {"engine": "NInfer", "weights": "qwen3_8_27b.ninfer",
                   "load_s": 106, "vram_mib": 27905, "ctx_max": 240000,
                   "prompt_style": "preserve-thinking (默认开思考)"},
}
LABELS = ["sglang", "ninfer-nvfp4", "ninfer-mtp"]
SCENARIOS = [
    "short_coding_gen", "longctx_32k_prefill", "longctx_32k_longgen",
    "longctx_128k_prefill", "conc2_short", "conc2_long_distinct",
]


def load(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[r["scenario"]] = r
    return recs


def prefill_tps(rec):
    """prompt_tokens / ttft，跨引擎统一口径的 prefill 速度估计。"""
    if not rec or not rec.get("ttft_s_med"):
        return None
    pte = rec.get("prompt_tokens_est")
    if not pte:
        return None
    return pte / rec["ttft_s_med"]


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/summary.md")
    args = ap.parse_args()

    data = {lb: load(os.path.join(args.data_dir, f"{lb}.jsonl")) for lb in LABELS}

    lines = []
    lines.append("# 三实例实测汇总 — Qwen3.8-27B @ RTX 5090D 32G\n")
    lines.append("> 统一 harness（`bench/bench_unified.py`），每场景 3 轮取中位数；")
    lines.append("> 每轮 prompt 加唯一 salt 使前缀缓存失效（`cache_n=0` 已验证）。")
    lines.append("> TTFT 口径 = 第一个非空 token（含 reasoning_content）。\n")

    # 冷启动 / 显存
    lines.append("## 1. 冷启动与资源\n")
    lines.append("| 实例 | 引擎 | 权重 | 冷启动 | 启动后显存 | 上下文上限 | 思考策略 |")
    lines.append("|---|---|---|---|---|---|---|")
    for lb in LABELS:
        m = META[lb]
        lines.append(
            f"| `{lb}` | {m['engine']} | {m['weights']} | {m['load_s']}s | "
            f"{m['vram_mib']} MiB | {m['ctx_max']} | {m['prompt_style']} |"
        )
    lines.append("")

    # 指标表
    def metric_table(title, fn, nd=2):
        lines.append(f"## {title}\n")
        lines.append("| 场景 | " + " | ".join(LABELS) + " |")
        lines.append("|---|" + "---|" * len(LABELS))
        for sc in SCENARIOS:
            cells = []
            for lb in LABELS:
                rec = data[lb].get(sc)
                cells.append(fmt(fn(rec), nd))
            lines.append(f"| {sc} | " + " | ".join(cells) + " |")
        lines.append("")

    metric_table("2. TTFT (s，越低越好)", lambda r: r.get("ttft_s_med") if r else None, 3)
    metric_table("3. Prefill 速度 (prompt_tokens / TTFT，tok/s，越高越好)", prefill_tps, 0)
    metric_table("4. Decode 速度 (tok/s，越高越好)", lambda r: r.get("decode_tok_s_med") if r else None, 1)
    metric_table("5. 端到端总耗时 (s，越低越好)", lambda r: r.get("total_s_med") if r else None, 2)

    # 并发墙钟
    lines.append("## 6. 并发 2 墙钟 (wall_s) 与单流劣化\n")
    lines.append("| 场景 | 指标 | " + " | ".join(LABELS) + " |")
    lines.append("|---|---|" + "---|" * len(LABELS))
    for sc in ["conc2_short", "conc2_long_distinct"]:
        wall_row = []
        slow_row = []
        for lb in LABELS:
            rec = data[lb].get(sc)
            wall_row.append(fmt(rec.get("wall_s") if rec else None, 2))
            # 本轮各请求 decode 的最小值（慢流）
            mn = None
            if rec:
                vals = []
                for rd in rec.get("rounds", []):
                    for q in rd.get("requests", []) or []:
                        if q.get("decode_tok_s"):
                            vals.append(q["decode_tok_s"])
                mn = min(vals) if vals else None
            slow_row.append(fmt(mn, 1))
        lines.append(f"| {sc} | wall_s | " + " | ".join(wall_row) + " |")
        lines.append(f"| {sc} | 最慢请求 decode | " + " | ".join(slow_row) + " |")
    lines.append("")

    text = "\n".join(lines)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[info] wrote {out_path}")

    # 同步生成 CSV
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scenario,label,ttft_s,prefill_tps,decode_tps,total_s,wall_s\n")
        for sc in SCENARIOS:
            for lb in LABELS:
                rec = data[lb].get(sc)
                if not rec:
                    continue
                f.write(
                    f"{sc},{lb},{fmt(rec.get('ttft_s_med'),4)},"
                    f"{fmt(prefill_tps(rec),1)},{fmt(rec.get('decode_tok_s_med'),1)},"
                    f"{fmt(rec.get('total_s_med'),3)},{fmt(rec.get('wall_s'),3)}\n"
                )
    print(f"[info] wrote {csv_path}")


if __name__ == "__main__":
    main()
