# -*- coding: utf-8 -*-
"""stability.py —— 基于逐轮数据做稳定性分析（中位数 / 极值 / 标准差 / 变异系数 CV%）。

    python stability.py --data-dir ..\\data --out ..\\data\\stability.md
"""
import argparse
import json
import os
import statistics

SCEN = [
    "short_coding_gen", "longctx_32k_prefill", "longctx_32k_longgen",
    "longctx_128k_prefill", "conc2_short", "conc2_long_distinct",
]
LABELS = ["sglang", "ninfer-nvfp4", "ninfer-mtp", "ninfer-nvfp4-native", "ninfer-mtp-native"]


def load(path):
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["scenario"]] = r
    return recs


def col(rec, key):
    """取该场景逐轮某指标的原始值列表（跳过 None）。"""
    vals = []
    for rd in (rec or {}).get("rounds", []):
        v = rd.get(key)
        if v is not None:
            vals.append(float(v))
    return vals


def prefill_rounds(rec):
    """逐轮 prefill 速度 = prompt_tokens / ttft_s。"""
    out = []
    for rd in (rec or {}).get("rounds", []):
        pt, tt = rd.get("prompt_tokens"), rd.get("ttft_s")
        if pt and tt:
            out.append(pt / tt)
    return out


def stat(vals):
    """返回 (n, median, min, max, stdev, cv%)；样本不足返回 None。"""
    if len(vals) < 2:
        return None
    med = statistics.median(vals)
    sd = statistics.stdev(vals)
    cv = (sd / med * 100) if med else None
    return len(vals), med, min(vals), max(vals), sd, cv


def fmt(s, nd=2):
    if not s:
        return "-"
    n, med, lo, hi, sd, cv = s
    return f"{med:.{nd}f} [{lo:.{nd}f}–{hi:.{nd}f}] ±{sd:.{nd}f} CV={cv:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/stability.md")
    a = ap.parse_args()

    data = {lb: load(os.path.join(a.data_dir, f"{lb}.jsonl")) for lb in LABELS}

    L = []
    L.append("# 稳定性分析（每场景 5 轮）\n")
    L.append("> 指标格式：`中位数 [最小–最大] ±标准差 CV=变异系数%`。")
    L.append("> CV（标准差/中位数）越低越稳定；>5% 视为波动明显。\n")

    for sc in SCEN:
        L.append(f"## {sc}\n")
        L.append("| 方案 | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | 总耗时 (s) |")
        L.append("|---|---|---|---|---|")
        for lb in LABELS:
            rec = data[lb].get(sc)
            L.append(
                f"| {lb} | {fmt(stat(col(rec,'ttft_s')),3)} | {fmt(stat(prefill_rounds(rec)),0)} "
                f"| {fmt(stat(col(rec,'decode_tok_s')),1)} | {fmt(stat(col(rec,'total_s')),2)} |"
            )
        L.append("")

    # 汇总：每个方案的 CV 概况（取关键指标的中位 CV）
    L.append("## 汇总：波动概况（各方案跨场景 CV% 的中位数）\n")
    L.append("| 方案 | TTFT CV% | Prefill CV% | Decode CV% | 总耗时 CV% |")
    L.append("|---|---|---|---|---|")
    for lb in LABELS:
        cvs = {"ttft": [], "pre": [], "dec": [], "tot": []}
        for sc in SCEN:
            rec = data[lb].get(sc)
            for key, kk in [("ttft", col(rec, "ttft_s")), ("dec", col(rec, "decode_tok_s")),
                            ("tot", col(rec, "total_s")), ("pre", prefill_rounds(rec))]:
                s = stat(kk)
                if s and s[5] is not None:
                    cvs[key].append(s[5])
        med = lambda xs: f"{statistics.median(xs):.1f}%" if xs else "-"
        L.append(f"| {lb} | {med(cvs['ttft'])} | {med(cvs['pre'])} | {med(cvs['dec'])} | {med(cvs['tot'])} |")
    L.append("")

    path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"[info] wrote {path}")


if __name__ == "__main__":
    main()
