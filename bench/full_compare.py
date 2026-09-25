# -*- coding: utf-8 -*-
"""full_compare.py —— 全部 5 次运行的完整数据对比（Markdown + CSV）。

运行：3 个 Docker 实例 + 2 个 WSL 原生实例，全部覆盖 6 个场景 × 3 轮。

    python full_compare.py --data-dir ..\\data --out ..\\data\\full-comparison.md
"""
import argparse
import json
import os
import statistics

SCENARIOS = [
    "short_coding_gen", "longctx_32k_prefill", "longctx_32k_longgen",
    "longctx_128k_prefill", "conc2_short", "conc2_long_distinct",
]
SHORT_SCENARIOS = {"short_coding_gen", "conc2_short"}  # prefill 受固定开销主导
CONC_SCENARIOS = {"conc2_short", "conc2_long_distinct"}

# 运行清单：冷启动数据来自各次容器/进程启动日志
RUNS = [
    {"id": "sglang",              "short": "SGLang/docker",  "run": "Docker",   "engine": "SGLang",
     "weights": "NVFP4-LMHead4 + DSpark-NVFP4", "load_s": 157.0, "w_s": None, "io": None, "vram": 30786, "ctx": 163840, "spec": "DSPARK(block7)"},
    {"id": "ninfer-nvfp4",        "short": "NVFP4/docker",   "run": "Docker",   "engine": "NInfer",
     "weights": "qwen3_8_27b_nvfp4.ninfer", "load_s": 121.6, "w_s": 111.1, "io": 181.8, "vram": 30751, "ctx": 240000, "spec": "MTP(3)"},
    {"id": "ninfer-mtp",          "short": "GWint/docker",   "run": "Docker",   "engine": "NInfer",
     "weights": "qwen3_8_27b.ninfer", "load_s": 105.8, "w_s": 92.8, "io": 183.9, "vram": 27905, "ctx": 240000, "spec": "MTP(3)"},
    {"id": "ninfer-nvfp4-native", "short": "NVFP4/native",   "run": "WSL原生",  "engine": "NInfer",
     "weights": "qwen3_8_27b_nvfp4.ninfer", "load_s": 109.1, "w_s": 100.4, "io": 201.3, "vram": 30828, "ctx": 240000, "spec": "MTP(3)"},
    {"id": "ninfer-mtp-native",   "short": "GWint/native",   "run": "WSL原生",  "engine": "NInfer",
     "weights": "qwen3_8_27b.ninfer", "load_s": 102.4, "w_s": 92.0, "io": 185.5, "vram": None, "ctx": 240000, "spec": "MTP(3)"},
]
IDS = [r["id"] for r in RUNS]
SHORT = {r["id"]: r["short"] for r in RUNS}


def load_recs(path):
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["scenario"]] = r
    return recs


def prefill_tps(rec):
    if not rec or not rec.get("ttft_s_med") or not rec.get("prompt_tokens_est"):
        return None
    return rec["prompt_tokens_est"] / rec["ttft_s_med"]


def accept_rate(rec):
    if not rec:
        return None
    a = dd = 0
    for rd in rec.get("rounds", []):
        t = rd.get("engine_timings") or {}
        a += t.get("draft_n_accepted", 0)
        dd += t.get("draft_n", 0)
    return (a / dd) if dd else None


def slowest_decode(rec):
    """并发场景：本轮所有请求里最慢的 decode。"""
    if not rec:
        return None
    vals = []
    for rd in rec.get("rounds", []):
        for q in rd.get("requests", []) or []:
            if q.get("decode_tok_s"):
                vals.append(q["decode_tok_s"])
    return min(vals) if vals else None


def fm(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def best_mark(vals, higher_better):
    """返回"最优"下标集合；None 视为不参与。"""
    real = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not real:
        return set()
    target = (max if higher_better else min)(v for _, v in real)
    return {i for i, v in real if abs(v - target) < 1e-9}


def metric_table(data, title, key_fn, higher_better, nd, annotate_short=False):
    rows = []
    marks = {sc: best_mark([key_fn(data[i].get(sc)) for i in IDS], higher_better) for sc in SCENARIOS}
    out = [f"### {title}\n"]
    out.append("| 场景 | " + " | ".join(SHORT[i] for i in IDS) + " |")
    out.append("|---|" + "---|" * len(IDS))
    for sc in SCENARIOS:
        cells = []
        for idx, i in enumerate(IDS):
            v = key_fn(data[i].get(sc))
            s = fm(v, nd)
            if annotate_short and sc in SHORT_SCENARIOS and key_fn is prefill_tps:
                s = f"({s})"
            if idx in marks[sc]:
                s = f"**{s}**"
            cells.append(s)
        tag = " *(短，不可比)*" if (annotate_short and sc in SHORT_SCENARIOS) else ""
        out.append(f"| {sc}{tag} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/full-comparison.md")
    a = ap.parse_args()

    data = {i: load_recs(os.path.join(a.data_dir, f"{i}.jsonl")) for i in IDS}

    L = []
    L.append("# 全量数据对比 — Qwen3.8-27B 五个实例 @ RTX 5090D 32G\n")
    L.append("> 5 次运行 = 3 个 Docker 实例（SGLang / NInfer-NVFP4 / NInfer-groupwise-int）")
    L.append("> + 2 个 WSL 原生实例（NInfer-NVFP4 / NInfer-groupwise-int）。")
    L.append("> 统一 harness、每场景 3 轮中位数、每轮 salt 失效前缀缓存（`cache_n=0`）。")
    L.append("> TTFT = 第一个非空 token（含 `reasoning_content`）。**加粗 = 该行最优**。")
    L.append("> 逐轮原始数据见 `data/all-runs.csv`。\n")

    # 0. 覆盖度自检
    L.append("## 0. 数据覆盖度自检\n")
    L.append("| 运行 | " + " | ".join(SCENARIOS) + " |")
    L.append("|---|" + "---|" * len(SCENARIOS))
    for i in IDS:
        cells = ["✓" if sc in data[i] else "✗缺" for sc in SCENARIOS]
        L.append(f"| {SHORT[i]} | " + " | ".join(cells) + " |")
    L.append("")

    # 1. 运行清单
    L.append("## 1. 运行清单（环境 / 冷启动 / 资源）\n")
    L.append("| 运行 | 引擎 | 运行方式 | 权重 | 投机 | 权重加载 | 读取速率 | 总启动 | 显存 | 上下文上限 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in RUNS:
        L.append(
            f"| {r['short']} | {r['engine']} | {r['run']} | {r['weights']} | {r['spec']} | "
            f"{fm(r['w_s'],1) if r['w_s'] else '—'}s | {fm(r['io'],1) if r['io'] else '—'} MiB/s | "
            f"{r['load_s']:.1f}s | {r['vram'] if r['vram'] else '—'} MiB | {r['ctx']} |"
        )
    L.append("")

    # 2-5. 指标表
    L.append("## 2. 逐场景指标对比\n")
    L += metric_table(data, "2.1 TTFT（s，越低越好）", lambda r: r.get("ttft_s_med") if r else None, False, 3)
    L += metric_table(data, "2.2 Prefill 速度（tok/s，越高越好）", prefill_tps, True, 0, annotate_short=True)
    L.append("> `short_coding_gen` / `conc2_short` 的 prefill 受固定开销主导（括号值仅供参照），判断 prefill 只看 `longctx_*`。\n")
    L += metric_table(data, "2.3 Decode 速度（tok/s，越高越好）", lambda r: r.get("decode_tok_s_med") if r else None, True, 1)
    L += metric_table(data, "2.4 端到端总耗时（s，越低越好）", lambda r: r.get("total_s_med") if r else None, False, 2)

    # 6. 并发
    L.append("## 3. 并发 2 表现\n")
    L.append("| 场景 | 指标 | " + " | ".join(SHORT[i] for i in IDS) + " |")
    L.append("|---|" + "---|" * (len(IDS) + 1))
    for sc in ["conc2_short", "conc2_long_distinct"]:
        wall = [data[i].get(sc, {}).get("wall_s") if data[i].get(sc) else None for i in IDS]
        slow = [slowest_decode(data[i].get(sc)) for i in IDS]
        wm = best_mark(wall, False)
        sm = best_mark(slow, True)
        wcells = [f"**{fm(v,2)}**" if k in wm else fm(v, 2) for k, v in enumerate(wall)]
        scells = [f"**{fm(v,1)}**" if k in sm else fm(v, 1) for k, v in enumerate(slow)]
        L.append(f"| {sc} | 墙钟 wall_s | " + " | ".join(wcells) + " |")
        L.append(f"| {sc} | 最慢请求 decode | " + " | ".join(scells) + " |")
    L.append("")

    # 7. 接受率
    L.append("## 4. MTP 接受率（Σdraft_n_accepted / Σdraft_n，原始 NInfer timings；SGLang 无此字段）\n")
    L.append("| 场景 | " + " | ".join(SHORT[i] for i in IDS) + " |")
    L.append("|---|" + "---|" * len(IDS))
    for sc in SCENARIOS:
        cells = []
        for i in IDS:
            ar = accept_rate(data[i].get(sc))
            cells.append(f"{ar*100:.1f}%" if ar else "—")
        L.append(f"| {sc} | " + " | ".join(cells) + " |")
    L.append("")

    # 8. 质量
    qdir = os.path.join(a.data_dir, "..", "quality")
    L.append("## 5. 质量抽查（同 prompt，原始输出见 `quality/*.json`）\n")
    L.append("| 任务 | 指标 | SGLang | NVFP4/docker | GWint/docker |")
    L.append("|---|---|---|---|---|")
    qfiles = ["sglang", "ninfer-nvfp4", "ninfer-mtp"]
    qdata = {}
    for name in qfiles:
        p = os.path.join(qdir, f"{name}.json")
        if os.path.exists(p):
            qdata[name] = {r["id"]: r for r in json.load(open(p, encoding="utf-8"))["results"]}
    for tid in ["lru", "bugfix", "toolcall", "design"]:
        for label, fn in [("finish", lambda r: r["finish_reason"]),
                          ("content 字数", lambda r: len(r["content"])),
                          ("reasoning 字数", lambda r: len(r["reasoning"])),
                          ("tool_calls", lambda r: len(r["tool_calls"]))]:
            cells = []
            for name in qfiles:
                r = qdata.get(name, {}).get(tid)
                cells.append(str(fn(r)) if r else "—")
            L.append(f"| {tid} | {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("> 关键：两个 NInfer 实例在 `lru`/`design` 上 **content=0**（思考吃满预算，默认无界思考）；")
    L.append("> SGLang 因 `reasoning_effort=medium` 正常产出正文。详见 `quality/quality.md`。\n")

    # 9. 要点
    L.append("## 6. 全量数据要点\n")
    L.append("1. **短输出/S 级差异**：所有实例短 prompt decode 都在 118–162 tok/s 区间，彼此接近。")
    L.append("2. **Prefill 决定长上下文体验**：SGLang 8110 / NVFP4 7202 / groupwise 2881（32K，tok/s）；")
    L.append("   128K 时 SGLang 3686 ≈ NVFP4 3843，而 groupwise 仅 2124 → **NInfer 必须用 nvfp4 权重**。")
    L.append("3. **64K→128K 的 TTFT 成本**：SGLang 4.0s→35.4s、NVFP4 4.5s→34.0s、groupwise 11.3s→61.5s。")
    L.append("4. **并发 2 独立长会话都会被 prefill 拖慢**：最慢流 SGLang 44 / NVFP4 38 / groupwise 19 tok/s；")
    L.append("   墙钟 SGLang 9.9s < NVFP4 11.6s << groupwise 25.5s。")
    L.append("5. **Docker vs 原生：等价**（见 `native-vs-docker.md`）：prefill 差异 ≤±6% 且方向不一致；")
    L.append("   decode 波动来自 MTP 接受率（表中第 4 节），不是环境。")
    L.append("6. **上下文上限**：NInfer 240,000 > SGLang 163,840。")
    L.append("7. **质量可用性**：默认配置下 SGLang 可产出正文，NInfer 需先修思考预算。\n")

    text = "\n".join(L)
    out_path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[info] wrote {out_path}")

    # 逐轮原始 CSV
    csv_path = os.path.join(os.path.dirname(out_path), "all-runs.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("run,scenario,round,ttft_s,prefill_tps,decode_tps,total_s,completion_tokens,accept_rate\n")
        for i in IDS:
            for sc in SCENARIOS:
                rec = data[i].get(sc)
                if not rec:
                    continue
                for rd in rec.get("rounds", []):
                    t = rd.get("engine_timings") or {}
                    ar = (t.get("draft_n_accepted", 0) / t["draft_n"]) if t.get("draft_n") else None
                    ttft = rd.get("ttft_s")
                    pte = rec.get("prompt_tokens_est")
                    pf = (pte / ttft) if (pte and ttft) else None
                    fh.write(
                        f"{i},{sc},{rd.get('round')},{fm(ttft,4)},{fm(pf,1)},"
                        f"{fm(rd.get('decode_tok_s'),1)},{fm(rd.get('total_s'),3)},"
                        f"{rd.get('completion_tokens') or ''},{fm(ar,4) if ar else ''}\n"
                    )
    print(f"[info] wrote {csv_path}")


if __name__ == "__main__":
    main()
