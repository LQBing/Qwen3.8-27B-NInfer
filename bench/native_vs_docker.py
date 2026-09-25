# -*- coding: utf-8 -*-
"""native_vs_docker.py —— 对比同一 ninfer 的 Docker 运行 vs WSL 原生运行。

    python native_vs_docker.py --data-dir ..\\data --out ..\\data\\native-vs-docker.md
"""
import argparse
import json
import os

SCENARIOS = [
    "short_coding_gen", "longctx_32k_prefill", "longctx_32k_longgen",
    "longctx_128k_prefill", "conc2_short", "conc2_long_distinct",
]

# 冷启动数据：来自各次启动日志（weights ready / engine ready 行）
LOAD = {
    ("nvfp4", "docker"): {"w_gib": 19.7, "w_s": 111.1, "mbps": 181.8, "total_s": 121.6, "vram": 30751},
    ("nvfp4", "native"): {"w_gib": 19.7, "w_s": 100.4, "mbps": 201.3, "total_s": 109.1, "vram": 30828},
    ("mtp", "docker"):   {"w_gib": 16.7, "w_s": 92.8,  "mbps": 183.9, "total_s": 105.8, "vram": 27905},
    ("mtp", "native"):   {"w_gib": 16.7, "w_s": 92.0,  "mbps": 185.5, "total_s": 102.4, "vram": None},
}
PAIRS = {
    "NVFP4 权重（qwen3_8_27b_nvfp4.ninfer）": {
        "docker": "ninfer-nvfp4", "native": "ninfer-nvfp4-native", "key": "nvfp4",
    },
    "groupwise-int 权重（qwen3_8_27b.ninfer）": {
        "docker": "ninfer-mtp", "native": "ninfer-mtp-native", "key": "mtp",
    },
}


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


def prefill_tps(rec):
    if not rec or not rec.get("ttft_s_med") or not rec.get("prompt_tokens_est"):
        return None
    return rec["prompt_tokens_est"] / rec["ttft_s_med"]


def accept_rate(rec):
    """MTP 接受率 = Σdraft_n_accepted / Σdraft_n（来自 NInfer 原生 timings）。"""
    if not rec:
        return None
    a = d = 0
    for rd in rec.get("rounds", []):
        t = rd.get("engine_timings") or {}
        a += t.get("draft_n_accepted", 0)
        d += t.get("draft_n", 0)
    return (a / d) if d else None


def f(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def pct(nat, doc):
    if not nat or not doc:
        return "-"
    return f"{(nat - doc) / doc * 100:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/native-vs-docker.md")
    a = ap.parse_args()

    L = []
    L.append("# Docker vs WSL 原生 —— 同一 ninfer 的性能对比\n")
    L.append("> 用途：回答「改用 WSL 原生启动（而非 Docker）是否带来性能差异」。")
    L.append("> 两个环境使用**同一份本地编译二进制** `C:\\Tools\\ninfer\\build\\apps\\ninfer-serve`、")
    L.append("> 同一模型文件、同一启动参数、同一 bench harness、同一 GPU（串行执行）。")
    L.append("> Docker = `lqbing/ninfer:local`（CUDA 13.1 runtime）；原生 = WSL Ubuntu-24.04（CUDA 13.0）。\n")

    # 冷启动
    L.append("## 1. 冷启动与显存\n")
    L.append("| 权重 | 运行 | 权重大小 | 加载权重耗时 | 读取速率 | 总启动 | 启动后显存 |")
    L.append("|---|---|---|---|---|---|---|")
    for key, label in [("nvfp4", "NVFP4"), ("mtp", "groupwise-int")]:
        for mode in ("docker", "native"):
            m = LOAD[(key, mode)]
            L.append(
                f"| {label} | {'Docker' if mode=='docker' else 'WSL 原生'} | {m['w_gib']} GiB | "
                f"{m['w_s']:.1f}s | {m['mbps']} MiB/s | {m['total_s']:.1f}s | "
                f"{m['vram'] if m['vram'] else '未采集'} MiB |"
            )
    L.append("")

    # 性能对比
    L.append("## 2. 性能对比（每场景 3 轮中位数）\n")
    L.append("> 注：`short_coding_gen` / `conc2_short` 的 **prefill 列受固定开销主导，不具可比性**，")
    L.append("> 判断 prefill 只看 `longctx_*` 行。`MTP接受率` = Σdraft_n_accepted/Σdraft_n（NInfer 原生 timings），")
    L.append("> 并发场景未聚合该列（显示 `-`）。\n")
    for title, pair in PAIRS.items():
        d = load(os.path.join(a.data_dir, f"{pair['docker']}.jsonl"))
        n = load(os.path.join(a.data_dir, f"{pair['native']}.jsonl"))
        L.append(f"### {title}\n")
        L.append("| 场景 | TTFT d/n (s) | Prefill d/n (tok/s) | Prefill Δ | Decode d/n (tok/s) | MTP接受率 d/n | 总耗时 d/n (s) | wall d/n (s) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for sc in SCENARIOS:
            rd, rn = d.get(sc), n.get(sc)
            L.append(
                f"| {sc} | {f(rd.get('ttft_s_med') if rd else None,2)} / {f(rn.get('ttft_s_med') if rn else None,2)} "
                f"| {f(prefill_tps(rd),0)} / {f(prefill_tps(rn),0)} "
                f"| {pct(prefill_tps(rn), prefill_tps(rd))} "
                f"| {f(rd.get('decode_tok_s_med') if rd else None,1)} / {f(rn.get('decode_tok_s_med') if rn else None,1)} "
                f"| {f((accept_rate(rd)*100) if accept_rate(rd) else None,1)}% / {f((accept_rate(rn)*100) if accept_rate(rn) else None,1)}% "
                f"| {f(rd.get('total_s_med') if rd else None,2)} / {f(rn.get('total_s_med') if rn else None,2)} "
                f"| {f(rd.get('wall_s') if rd else None,2)} / {f(rn.get('wall_s') if rn else None,2)} |"
            )
        L.append("")

    L.append("## 3. 结论\n")
    L.append("### 3.1 Prefill（最可靠的指标，纯算力、确定性）：**无系统性差异**")
    L.append("| 权重 | 场景 | Docker | 原生 | Δ |")
    L.append("|---|---|---|---|---|")
    L.append("| NVFP4 | 32K prefill | 7202 | 7077 | −1.7% |")
    L.append("| NVFP4 | 128K prefill | 3843 | 3760 | −2.2% |")
    L.append("| groupwise-int | 32K prefill | 2881 | 2917 | +1.2% |")
    L.append("| groupwise-int | 128K prefill | 2124 | 2247 | +5.8% |")
    L.append("")
    L.append("prefill 不受采样/接受率影响，四组差异都在 ±6% 内且方向不一致（两负两正）→ ")
    L.append("**Docker 容器层没有带来可测量的计算惩罚**。")
    L.append("")
    L.append("### 3.2 Decode：波动大，**由 MTP 接受率驱动，而非运行环境**")
    L.append("groupwise-int 上原生 decode 看着高（32K +22.4%），但**同时接受率也更高**")
    L.append("（原生 ~63% vs Docker ~50%）——decode ∝ 接受率，两者是同源变化。")
    L.append("NVFP4 上接受率相当（两侧均 ~53%），decode 也基本相等（144.6 vs 144.3）。")
    L.append("→ **decode 的跨轮差异主要来自投机接受率随内容/采样的波动，不应解读为 Docker vs 原生的差异。**")
    L.append("")
    L.append("### 3.3 冷启动：原生略快")
    L.append("NVFP4 权重读取 201.3 vs 181.8 MiB/s（+10.7%，总启动 109s vs 122s）；")
    L.append("groupwise-int 基本持平（185.5 vs 183.9 MiB/s，102s vs 106s）。")
    L.append("差异来自 I/O 路径（Docker bind mount vs drvfs `/mnt/c`）与 CUDA 运行时版本（13.1 vs 13.0）。")
    L.append("")
    L.append("### 3.4 选型建议")
    L.append("- **性能上：Docker 与 WSL 原生等价**，无需为性能而改架构。")
    L.append("- **Docker 更省心**：依赖、隔离、复现、进程生命周期都由容器管理（本机当前即用 Docker）。")
    L.append("- **WSL 原生**：省去容器层、冷启动略快，但需自行维护 CUDA 依赖与进程守护；")
    L.append("  且要注意 `wsl -e` 拉起后台进程会被回收（须用 `setsid` 或由常驻 wsl.exe 持有，见 `bench/wsl_serve_fg.sh`）。\n")

    text = "\n".join(L)
    path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[info] wrote {path}")


if __name__ == "__main__":
    main()
