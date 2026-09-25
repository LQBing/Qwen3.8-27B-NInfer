# -*- coding: utf-8 -*-
"""prefill_sweep.py —— 扫描不同上下文长度的 prefill 速度，定位劣化曲线/拐点。"""
import argparse
import json
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")

import ninfer_probe as np  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--targets", default="32000,64000,96000,128000,160000,192000,224000,256000")
_ap.add_argument("--max-tokens", type=int, default=64)
_args = _ap.parse_args()
TARGETS = [int(x) for x in _args.targets.split(",") if x.strip()]

body = np.build_body(min_chars=12_000_000)
pt, nchar = np.probe_tokens(body[:120000])
cpt = nchar / max(pt, 1)
print(f"[info] chars/token = {cpt:.3f}")

rows = []
for idx, t in enumerate(TARGETS):
    cut = min(int(t * cpt * 1.02), len(body))
    # 关键：各点用不重叠的 body 区间 → 前缀互不相同，避免前缀缓存污染
    off = min(idx * cut, max(0, len(body) - cut))
    prompt = body[off:off + cut] + f"\n# salt {uuid.uuid4()}\n"
    m = np.stream_once(prompt, _args.max_tokens)
    if m["err"]:
        print(f"{t:>7d} tok  ERR {m['err'][:100]}")
        continue
    # 用真实 prompt_tokens 校正
    tp = int(cut / cpt)
    tps = tp / m["ttft"] if m["ttft"] else None
    rows.append((t, m["ttft"], tps, m["decode"], m.get("cache_n")))
    print(f"{t:>7d} tok  TTFT={m['ttft']:8.2f}s  prefill={tps:8.0f} tok/s  "
          f"decode={m['decode']:.1f}  cache_n={m.get('cache_n')}")

print("\n## 汇总（相对最强点的倍数）")
base = rows[0][2] if rows else None
for t, ttft, tps, dec, cn in rows:
    print(f"| {t:,} | {tps:,.0f} | {ttft:.1f}s | {tps/base:.2f}× |")
