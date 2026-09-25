# -*- coding: utf-8 -*-
"""prefix_test.py —— 验证前缀复用（同一前缀第二次请求应命中缓存，TTFT 大幅下降）。"""
import json
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")

import ninfer_probe as np  # noqa: E402

body = np.build_body()
pt, nchar = np.probe_tokens(body[:120000])
cpt = nchar / max(pt, 1)
prompt = body[: int(32000 * cpt * 1.02)]  # ~32K 固定前缀（不加 salt）

print(f"[info] 前缀约 {len(prompt)} 字符 / ~{int(len(prompt)/cpt)} tok")
r1 = np.stream_once(prompt, 128)
print("第1次(冷)  ", json.dumps({k: r1[k] for k in ("ttft", "decode", "cache_n", "err")}, ensure_ascii=False))
r2 = np.stream_once(prompt, 128)
print("第2次(同前缀)", json.dumps({k: r2[k] for k in ("ttft", "decode", "cache_n", "err")}, ensure_ascii=False))
if r1["ttft"] and r2["ttft"]:
    print(f"[结论] TTFT {r1['ttft']:.2f}s -> {r2['ttft']:.2f}s（降 {(1-r2['ttft']/r1['ttft'])*100:.0f}%）")
