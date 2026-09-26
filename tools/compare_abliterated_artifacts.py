# -*- coding: utf-8 -*-
"""compare_abliterated_artifacts.py —— 实测对比 abliterated 的 v2 与 v3 容器。

回答：qwen3_8_27b_abliterated_original.ninfer 与 qwen3_8_27b_abliterated_nvfp4.ninfer
      到底差在哪（容器框架 / 目录 schema / 权重载荷是否一致）。

只读，不修改任何文件。
"""
import hashlib
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "models", "qwen3_8_27b_abliterated_original.ninfer")
V3 = os.path.join(ROOT, "models", "qwen3_8_27b_abliterated_nvfp4.ninfer")

# 与 upgrade_ninfer_v2_to_v3.py 的映射表一致，用于归一化格式名做比较
FORMATS = {
    "BF16": "bf16", "FP32": "fp32", "I32": "int32",
    "Q4G64_F16S": "q4_g64_fp16", "Q5G64_F16S": "q5_g64_fp16",
    "Q6G64_F16S": "q6_g64_fp16", "W8G32_F16S": "q8_g32_fp16",
    "NVFP4": "nvfp4", "FP8_E4M3FN_ROW_BF16S": "fp8_e4m3fn_row_bf16",
}
LAYOUTS = {
    "contiguous-le-v1": "contiguous_le_v1",
    "row-split-k128-v1": "row_split_k128_v1",
    "blockscale-k16-m128x4-v1": "block_scale_k16_m128x4_v1",
    "row-scale-v1": "row_scale_v1",
}


def align(v, a=4096):
    return (v + a - 1) // a * a


def parse_v2(path):
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        header = f.read(16)
        magic, count = header[:8], struct.unpack_from("<Q", header, 8)[0]
        directory = json.loads(f.read(count))
        start = align(16 + count)
    return {
        "magic": magic, "version": magic[7], "dir_bytes": count,
        "payload_start": start, "size": size, "payload_bytes": size - start,
        "directory": directory,
    }


def parse_v3(path):
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        header = f.read(32)
        magic, dlen = header[:8], struct.unpack_from("<Q", header, 8)[0]
        identity = header[16:32]
        directory = json.loads(f.read(dlen))
        start = align(32 + dlen)
    return {
        "magic": magic, "version": magic[7], "dir_bytes": dlen,
        "identity": identity.hex(), "payload_start": start, "size": size,
        "payload_bytes": size - start, "directory": directory,
    }


def sha256_region(path, off, length, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = length
        while left:
            b = f.read(min(chunk, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def tensor_map(directory, key_field):
    """归一化 (name -> (bytes, shape, format, layout))，用于证明张量集合一致。"""
    out = {}
    for o in directory["objects"]:
        if o.get("kind") != "tensor":
            continue
        out[o[key_field]] = (
            o["bytes"], tuple(o["shape"]),
            FORMATS.get(o["format"], o["format"]),
            LAYOUTS.get(o["layout"], o["layout"]),
        )
    return out


def main():
    v2, v3 = parse_v2(V2), parse_v3(V3)
    print("=" * 78)
    print("容器框架")
    print("=" * 78)
    print(f"{'项':<18}{'v2':>22}{'v3':>22}")
    for label, a, b in [
        ("magic", v2["magic"], v3["magic"]),
        ("version byte", f"{v2['version']:#04x}", f"{v3['version']:#04x}"),
        ("目录 JSON 字节", f"{v2['dir_bytes']:,}", f"{v3['dir_bytes']:,}"),
        ("载荷起始偏移", f"{v2['payload_start']:,}", f"{v3['payload_start']:,}"),
        ("文件总字节", f"{v2['size']:,}", f"{v3['size']:,}"),
        ("载荷区字节", f"{v2['payload_bytes']:,}", f"{v3['payload_bytes']:,}"),
    ]:
        print(f"{label:<18}{str(a):>22}{str(b):>22}")
    print(f"{'16字节身份/哈希':<16}{'(无)':>22}{v3['identity']:>22}")

    print("\n" + "=" * 78)
    print("目录 schema（顶层键）")
    print("=" * 78)
    print("v2:", sorted(v2["directory"].keys()))
    print("v3:", sorted(v3["directory"].keys()))

    print("\n" + "=" * 78)
    print("对象统计")
    print("=" * 78)
    for tag, d, key in [("v2", v2, "name"), ("v3", v3, "id")]:
        objs = d["directory"]["objects"]
        kinds = {}
        for o in objs:
            kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
        fmts = {}
        for o in objs:
            if o["kind"] == "tensor":
                fmts[o["format"]] = fmts.get(o["format"], 0) + 1
        print(f"{tag}: objects={len(objs)} kinds={kinds}")
        print(f"    formats={dict(sorted(fmts.items()))}")

    print("\n" + "=" * 78)
    print("v3 独有的语义信息")
    print("=" * 78)
    d3 = v3["directory"]
    print("components:", list(d3.get("components", {}).keys()))
    print("metadata:", d3.get("metadata"))
    print("provenance:", d3.get("provenance"))
    print("bindings 条数:", len(d3.get("bindings", {})))
    print("uses 条数:", len(d3.get("uses", [])))
    print("files:", d3.get("files"))
    res = d3.get("components", {}).get("text", {}).get("resources", {})
    print("text resources:", list(res.keys()))
    vis = d3.get("components", {}).get("vision", {}).get("resources", {})
    print("vision resources:", list(vis.keys()))
    print("v2 identity:", v2["directory"].get("identity"))

    print("\n" + "=" * 78)
    print("张量集合是否一致（name -> bytes/shape/format/layout）")
    print("=" * 78)
    m2 = tensor_map(v2["directory"], "name")
    m3 = tensor_map(v3["directory"], "id")
    only2 = sorted(set(m2) - set(m3))
    only3 = sorted(set(m3) - set(m2))
    diff = sorted(k for k in set(m2) & set(m3) if m2[k] != m3[k])
    print(f"张量数: v2={len(m2)} v3={len(m3)}")
    print(f"仅在 v2: {len(only2)} -> {only2[:5]}")
    print(f"仅在 v3: {len(only3)} -> {only3[:5]}")
    print(f"几何/编码不同: {len(diff)} -> {diff[:5]}")

    print("\n" + "=" * 78)
    print("权重载荷逐字节比对（v2 载荷 vs v3 载荷头部）")
    print("=" * 78)
    n = v2["payload_bytes"]
    h2 = sha256_region(V2, v2["payload_start"], n)
    h3 = sha256_region(V3, v3["payload_start"], n)
    print(f"v2 载荷 [{v2['payload_start']:,} , {v2['size']:,})  sha256={h2}")
    print(f"v3 载荷 [{v3['payload_start']:,} , {v3['payload_start']+n:,})  sha256={h3}")
    print("结论:", "✅ 权重载荷逐字节相同" if h2 == h3 else "❌ 载荷不同")

    tail = v3["size"] - (v3["payload_start"] + n)
    print(f"\nv3 载荷之后的额外 {tail:,} 字节 = 对齐填充 + 新增 chat template")
    with open(V3, "rb") as f:
        f.seek(v3["size"] - 9712)
        head = f.read(120).decode("utf-8", "replace")
    print("v3 末尾 9712 字节（template）开头:", repr(head[:80]))


if __name__ == "__main__":
    main()
