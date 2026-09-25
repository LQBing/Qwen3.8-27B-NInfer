# 三实例实测汇总 — Qwen3.8-27B @ RTX 5090D 32G

> 统一 harness（`bench/bench_unified.py`），每场景 3 轮取中位数；
> 每轮 prompt 加唯一 salt 使前缀缓存失效（`cache_n=0` 已验证）。
> TTFT 口径 = 第一个非空 token（含 reasoning_content）。

## 1. 冷启动与资源

| 实例 | 引擎 | 权重 | 冷启动 | 启动后显存 | 上下文上限 | 思考策略 |
|---|---|---|---|---|---|---|
| `sglang` | SGLang | NVFP4 LMHead4 + DSpark-NVFP4 drafter | 157s | 30786 MiB | 163840 | reasoning_effort=medium |
| `ninfer-nvfp4` | NInfer | qwen3_8_27b_nvfp4.ninfer | 122s | 30751 MiB | 240000 | preserve-thinking (默认开思考) |
| `ninfer-mtp` | NInfer | qwen3_8_27b.ninfer | 106s | 27905 MiB | 240000 | preserve-thinking (默认开思考) |

## 2. TTFT (s，越低越好)

| 场景 | sglang | ninfer-nvfp4 | ninfer-mtp |
|---|---|---|---|
| short_coding_gen | 0.128 | 0.106 | 0.150 |
| longctx_32k_prefill | 4.135 | 4.657 | 11.647 |
| longctx_32k_longgen | 4.079 | 4.668 | 11.688 |
| longctx_128k_prefill | 35.659 | 34.637 | 63.131 |
| conc2_short | 0.213 | 0.130 | 0.242 |
| conc2_long_distinct | 6.077 | 7.386 | 17.829 |

## 3. Prefill 速度 (prompt_tokens / TTFT，tok/s，越高越好)

| 场景 | sglang | ninfer-nvfp4 | ninfer-mtp |
|---|---|---|---|
| short_coding_gen | 976 | 1571 | 1083 |
| longctx_32k_prefill | 7893 | 7009 | 2802 |
| longctx_32k_longgen | 8002 | 6992 | 2793 |
| longctx_128k_prefill | 3661 | 3769 | 2068 |
| conc2_short | 586 | 1278 | 672 |
| conc2_long_distinct | 5371 | 4419 | 1831 |

## 4. Decode 速度 (tok/s，越高越好)

| 场景 | sglang | ninfer-nvfp4 | ninfer-mtp |
|---|---|---|---|
| short_coding_gen | 166.2 | 154.0 | 153.6 |
| longctx_32k_prefill | 124.5 | 136.2 | 146.0 |
| longctx_32k_longgen | 130.1 | 145.9 | 144.7 |
| longctx_128k_prefill | 113.6 | 147.4 | 143.9 |
| conc2_short | 157.2 | 161.7 | 116.3 |
| conc2_long_distinct | 82.0 | 82.9 | 66.3 |

## 5. 端到端总耗时 (s，越低越好)

| 场景 | sglang | ninfer-nvfp4 | ninfer-mtp |
|---|---|---|---|
| short_coding_gen | 3.20 | 3.42 | 3.50 |
| longctx_32k_prefill | 6.17 | 6.54 | 13.40 |
| longctx_32k_longgen | 8.39 | 11.68 | 18.74 |
| longctx_128k_prefill | 36.80 | 35.56 | 64.05 |
| conc2_short | 3.45 | 3.32 | 4.65 |
| conc2_long_distinct | 10.17 | 11.72 | 25.84 |

## 6. 并发 2 墙钟 (wall_s) 与单流劣化

| 场景 | 指标 | sglang | ninfer-nvfp4 | ninfer-mtp |
|---|---|---|---|---|
| conc2_short | wall_s | 3.50 | 3.45 | 4.81 |
| conc2_short | 最慢请求 decode | 136.1 | 138.3 | 106.1 |
| conc2_long_distinct | wall_s | 10.22 | 12.12 | 26.22 |
| conc2_long_distinct | 最慢请求 decode | 42.2 | 38.0 | 18.4 |
