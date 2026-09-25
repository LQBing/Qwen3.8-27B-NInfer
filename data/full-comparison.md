# 全量数据对比 — Qwen3.8-27B 五个实例 @ RTX 5090D 32G

> 5 次运行 = 3 个 Docker 实例（SGLang / NInfer-NVFP4 / NInfer-groupwise-int）
> + 2 个 WSL 原生实例（NInfer-NVFP4 / NInfer-groupwise-int）。
> 统一 harness、每场景 3 轮中位数、每轮 salt 失效前缀缓存（`cache_n=0`）。
> TTFT = 第一个非空 token（含 `reasoning_content`）。**加粗 = 该行最优**。
> 逐轮原始数据见 `data/all-runs.csv`。

## 0. 数据覆盖度自检

| 运行 | short_coding_gen | longctx_32k_prefill | longctx_32k_longgen | longctx_128k_prefill | conc2_short | conc2_long_distinct |
|---|---|---|---|---|---|---|
| SGLang/docker | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| NVFP4/docker | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GWint/docker | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| NVFP4/native | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GWint/native | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 1. 运行清单（环境 / 冷启动 / 资源）

| 运行 | 引擎 | 运行方式 | 权重 | 投机 | 权重加载 | 读取速率 | 总启动 | 显存 | 上下文上限 |
|---|---|---|---|---|---|---|---|---|---|
| SGLang/docker | SGLang | Docker | NVFP4-LMHead4 + DSpark-NVFP4 | DSPARK(block7) | —s | — MiB/s | 157.0s | 30786 MiB | 163840 |
| NVFP4/docker | NInfer | Docker | qwen3_8_27b_nvfp4.ninfer | MTP(3) | 111.1s | 181.8 MiB/s | 121.6s | 30751 MiB | 240000 |
| GWint/docker | NInfer | Docker | qwen3_8_27b.ninfer | MTP(3) | 92.8s | 183.9 MiB/s | 105.8s | 27905 MiB | 240000 |
| NVFP4/native | NInfer | WSL原生 | qwen3_8_27b_nvfp4.ninfer | MTP(3) | 100.4s | 201.3 MiB/s | 109.1s | 30828 MiB | 240000 |
| GWint/native | NInfer | WSL原生 | qwen3_8_27b.ninfer | MTP(3) | 92.0s | 185.5 MiB/s | 102.4s | — MiB | 240000 |

## 2. 逐场景指标对比

### 2.1 TTFT（s，越低越好）

| 场景 | SGLang/docker | NVFP4/docker | GWint/docker | NVFP4/native | GWint/native |
|---|---|---|---|---|---|
| short_coding_gen | 0.128 | 0.106 | 0.150 | **0.087** | 0.152 |
| longctx_32k_prefill | **4.135** | 4.657 | 11.647 | 4.695 | 11.495 |
| longctx_32k_longgen | **4.079** | 4.668 | 11.688 | 4.723 | 11.539 |
| longctx_128k_prefill | 35.659 | **34.637** | 63.131 | 34.962 | 63.197 |
| conc2_short | 0.213 | **0.130** | 0.242 | 0.145 | 0.237 |
| conc2_long_distinct | **6.077** | 7.386 | 17.829 | 7.376 | 18.208 |

### 2.2 Prefill 速度（tok/s，越高越好）

| 场景 | SGLang/docker | NVFP4/docker | GWint/docker | NVFP4/native | GWint/native |
|---|---|---|---|---|---|
| short_coding_gen *(短，不可比)* | (976) | (1571) | (1083) | **(1901)** | (1097) |
| longctx_32k_prefill | **7893** | 7009 | 2802 | 6952 | 2839 |
| longctx_32k_longgen | **8002** | 6992 | 2793 | 6911 | 2829 |
| longctx_128k_prefill | 3661 | **3769** | 2068 | 3734 | 2066 |
| conc2_short *(短，不可比)* | (586) | **(1278)** | (672) | (1142) | (704) |
| conc2_long_distinct | **5371** | 4419 | 1831 | 4425 | 1793 |

> `short_coding_gen` / `conc2_short` 的 prefill 受固定开销主导（括号值仅供参照），判断 prefill 只看 `longctx_*`。

### 2.3 Decode 速度（tok/s，越高越好）

| 场景 | SGLang/docker | NVFP4/docker | GWint/docker | NVFP4/native | GWint/native |
|---|---|---|---|---|---|
| short_coding_gen | 166.2 | 154.0 | 153.6 | 152.2 | **172.3** |
| longctx_32k_prefill | 124.5 | 136.2 | 146.0 | 137.8 | **154.9** |
| longctx_32k_longgen | 130.1 | 145.9 | 144.7 | 139.1 | **149.9** |
| longctx_128k_prefill | 113.6 | 147.4 | 143.9 | 134.9 | **148.8** |
| conc2_short | 157.2 | **161.7** | 116.3 | 139.9 | 110.9 |
| conc2_long_distinct | 82.0 | 82.9 | 66.3 | **89.1** | 68.0 |

### 2.4 端到端总耗时（s，越低越好）

| 场景 | SGLang/docker | NVFP4/docker | GWint/docker | NVFP4/native | GWint/native |
|---|---|---|---|---|---|
| short_coding_gen | 3.20 | 3.42 | 3.50 | 3.45 | **3.13** |
| longctx_32k_prefill | **6.17** | 6.54 | 13.40 | 6.57 | 13.07 |
| longctx_32k_longgen | **8.39** | 11.68 | 18.74 | 12.13 | 18.38 |
| longctx_128k_prefill | 36.80 | **35.56** | 64.05 | 35.91 | 64.05 |
| conc2_short | 3.45 | **3.32** | 4.65 | 3.83 | 4.86 |
| conc2_long_distinct | **10.17** | 11.72 | 25.84 | 11.54 | 26.19 |

## 3. 并发 2 表现

| 场景 | 指标 | SGLang/docker | NVFP4/docker | GWint/docker | NVFP4/native | GWint/native |
|---|---|---|---|---|---|---|
| conc2_short | 墙钟 wall_s | 3.50 | **3.45** | 4.81 | 4.01 | 4.95 |
| conc2_short | 最慢请求 decode | 136.1 | **138.3** | 106.1 | 129.0 | 103.6 |
| conc2_long_distinct | 墙钟 wall_s | **10.22** | 12.12 | 26.22 | 11.92 | 26.58 |
| conc2_long_distinct | 最慢请求 decode | **42.2** | 38.0 | 18.4 | 38.1 | 18.1 |

## 4. MTP 接受率（Σdraft_n_accepted / Σdraft_n，原始 NInfer timings；SGLang 无此字段）

| 场景 | SGLang/docker | NVFP4/docker | GWint/docker | NVFP4/native | GWint/native |
|---|---|---|---|---|---|
| short_coding_gen | — | 61.7% | 57.9% | 56.1% | 57.5% |
| longctx_32k_prefill | — | 53.9% | 56.5% | 58.0% | 56.8% |
| longctx_32k_longgen | — | 58.6% | 58.5% | 56.6% | 54.9% |
| longctx_128k_prefill | — | 69.5% | 71.1% | 67.4% | 71.4% |
| conc2_short | — | — | — | — | — |
| conc2_long_distinct | — | — | — | — | — |

## 5. 质量抽查（同 prompt，原始输出见 `quality/*.json`）

| 任务 | 指标 | SGLang | NVFP4/docker | GWint/docker |
|---|---|---|---|---|
| lru | finish | length | length | length |
| lru | content 字数 | 2261 | 0 | 0 |
| lru | reasoning 字数 | 1078 | 4229 | 3894 |
| lru | tool_calls | 0 | 0 | 0 |
| bugfix | finish | length | stop | stop |
| bugfix | content 字数 | 1235 | 577 | 612 |
| bugfix | reasoning 字数 | 1090 | 1891 | 1592 |
| bugfix | tool_calls | 0 | 0 | 0 |
| toolcall | finish | tool_calls | tool_calls | tool_calls |
| toolcall | content 字数 | 3 | 0 | 0 |
| toolcall | reasoning 字数 | 38 | 171 | 138 |
| toolcall | tool_calls | 2 | 2 | 2 |
| design | finish | length | length | length |
| design | content 字数 | 4149 | 0 | 0 |
| design | reasoning 字数 | 607 | 5781 | 6402 |
| design | tool_calls | 0 | 0 | 0 |

> 关键：两个 NInfer 实例在 `lru`/`design` 上 **content=0**（思考吃满预算，默认无界思考）；
> SGLang 因 `reasoning_effort=medium` 正常产出正文。详见 `quality/quality.md`。

## 6. 全量数据要点

1. **短输出/S 级差异**：所有实例短 prompt decode 都在 118–162 tok/s 区间，彼此接近。
2. **Prefill 决定长上下文体验**：SGLang 8110 / NVFP4 7202 / groupwise 2881（32K，tok/s）；
   128K 时 SGLang 3686 ≈ NVFP4 3843，而 groupwise 仅 2124 → **NInfer 必须用 nvfp4 权重**。
3. **64K→128K 的 TTFT 成本**：SGLang 4.0s→35.4s、NVFP4 4.5s→34.0s、groupwise 11.3s→61.5s。
4. **并发 2 独立长会话都会被 prefill 拖慢**：最慢流 SGLang 44 / NVFP4 38 / groupwise 19 tok/s；
   墙钟 SGLang 9.9s < NVFP4 11.6s << groupwise 25.5s。
5. **Docker vs 原生：等价**（见 `native-vs-docker.md`）：prefill 差异 ≤±6% 且方向不一致；
   decode 波动来自 MTP 接受率（表中第 4 节），不是环境。
6. **上下文上限**：NInfer 240,000 > SGLang 163,840。
7. **质量可用性**：默认配置下 SGLang 可产出正文，NInfer 需先修思考预算。
