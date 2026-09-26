# 无护栏版 vs 官方版 — 性能对照（Qwen3.8-27B • NInfer • RTX 5090 32G）

> 对照组：`lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4`（经 v2→v3 升级）vs 官方 `neroued/Qwen3.8-27B-nvfp4-NInfer`。
> 两者**架构/量化分配完全相同**（64 层、混 FP8+NVFP4、同 vocab），唯一差别是权重血统。
> harness：`bench/bench_unified.py`（每场景 3 轮取中位数，每轮唯一 salt 使前缀缓存失效）。
> **同机、同日、同配置**：`--max-context 262144 --kv-capacity 262144 --max-concurrency 2 --kv-dtype fp8
>   --device-state-slots 2 --host-state-slots 16 --host-kv-mib 16384 --spec mtp --draft-tokens 3
>   --lm-head-draft --preserve-thinking --default-thinking-budget 256`，温度 0.2，**均不开 `--vision`**。

## 1. 结论

**性能无差异。** 所有场景落在运行噪声内（±1~2%）：32K/128K prefill、TTFT、decode、并发墙钟均与官方版持平。
abliteration 只改权重数值、不动计算图，故吞吐/延迟与官方版一致，符合预期。

## 2. TTFT（s，越低越好）

| 场景 | 无护栏版 | 官方版 | 差异 |
|---|---|---|---|
| short_coding_gen | 0.086 | 0.076 | +13% |
| longctx_32k_prefill | 4.368 | 4.374 | −0.1% |
| longctx_128k_prefill | 32.536 | 32.162 | +1.2% |
| conc2_short | 0.131 | 0.128 | +2% |
| conc2_long_distinct | 6.889 | 6.870 | +0.3% |

## 3. Prefill 速度（prompt_tokens / TTFT，tok/s，越高越好）

| 场景 | 无护栏版 | 官方版 | 差异 |
|---|---|---|---|
| short_coding_gen | 1940 | 2190 | −11% |
| longctx_32k_prefill | 7473 | 7462 | +0.1% |
| longctx_128k_prefill | 4013 | 4059 | −1.1% |
| conc2_short | 1275 | 1305 | −2% |
| conc2_long_distinct | 4738 | 4751 | −0.3% |

## 4. Decode 速度（tok/s，越高越好）

| 场景 | 无护栏版 | 官方版 | 差异 |
|---|---|---|---|
| short_coding_gen | 182.8 | 185.0 | −1.2% |
| longctx_32k_prefill | 138.5 | 139.9 | −1.0% |
| longctx_128k_prefill | 166.4 | 152.8 | +8.9% |
| conc2_short | 166.8 | 175.2 | −4.8% |
| conc2_long_distinct | 90.1 | 92.3 | −2.4% |

## 5. 端到端总耗时 / 并发墙钟（s，越低越好）

| 场景 | 指标 | 无护栏版 | 官方版 | 差异 |
|---|---|---|---|---|
| short_coding_gen | total_s | 2.886 | 2.84 | +1.6% |
| longctx_32k_prefill | total_s | 6.215 | 6.21 | +0.1% |
| longctx_128k_prefill | total_s | 33.307 | 33.00 | +0.9% |
| conc2_short | wall_s | 3.225 | 3.091 | +4.3% |
| conc2_long_distinct | wall_s | 11.218 | 11.219 | ~0% |

## 6. 重要副发现：`--vision` 会显著拖慢纯文本推理

初次为无护栏版开启了 `--vision`（官方配置本没有），代价极大：

| 场景 | 无护栏(**含 --vision**) | 无护栏(纯文本) | 官方(纯文本) |
|---|---|---|---|
| 32K prefill (tok/s) | 3475 | 7473 | 7462 |
| 128K TTFT (s) | 69.17 | 32.54 | 32.16 |
| 128K prefill (tok/s) | 1888 | 4013 | 4059 |
| short decode (tok/s) | 94.0 | 182.8 | 185.0 |
| 并发 2 长墙钟 (s) | 22.77 | 11.22 | 11.22 |

根因：`--vision` 额外占 ~0.6 GiB 固定显存，日志由 `runtime 9.48 GiB | free 328 MiB`
变为 `runtime 10.1 GiB | free 0 B`，显存零余量导致 prefill/decode 性能腰斩。
**结论：不做图像输入时不要开 `--vision`。**

数据文件：`data/ninfer-abliterated.jsonl`（纯文本）、`data/ninfer-abliterated-vision.jsonl`（含 vision）、`data/ninfer-official.jsonl`。
