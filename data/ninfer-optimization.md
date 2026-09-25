# NInfer-NVFP4 优化实验报告（RTX 5090D 32G）

> 日期：2026-09-25 ｜ 目标：在 32G 显存下榨出**更长上下文**或**更高并发**，服务本地复杂软件项目 coding
> 全部为实测（容器重启 + `bench/ninfer_probe.py` / `prefix_test.py`）

---

## 1. 结论速览

| 方向 | 实测结果 | 采纳 |
|---|---|---|
| **更高并发（3）** | ❌ **反而更差**：3 并发 32K 独立会话墙钟 **36.5s**（2 并发仅 11.7s）；短任务 decode 中位 **87**（2 并发 148） | 不采纳，**保持 `--max-concurrency 2`** |
| **更长上下文（262,144）** | ✅ 可用（模型上限即 262144）；但 >128K prefill 超线性变慢 | **采纳 262144** |
| **host KV/state 放大** | ✅ **不占显存**：32 GiB host KV + 16 state slots（RAM 充裕） | **采纳** |
| **前缀复用** | ✅ 极有效：同前缀第 2 次 **TTFT 4.48s → 0.07s（−98%）**（`cache_n=32624`） | 依赖上面两项 |

---

## 2. 内存模型（实测）

| 项 | 值 |
|---|---|
| 权重 | 19.7 GiB |
| KV（fp8） | ≈ **32 KB/token**（16 个全注意力层 × KV4头 × head_dim256 × 1B） |
| 240,000 tok | runtime 8.76 GiB，**free 1.49 GiB** |
| 262,144 tok | runtime 9.48 GiB，**free 0.75 GiB**（更紧，但实测稳定） |
| 模型上限 | `max_position_embeddings = 262144` |

四个容量轴**相互独立**（Device StateImage slots / Host StateImage slots / Device KV pages / Host KV bytes），一个轴的余量不能补另一个（NInfer `resource-scheduling-and-context-cache.md`）。

---

## 3. 实验明细

### 3.1 更高并发 —— 失败
| 配置 | 场景 | 墙钟 | decode 中位 | 最慢流 |
|---|---|---|---|---|
| conc=2（基线） | 2×32K 独立 | 11.17s | 100.9 | 40.4 |
| **conc=3** | 3×32K 独立 | **36.55s** | **19.4** | **11.2** |
| conc=3 | 3×短(512) | 6.49s | 86.8 | 81.4 |
| conc=2（对照短） | 2×短(512) | 3.75s | 147.9 | — |

**原因**：NInfer 是「**单批 decode + prefill 不与 decode 重叠**」模型，新请求的 prefill 会阻塞在跑请求的 decode；并发越高，长 prompt 的排队/阻塞越严重。聚合吞吐也下降（3×512/6.49s = 236 tok/s < 2×512/3.75s = 273 tok/s）。

→ **并发 2 是甜点**。需要真并发请用 SGLang（其 3 并发 32K 实测墙钟 15.0s，远优于 NInfer 的 36.5s）。

### 3.2 更长上下文 —— 可用；prefill 是**平滑二次劣化**（**无 128K 悬崖**）

实测扫描（nvfp4 + MTP3，单请求，最终 conc2 配置，脚本 `bench/prefill_sweep.py`）：

| Prompt tokens | prefill (tok/s) | TTFT |
|---:|---:|---:|
| 32,000 | 7,721 | 4.2s |
| 64,000 | 6,057 | 10.8s |
| 96,000 | 4,929 | 19.9s |
| 128,000 | 4,158 | 31.4s |
| 160,000 | 3,587 | 45.5s |
| 192,000 | 3,154 | 62.1s |
| 224,000 | 2,824 | 80.9s |
| 256,000 | 2,552 | 102.3s |

官方 nvfp4（MTP0）曲线方向一致：7,680→8,340、64,512→5,298、130,048→3,545、260,096→2,203 tok/s。

**机制**：模型 64 层里只有每 4 层取 1 的 **16 层是全注意力**，其余 48 层是线性注意力（Gated DeltaNet）。全注意力 O(n²)、线性 O(n)，于是每 token 耗时

```
t(n) ≈ a + b·n        →  prefill 总时间 ∝ n + b·n²   →  O(n²)
```

实测拟合：**b ≈ 1.18e-9 /token，a ≈ 9.2e-5**（每 32K 区间的增量恒为 ~3.8e-5，线性拟合极好）。
- **拐点**（二次项 = 常数项）在 **n ≈ a/b ≈ 78K**（官方数据外推约 85K）
- 所以下滑是**连续的**：32K→7.7K tok/s，128K→4.2K，256K→2.6K（8 倍长度掉到 1/3）；TTFT 近似按 n² 增长（128K 31s，256K 102s）

→ **没有"128K 悬崖"**：劣化从 ~78K 起平滑加速，128K 只是这条平滑曲线上的一个点。

> 勘误：早先记录的"250K → 1163 tok/s"是**内存紧张配置**（`--max-concurrency 3`，free 仅 429 MiB）下的异常值；最终 conc2 配置下 256K 实测 **2552 tok/s**。

### 3.3 host 缓存放大 —— 免费收益
`--host-kv-mib 8192 → 32768`、`--host-state-slots 8 → 16`：显存占用**完全不变**（host 是 pinned RAM；本机 96 GB，用 32 GB 仍宽裕），只增加**不活跃检查点的保留量**，减少跨轮重复 prefill。

### 3.4 前缀复用 —— 多轮 coding 的关键
同一 32K 前缀第 2 次请求：`cache_n=32624`，**TTFT 4.48s → 0.07s**。多轮会话（系统提示+仓库上下文固定，逐轮追加）因此几乎零首字延迟——这是本地 coding agent 最重要的加速点。

---

## 3.5 与 SGLang 的 prefill 劣化对比（含方法学更正）

同机、同模型（Qwen3.8-27B 混合架构）、同一提示词构造。
**方法学要点**：各长度点必须用**互不重叠的前缀**——否则后一点会复用前一点的前缀，被前缀缓存命中而把数字抬高。

| Prompt tokens | NInfer（nvfp4, MTP3） | SGLang（DSPARK） | 差异 |
|---:|---:|---:|---|
| 32,000 | 7,747 | 9,051 | SGLang +17% |
| 64,000 | 6,066 | 6,487 | +7% |
| 96,000 | 4,938 | 5,059 | +2% |
| 128,000 | **4,135** | **4,110** | **打平** |
| 160,000 | 3,587 | — | SGLang 上限 163,840 |
| 192,000 | 3,153 | — | 仅 NInfer |
| 256,000 | 2,543 | — | 仅 NInfer |

（单位 tok/s；NInfer 复测 `cache_n=0`，无缓存污染。）

**结论**：
1. **两者 prefill 劣化曲线几乎重合**——同一种平滑二次劣化（同架构使然）。差异 ≤17% 且随长度收窄，**128K 处打平**。
2. **SGLang 的独有优势是并发**（§3.1：3 并发 32K 实测 15.0s vs NInfer 36.5s），短上下文也略快（+17%）。
3. **NInfer 的独有优势是上下文上限 262,144**（SGLang 仅 163,840，受 KV 池 161,106 限制）。
4. ⚠️ **方法学更正**：早先一次未控制前缀的扫描显示 SGLang 在 64–128K "保持 ~10K tok/s 且不下降"，那是**前缀缓存命中**的假象（后一点复用了前一点的前缀）；改为互不重叠前缀后，SGLang 立即回落到与 NInfer 相同的曲线。据此，"SGLang 长上下文 prefill 远快于 NInfer"的说法**不成立**。

## 3.6 上下文还能再拉高吗？—— **单序列不能；KV 池可以**

实测（`ninfer-serve` 直接启动）：

| 尝试 | 结果 |
|---|---|
| `--max-context 393216` | ❌ **FATAL: max_context exceeds the configured position capacity**（拒绝启动） |
| `--max-context 262144 --kv-capacity 524288 --kv-dtype nvfp4` | ❌ 差 **4 MB** 装不下（需 11.2481 GB，可用 11.2439 GB） |
| `--max-context 262144 --kv-capacity 491520 --kv-dtype nvfp4` | ✅ 装下：`KV 491,520 tokens, nvfp4 | runtime 9.88 GiB | free 418 MiB`，但 **`max_model_len` 仍是 262144** |

**结论**：
1. **单序列逻辑上下文 = 262,144 是硬上限**，由模型配置 `max_position_embeddings = 262144`（`rope_type=default`、`rope_theta=1e7`）决定；NInfer 会直接拒绝更大的 `--max-context`。
2. **KV「池」可以单独拉大**，但它只服务**并发共享**，不改变单序列上限：
   - fp8：262,144 池 ≈ 8.4 GB，free 仅 ~0.73 GB → 基本到顶
   - **nvfp4 KV 可把池子提到 ~491,520**（free 0.42 GB），代价是 **KV 量化（质量未验证）**
   - 池大于 262,144 只在"2 个会话都很长"时有用（且 NInfer 并发本身就弱）
3. 要单序列 **>262,144**，只能**改模型**（提高 `max_position_embeddings` + 正确的 rope/YaRN 缩放），属模型转换层，且质量未知——不建议在本地随意改。

→ **不建议继续拉**：单序列拿不到更多；池子放大收益窄且要牺牲 KV 精度。当前配置（fp8 262144）已是"安全区最优"。

## 4. 最终配置（已写入 `Qwen3.8-27B-NInfer/docker-compose.yaml`）

```bash
ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0 --port 30000 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 2 \
  --kv-dtype fp8 --device-state-slots 2 --host-state-slots 16 --host-kv-mib 32768 \
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking \
  --default-thinking-budget 256
```

相对你原配置的**净变更**：

```diff
- --max-context 240000 --kv-capacity 240000 --max-concurrency 2
+ --max-context 262144 --kv-capacity 262144 --max-concurrency 2
- --host-state-slots 8 --host-kv-mib 8192
+ --host-state-slots 16 --host-kv-mib 32768
+ --default-thinking-budget 256      # 上一轮加入（解决"只思考不产出"）
```

启动画像：`KV 262,144 tokens, fp8 | runtime 9.48 GiB | free 746.9 MiB | host 16 states, 32.0 GiB KV`

**回退**：把 `262144` 改回 `240000`（多拿回 ~0.75 GiB 余量）；`host-*` 参数独立，可单独回退。

---

## 5. 使用建议

1. **并发固定 2**：不要再调高（实测更慢）。
2. **长会话务必保持前缀稳定**：把系统提示/仓库上下文放在最前面且不变，之后的增量追加——靠前缀复用拿到 98% 的 TTFT 降幅。
3. **只在必要时用满 262K**：>128K 的 prefill 很慢，且占满显存。日常把工作集控制在 ~100K 以内更划算。
4. **压力更真实时（多 agent + 长上下文）考虑 SGLang**：它的并发扩展性明显更好。

---

## 6. 新增脚本

```
bench/ninfer_probe.py   并发/长上下文探针（--mode conc|longctx）
bench/prefix_test.py    前缀复用验证
bench/prefill_sweep.py  prefill 随上下文的扫描曲线（--targets，互不重叠前缀）
```
