# Qwen3.8-27B 三实例对比报告（RTX 5090D 32G / Windows / Docker）

> 生成：2026-09-24 ｜ 用途：本地并发复杂项目 coding，长上下文、长任务
> 数据：`data/*.jsonl`（**每场景 5 轮**）+ `summary.md` / `full-comparison.md` / `stability.md` ｜ 质量：`quality/quality.md` ｜ 计划：`PLAN.md`

---

## 0. 结论速览（TL;DR）

| 维度 | 胜者 | 说明 |
|---|---|---|
| 32K 上下文 prefill（TTFT） | **SGLang** | 4.0s vs NInfer-NVFP4 4.5s vs NInfer-MTP 11.3s |
| 128K 上下文 prefill（TTFT） | **SGLang ≈ NInfer-NVFP4** | 35.4s vs 34.0s（打平）；MTP 61.5s |
| 长上下文 decode 吞吐 | **NInfer-NVFP4** | 128K 下 145 tok/s vs SGLang 111 |
| 并发 2（独立长会话） | **SGLang** | 墙钟 9.9s vs 11.6s vs 25.5s |
| 上下文上限 | **NInfer** | 240,000 vs SGLang 163,840 |
| 冷启动 | **NInfer-MTP** | 106s vs 122s vs 157s |
| 显存占用 | NInfer-MTP 最低 | 27.9GB vs 30.8 / 30.75 GB（后两者贴近 32G 上限） |
| **能否直接用于 coding** | **SGLang** | NInfer 默认「只思考不产出」，见 §5 |
| 工具调用 | 三者平手 | 均正确产出 2 个 `get_weather` 调用 |

**推荐**：
1. **保持现状（SGLang）** 作为日常主力。它是三者中 f 综合最稳的：prefill 最快（128K 与 NVFP4 打平）、并发最好、开箱即用能产出正文。
2. 若**确实需要 >163K 上下文**，改用 **NInfer-NVFP4**（`Qwen3.8-27B-NInfer/docker-compose.yaml`），但必须先修两件事：
   - **思考预算**：服务端加 `--default-thinking-budget`（或 `--no-thinking`），或客户端固定 `reasoning_effort`，否则复杂 coding 任务返回空正文（§5）。
   - 接受更弱的并发表现（并发 2 独立长会话墙钟 11.6s，慢流 decode 跌到 ~38 tok/s）。
3. **不要用 NInfer-MTP（groupwise-int）** 跑长上下文：prefill 只有 NVFP4 的 ~40%，并发 2 长会话慢流跌到 19 tok/s，明显不适合你的用途。

---

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5090 D，32607 MiB（sm_120 / Blackwell） |
| 驱动 / CUDA | 591.86 / CUDA 13.1 |
| 内存 | 96 GB |
| 宿主 | Windows + Docker Desktop（WSL2 backend） |
| 模型 | Qwen3.8-27B（Qwen3.5 混合线性注意力，64 层 / 每 4 层 1 层全注意力，vocab 248320，max_pos 262144） |

## 2. 三个实例

| # | compose | 引擎 | 权重 | 关键参数 |
|---|---|---|---|---|
| ① | `sglang-qwen38/docker-compose.yml` | **SGLang**（`lmsysorg/sglang:qwen38-27b`） | NVFP4 LMHead4 + DSpark-NVFP4 drafter | `ctx 163840`、KV fp8、并发 2、mamba-cache 8、DSPARK block 7、`reasoning_effort=medium` |
| ② | `Qwen3.8-27B-NInfer/docker-compose-no-nvfp4.yaml` | **NInfer** | `qwen3_8_27b.ninfer`（groupwise-int） | `ctx 240000`、KV fp8、并发 2、host-KV 8 GiB / 8 states、`--spec mtp --draft-tokens 3` |
| ③ | `Qwen3.8-27B-NInfer/docker-compose.yaml` | **NInfer** | `qwen3_8_27b_nvfp4.ninfer`（NVFP4） | `ctx 262144`、KV fp8、并发 2、host-KV **16 GiB / 16 states**、`--default-thinking-budget 256`（见 §7 / `data/ninfer-optimization.md`） |

> **文件名说明（2026-09-25 重命名）**：`Qwen3.8-27B-NInfer/docker-compose.yaml` = **NVFP4**（已调优，推荐）；`Qwen3.8-27B-NInfer/docker-compose-no-nvfp4.yaml` = **groupwise-int**。两者镜像均为 `lqbing/ninfer`（本地已存在；它与 `ninfer:local` 为同一镜像 `sha256:d43d32b6`）。

> **模型来源**：
> - ① SGLang：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4` + `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`（HuggingFace）
> - ②③ NInfer：`neroued/Qwen3.8-27B-NInfer`（groupwise-int）/ `neroued/Qwen3.8-27B-nvfp4-NInfer`（NVFP4）；基座为 `Qwen/Qwen3.8-27B`，转换器 `github.com/Neroued/ninfer`

## 3. 方法（保证跨引擎可比）

- 统一 harness `bench/bench_unified.py`：仅依赖 `requests`，走 OpenAI `/v1/chat/completions`，自动从 `/v1/models` 发现模型名。
- **TTFT 口径修正**：两个引擎都先发一个仅含 `role` 的空 delta。若按"第一个含 choices 的 chunk"计时，TTFT 会假性接近 0。统一改为**第一个非空 token**（`content` 或 `reasoning_content` 或 `tool_calls`）。
- **冷缓存**：每轮 prompt 追加唯一 salt；`cache_n=0` 已验证前缀缓存未命中。
- 每场景 **5 轮**取中位数（2026-09-25 由 3 轮复测提升到 5 轮；旧数据留存于 `data/rounds3-backup/`）。并发场景用 `threading.Event` 同时放行，记录墙钟与每请求指标。
- 完整原始数据见 `data/*.jsonl`，汇总见 `data/summary.md` / `summary.csv`。

## 4. 性能结果

### 4.1 TTFT（s，越低越好）

| 场景 | SGLang | NInfer-NVFP4 | NInfer-MTP |
|---|---|---|---|
| 短 coding 生成 | 0.140 | **0.080** | 0.155 |
| 32K prefill | **4.03** | 4.53 | 11.33 |
| 32K 长生成 | **4.01** | 4.56 | 11.39 |
| **128K prefill** | 35.42 | **33.97** | 61.46 |
| 并发 2 短 | 0.217 | **0.141** | 0.242 |
| 并发 2×32K 独立 | **6.04** | 7.15 | 17.39 |

### 4.2 Prefill 速度（tok/s，越高越好；长 prompt 有效）

| 场景 | SGLang | NInfer-NVFP4 | NInfer-MTP |
|---|---|---|---|
| 32K | **8110** | 7202 | 2881 |
| 128K | 3686 | **3843** | 2124 |

### 4.3 Decode 速度（tok/s，越高越好）

| 场景 | SGLang | NInfer-NVFP4 | NInfer-MTP |
|---|---|---|---|
| 短 | 157.5 | **162.1** | **162.1** |
| 32K | 131.0 | **144.6** | 134.6 |
| 32K 长生成(1024) | 123.1 | 148.2 | **151.7** |
| 128K | 111.1 | **145.2** | 144.6 |
| 并发 2 短 | **164.2** | 147.9 | 118.2 |
| 并发 2×32K 独立（中位） | 89.3 | 88.7 | 71.8 |
| 并发 2×32K 独立（**最慢流**） | **44.0** | 37.9 | 19.2 |

### 4.4 并发 2 墙钟（s）

| 场景 | SGLang | NInfer-NVFP4 | NInfer-MTP |
|---|---|---|---|
| 并发 2 短 | **3.63** | 3.75 | 4.67 |
| 并发 2×32K 独立 | **9.93** | 11.59 | 25.46 |

## 5. 关键发现

### 5.1 Prefill 是长上下文的分水岭，权重格式决定 NInfer 成败
NInfer 两个变体的差值全在 prefill：**NVFP4 版 7202 tok/s（32K）vs groupwise-int 版 2881**——差 2.5 倍。128K 下 NVFP4 反而略胜 SGLang（3843 vs 3686）。
→ **要用 NInfer 就用 nvfp4 那份**；groupwise-int 那份在你的场景里没有意义。

### 5.2 并发 2「独立长会话」三者都会劣化，但程度不同
两个 32K 请求并发时，**一个流会被另一个流的 prefill 拖住**（这是"prefill 与 decode 争抢算力"，两个引擎都有）：
- SGLang：慢流 44 tok/s，墙钟 9.9s
- NInfer-NVFP4：慢流 38 tok/s，墙钟 11.6s
- NInfer-MTP：慢流 **19 tok/s**，墙钟 **25.5s**（≈串行 26.4s，并发几乎零收益）

→ 你的用例是"本地并发复杂项目 coding"，**并发质量 SGLang 最优**，NInfer-NVFP4 次之，MTP 最差。

### 5.3 【最重要】NInfer 默认「只思考、不产出」
质量抽查里，两个 NInfer 在 4 个任务中的 2 个（LRU、复杂设计）**正文为空**：模型把整个输出预算花在 `reasoning_content` 上，从不关闭思考。控制变量验证：加 `enable_thinking=false` / `reasoning_effort=none` 后立刻正常产出（content 3000+ 字，`finish=stop`）。
SGLang 因配置了 `reasoning_effort=medium`，不受影响。
→ **这是配置问题，不是能力问题**，但用 NInfer 必须处理，否则复杂 coding 返回空。详见 `quality/quality.md`。

### 5.4 工具调用可靠
三者都能正确产出 `get_weather(北京)` / `get_weather(上海)` 两个调用。

### 5.5 全部代码任务都撞输出上限
除 toolcall 外所有任务 `finish_reason=length`（含 SGLang）。该模型 coding 偏好长输出+长思考，客户端 `max_tokens` 建议 ≥ 4096。

## 6. 引擎与生态调研要点（外部资料，非本机实测）

### NInfer（`github.com/Neroued/ninfer`）
- 从零手写的 **C++20 + CUDA** 单卡引擎，Apache-2.0，2.4k★（2026-06-26 首发，迭代活跃）。
- **启动期固定 1–8 并发、无连续批处理、无抢占**，FIFO；准入即预留完整 prompt+输出 KV；过载返回 429/503。
- `--host-kv-mib` / `--host-state-slots` 是 **Device+pinned-Host 的 KV/State 检查点缓存**（降低跨轮重 prefill），**不是权重/计算卸载**。
- `.ninfer` 为自研 v3 容器格式；原生 MTP/DFlash/DFlash2 投机。
- 官方自评：Qwen3.8-27B nvfp4 MTP3 decode C=1 143.8 → C=2 267.6 → C=8 766.6 tok/s（与我们实测 C=1 的 ~145–162 同量级）。
- **官方文档自曝**：Code 类负载 15 样本中大量撞输出上限/重复循环（与 §5.3 现象一致）。
- Docker Hub `lqbing/ninfer` 为**第三方镜像**（非官方发布通道），内容未核验；上游仅支持 Linux（Windows 需 Docker/WSL2）。

### DSPARK（SGLang 用的投机解码）
- 已发表方法（arXiv:2607.05147，DeepSeek-AI + 北大，2026-07）。
- ⚠️ 两份 5090 社区资料互相矛盾，**值得你自己 A/B**：
  - `darksidewalker/qwen3.8-27b-sglang-dspark-blackwell` 的**默认是 DFlash2 而非 DSpark**（DFlash2 median 221 tok/s vs DSpark 126）；且称 **BF16 drafter 明显优于 NVFP4 drafter**。
  - `gittensor` 模型卡则称 **NVFP4 drafter 最好**（accept 2.904 / 180 tok/s）。
  - 你的 compose 用的是 **NVFP4 DSpark drafter**。若想进一步压榨 SGLang，可试：(a) 换 BF16 DSpark drafter，(b) 试 DFlash2。

## 7. Docker vs WSL 原生启动（补充实测）

> 问题：改用 WSL 原生启动（而非 Docker）是否带来性能差异？
> 方法：用**同一份本地编译二进制**（ninfer 源码工程内的 `build/apps/ninfer-serve`），分别在 Docker（`lqbing/ninfer:local`，CUDA 13.1）与 WSL Ubuntu-24.04 原生（CUDA 13.0）下，跑同一模型 / 同一参数 / 同一 harness。完整表见 `data/native-vs-docker.md`。

**结论：性能等价，Docker 没有计算惩罚。**

| 维度 | 结果 |
|---|---|
| **Prefill**（最可靠，纯算力） | 四组差异 ≤ ±6% 且方向不一致（NVFP4 −1.7%/−2.2%，groupwise +1.3%/+5.8%）→ 无系统性差异 |
| **Decode** | 波动由 **MTP 接受率**驱动而非环境：groupwise 原生 decode +22% 的同时接受率也从 ~50%→~63% |
| **冷启动** | 原生略快：NVFP4 权重读取 **201.3 vs 181.8 MiB/s**（总启动 109s vs 122s）；groupwise 基本持平 |
| **运维** | `wsl -e` 拉起的后台进程会被 WSL 回收 → 需 `setsid` 或由常驻 `wsl.exe` 持有（见 `bench/wsl_serve_fg.sh`） |

**建议**：不必为性能从 Docker 迁到原生。Docker 在依赖/隔离/复现/进程管理上更省心；原生仅省去容器层、冷启动略快。

## 8. 局限与未验证

- **单机单次测量**：每场景 5 轮中位数，未做多日重复；`longgen` 场景 SGLang 提前 `stop`（450–550 tok）而 NInfer 跑到上限，两者可比性略打折。逐轮波动见 `data/stability.md`。
- **`accepted length` / 投机收益**：NInfer 原生 timings 已记录在 `data/*.jsonl`（`draft_n` / `draft_n_accepted`）；SGLang 需从容器日志额外解析，本次未系统对比。
- **短 prompt 的 `prefill tok/s` 无意义**（固定开销主导），只看长 prompt 行。
- **WSL#40401**（Blackwell 在 WSL2 下约 16 GiB 隐形显存开销）属实，但**只在 RTX PRO 6000 上实测**，不能直接外推到 5090；本机未见明显异常。
- 未改动任何原始 compose / 模型文件；仅在 NInfer 目录用 `docker tag` 补了本地镜像 tag。

## 9. 复现

```powershell
# 在本项目根目录下执行；服务需已在 :30000 提供 OpenAI 兼容接口
python bench\bench_unified.py --label sglang --out data\sglang.jsonl --rounds 5 --longgen
python bench\quality_probe.py --label sglang --out quality\sglang.json
python bench\summarize.py --data-dir data --out data\summary.md
```

切换实例（三者都占 30000 端口，须逐个切换）：

```powershell
# SGLang（当前运行中）
docker start sglang-qwen38            # 恢复；docker stop 停止

# NInfer（在本项目根目录下）
docker compose -f docker-compose.yaml up -d              # NVFP4（推荐）
docker compose -f docker-compose-no-nvfp4.yaml up -d     # groupwise-int
docker compose -f <file> down                            # 停止
```

## 10. 证据来源

- 本机实测：`data/sglang.jsonl`、`data/ninfer-nvfp4.jsonl`、`data/ninfer-mtp.jsonl`（每场景 5 轮原始指标，含 NInfer 原生 `timings`）
- 质量原始输出：`quality/*.json`；思考预算验证：`bench/_test_thinking.py`
- 容器启动日志：SGLang `max_total_num_tokens=161106`、NInfer `capacity | KV 240,000 tokens, fp8`
- 外部调研：NInfer 仓库文档（serving.md / cli.md / performance）、arXiv:2607.05147、`darksidewalker` 与 `gittensor` 模型卡、microsoft/WSL#40401

## 11. 文件清单

```
Qwen3.8-27B-NInfer\
├─ README.md                     ← 项目说明（compose 对照 + 参数解释 + 使用要点）
├─ docker-compose.yaml           ← NVFP4（推荐，已调优）
├─ docker-compose-no-nvfp4.yaml  ← groupwise-int
├─ docs\
│  └─ comparison.md              ← 本报告（三方案横向对比）
├─ bench\
│  ├─ bench_unified.py           ← 跨引擎统一压测（主工具）
│  ├─ ninfer_probe.py            ← 并发/长上下文探针
│  ├─ prefix_test.py             ← 前缀复用验证
│  ├─ prefill_sweep.py           ← prefill 随上下文的扫描曲线
│  ├─ quality_probe.py           ← 质量抽查
│  ├─ summarize.py / native_vs_docker.py / stability.py / full_compare.py
│  └─ _probe_sse.py / _test_*.py / wsl_*.sh
├─ data\
│  ├─ {"sglang","ninfer-nvfp4","ninfer-mtp"}.jsonl            ← Docker 三实例
│  ├─ {"ninfer-nvfp4-native","ninfer-mtp-native"}.jsonl      ← WSL 原生
│  ├─ summary.md / summary.csv              ← Docker 三实例汇总
│  ├─ full-comparison.md / all-runs.csv     ← 全量对比（5 方案）+ 逐轮原始数据
│  ├─ stability.md                          ← 稳定性分析（5 轮的中位/极值/标准差/CV）
│  ├─ native-vs-docker.md                   ← Docker vs WSL 原生
│  ├─ ninfer-optimization.md                ← NInfer 调优报告
│  └─ rounds3-backup\                       ← 旧的 3 轮数据（复测前留存）
└─ quality\
   ├─ sglang.json / ninfer-nvfp4.json / ninfer-mtp.json
   └─ quality.md
```
