# Qwen3.8-27B-NInfer

REF: `https://github.com/Neroued/ninfer`

本地用 NInfer 引擎（`ninfer-serve`）跑 Qwen3.8-27B 的两份部署配置。

---

## 三份 compose

| 文件 | 权重 | 说明 |
|---|---|---|
| **`docker-compose.yaml`** | `models/qwen3_8_27b_nvfp4.ninfer` | **NVFP4，推荐主力**。已调优：`ctx 262144`、host-KV 16 GiB、思考预算 256 |
| `docker-compose-no-nvfp4.yaml` | `models/qwen3_8_27b.ninfer`（groupwise-int） | 长上下文 prefill 明显更慢（32K 仅 ~2800 tok/s，约 NVFP4 的 40%），**不推荐** |
| `docker-compose-abliterated.yaml` | `models/qwen3_8_27b_abliterated_nvfp4.ninfer` | **无护栏版**（abliterated），与 NVFP4 版同配置。实测性能与官方版无差异 |

> 三份都占 **30000 端口**且共用同一张 GPU，**互斥**，需逐个切换（先 `down` 再 `up`）。
> 三者服务名均为 `qwen3.8-27b`，对上层客户端（litellm/opencode）可无感互换。
> 工件完整清单（含校验和、容器版本、来源）见 **`models/README.md`**。

## 模型与下载源

两个 `.ninfer` 工件均为 **NInfer 官方发布**（HuggingFace）。基座 `Qwen/Qwen3.8-27B`，转换器 `github.com/Neroued/ninfer`，格式为 **v3 `.ninfer` 容器**。

| 本地文件 | HF 仓库 | 大小 | 权重血统 |
|---|---|---|---|
| `models/qwen3_8_27b_nvfp4.ninfer` | [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | 22.09 GB | mixed FP8/NVFP4，源自 [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) |
| `models/qwen3_8_27b.ninfer` | [`neroued/Qwen3.8-27B-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.03 GB | `groupwise-int` |
| `models/qwen3_8_27b_abliterated_nvfp4.ninfer` | [`lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4`](https://huggingface.co/lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4)（经 v2→v3 升级） | 20.02 GB | **无护栏**，基座 [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) |
| `models/qwen3_8_27b_abliterated_original.ninfer` | 同上（原始下载件，未升级） | 20.02 GB | 同上；container **v2**，当前运行时**不可直接加载** |

### 下载命令

```bash
# 0) 前置：安装 HF 官方 CLI（已装可跳过）
pip install -U "huggingface_hub[cli]"

# 1) 必须进项目目录：compose 用 ./models 挂载
cd <项目根目录>    # 本 README / 两份 compose 所在目录

# 2) 下载（推荐只下 NVFP4；两者可共存）
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models   # NVFP4
hf download neroued/Qwen3.8-27B-NInfer      qwen3_8_27b.ninfer      --local-dir models   # groupwise-int
```

**命令拆解** `hf download <仓库ID> [<文件...>] --local-dir <目录>`：

| 部分 | 说明 |
|---|---|
| `<仓库ID>` | HF 仓库，如 `neroued/Qwen3.8-27B-nvfp4-NInfer` |
| `[<文件...>]` | 只下指定文件（省略 = 整仓）。这里指定单个 `.ninfer`，避免拉多余文件 |
| `--local-dir models` | 落盘目录。**必须是 `models/`** —— compose 挂载 `./models:/models:ro`，容器内按 `/models/qwen3_8_27b_nvfp4.ninfer` 读取 |

> 下载会在 `models/.cache/huggingface/` 留下元数据（本目录已有），属正常现象；换仓库/重下时可删。

### 镜像与加速（HF 直连不稳时）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 镜像；本机实测可行，偶发超时重跑即续
export HF_HUB_ENABLE_HF_TRANSFER=1         # 多线程加速（需 pip install hf_transfer）
```
Windows PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`。

### 断点续传与校验

- `hf download` **默认断点续传**：中断后重跑同一条命令，从断点继续。
- 官方校验和（来源 `model-cards/<repo>/SHA256SUMS`）：

| 文件 | 字节数 | SHA256 |
|---|---|---|
| `qwen3_8_27b_nvfp4.ninfer` | 23,719,715,844 | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| `qwen3_8_27b.ninfer` | 20,437,521,664 | `81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da` |

```powershell
# 校核（PowerShell）
Get-FileHash .\models\qwen3_8_27b_nvfp4.ninfer -Algorithm SHA256
```

### 备选下载方式

```bash
huggingface-cli download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
```
```python
from huggingface_hub import hf_hub_download
hf_hub_download("neroued/Qwen3.8-27B-nvfp4-NInfer", "qwen3_8_27b_nvfp4.ninfer", local_dir="models")
```
ModelScope 亦收录基座镜像：`modelscope.cn/models/Qwen/Qwen3.8-27B`。

**磁盘**：只下 NVFP4 ≈ **22 GB**；两个都下 ≈ **41 GB**（另加 HF 缓存开销）。

### 工件元信息（来自 `artifact-manifest.json`）

- `architecture`: `Qwen3_5ForCausalLM`；`container_version`: **3**；`components`: `[text, vision, mtp, dflash2]`
- `runtime.cuda_architecture`: **`sm_120a`**（为 RTX 5090 编译）
- 基座 `Qwen/Qwen3.8-27B` rev `1d4bf0f2…`（Apache-2.0）；NVFP4 版量化源 `unsloth/Qwen3.8-27B-NVFP4`
- 权威来源：ninfer 仓库 `README.md` + `model-cards/Qwen3.8-27B*-NInfer/`

> 对照（SGLang 方案，本项目未用）：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4` + `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`，见 `docs/comparison.md`。

## 无护栏版（abliterated）

基座 [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)，经 `lyf` 用**官方转换器**产出 `.ninfer`，再由本项目的升级工具转成当前运行时可用格式：

```
huihui BF16 ──┐
              ├─► NInfer 转换器 ─► container v2 (.ninfer) ─► upgrade_ninfer_v2_to_v3.py ─► v3 ✅
ModelOpt NVFP4┘
```

- **转换器是公开的**：`github.com/Neroued/ninfer` 的 `tools/convert/`（`pipeline.py`、`qwen3_5.py`、`sources/safetensors.py`、`quantization/`）。输入需两份权重：BF16 源 + 单独量化好的 NVFP4 checkpoint（ModelOpt 校准），且仅支持 Linux。
- **下载到的是 container v2**，当前运行时（`lqbing/ninfer:latest`）只认 v3，会直接拒绝；必须先升级。升级后权重**逐字节不变**（仅容器框架变化 + 换装维护版 chat template）。
- 启动：`docker compose -f docker-compose-abliterated.yaml up -d`

**实测（同配置、同机）**：性能与官方版**无差异**；质量官方 36/40 vs 无护栏 30/40，差距源于"更啰嗦、token 预算效率低"导致的长任务截断，非能力退化。详见 `data/abliterated-vs-official.md` 与 `data/quality-gap-abliterated-vs-official.md`。

**两个坑**：
1. 纯文本场景**别开 `--vision`** —— 额外占 ~0.6 GiB 固定显存，prefill/decode 约腰斩。
2. **别改 `--model-id`** —— 上层 litellm/opencode 按 `qwen3.8-27b` 调用，改名会导致 agent 工具链报 `model not found`。

## 当前配置（`docker-compose.yaml`）

```bash
ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0 --port 30000 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 2 \
  --kv-dtype fp8 --device-state-slots 2 --host-state-slots 16 --host-kv-mib 16384 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking --default-thinking-budget 256
```

| 参数 | 作用 |
|---|---|
| `--max-context 262144` | 单序列逻辑上下文上限。**模型的硬顶**（`max_position_embeddings`），设更大 NInfer 直接拒绝启动 |
| `--kv-capacity 262144` | 共享 KV 池容量（2 个序列共用） |
| `--kv-dtype fp8` | KV 精度（≈32 KB/token）；`nvfp4` 可省一半显存但属量化 KV |
| `--max-concurrency 2` | 并发上限。**实测 3 反而更差**，锁 2 |
| `--device-state-slots 2` | 活跃 lane 之外的额外 Device 检查点槽位（总容量 = 2+2） |
| `--host-state-slots 16` / `--host-kv-mib 16384` | **pinned 主机内存**的 State/KV 缓存（不占显存），用于显存吃紧时保留不活跃前缀 |
| `--spec mtp --draft-tokens 3 --lm-head-draft` | MTP 投机解码，每轮 3 草稿 |
| `--preserve-thinking --default-thinking-budget 256` | 保留思考 + 思考 token 上限（≈ SGLang `reasoning_effort=medium`） |

## 启动 / 停止

```powershell
cd <项目根目录>    # 本 README / 两份 compose 所在目录
docker compose -f docker-compose.yaml up -d            # 推荐配置
docker compose -f docker-compose-no-nvfp4.yaml up -d   # groupwise-int
docker compose -f docker-compose.yaml down             # 停止
```

> ⚠️ `docker compose` 必须在**本目录**执行（或把 `-f` 指向本目录的绝对路径），否则会因找不到文件而静默失败。

服务地址：`http://localhost:30000`（OpenAI/Anthropic 兼容：`/v1/chat/completions`、`/v1/models`、`/v1/messages`、`/v1/responses`）。

启动画像（大约）：`weights 19.7 GiB | runtime 9.48 GiB | KV 262,144 fp8 | free ~0.6 GiB | host 16 states, 16 GiB KV`

## 使用要点（踩过的坑）

1. **思考预算**：`--preserve-thinking` 开着但**不设** `--default-thinking-budget` 时，思考无上限且**计入 `max_tokens`** → coding 任务会把预算烧在推理上、**正文为空**。已设 256 解决。
   客户端也可覆盖：`reasoning_effort: "medium"`（有效值仅 `none`/`low`/`medium`；`minimal`/`high` 会被模板拒绝），或 `enable_thinking: false` 直接不思考。
2. **工具调用无需额外参数**：NInfer 没有 `--tool-call-parser` 之类的开关，工具由 `.ninfer` 内置 chat template 处理；opencode 用 `tool_choice: "auto"` 即可（不支持 `strict` / 具名 tool_choice）。
3. **上下文上限 262,144**：这是模型限制，调不上去。KV「池」可单独放大（nvfp4 KV 可到 ~491K）但只服务并发。
4. **并发 2 是甜点**：3 并发实测更慢（长会话墙钟 36.5s vs 11.7s）。
5. **长会话保持前缀稳定**：同一前缀再次请求 TTFT 可从 4.5s 降到 0.07s（−98%）。

## 数据与文档（本目录自带，无需跨项目引用）

| 目录 | 内容 |
|---|---|
| `bench/` | 压测与探针脚本：`bench_unified.py`（跨引擎压测）、`ninfer_probe.py`（并发/长上下文探针）、`prefix_test.py`（前缀复用）、`prefill_sweep.py`（prefill 曲线）、`quality_probe.py`、`summarize.py` 等 |
| `data/` | 逐轮原始数据 `*.jsonl`；汇总 `summary.md`/`summary.csv`、`full-comparison.md`、`all-runs.csv`；`stability.md`（5 轮稳定性）、`native-vs-docker.md`（Docker vs WSL 原生）、`ninfer-optimization.md`（**调优报告**）；**无护栏相关**：`abliterated-vs-official.md`（性能对照）、`quality-gap-abliterated-vs-official.md`（质量差距）、`ninfer-abliterated.jsonl`/`ninfer-abliterated-vision.jsonl`/`ninfer-official.jsonl` |
| `quality/` | `quality.md`（质量抽查 + 思考预算修复验证）、原始输出 `sglang.json`/`ninfer-*.json`；**无护栏相关**：`ninfer-abliterated.json`、`ninfer-official.json`、`compare-side-by-side.md`（解码后可读的并排对照） |
| `tools/` | **NInfer 转换/升级工具链**：`upgrade_ninfer_v2_to_v3.py`（v2→v3 升级）、`chat_templates/`（官方 Qwen chat template）、`do_upgrade.sh`（升级复现脚本，WSL 运行）、`compare_abliterated_artifacts.py`（工件逐字节对比） |
| `docs/` | `comparison.md`（**三方案横向对比报告**） |
| `logs/` | WSL 原生启动日志（`native-*.log`，含加载耗时/速率）；本次新增 `bench-*.log`/`quality-*.log`（基准原始 stdout）、`hf-download-abliterated.*.log` |

**推荐阅读顺序**：`data/ninfer-optimization.md`（本项目调优）→ `docs/comparison.md`（与 SGLang 横评）→ `quality/quality.md`（思考/工具调用验证）→ `models/README.md`（工件清单）→ `data/abliterated-vs-official.md`（无护栏横评）。
