# 质量抽查 — 三个实例对同一批 coding 任务的原始输出对比

> 数据来源：`quality/{sglang,ninfer-nvfp4,ninfer-mtp}.json`（原始全文，可自行复核）
> 采集脚本：`bench/quality_probe.py`（stream 采集 content / reasoning_content / tool_calls）

## 1. 方法

4 个任务，覆盖用户的核心场景（写代码 / 修 bug / 工具调用 / 复杂多步设计）：

| id | 类别 | max_tokens | 说明 |
|---|---|---|---|
| `lru` | code-gen | 1024 | 线程安全 LRU 缓存类 |
| `bugfix` | bugfix | 1024 | 指出并发缺陷并最小修复 |
| `toolcall` | tool-use | 512 | 调用 `get_weather` 查北京/上海 |
| `design` | complex | 1536 | 线程安全令牌桶限流器 + 单测 |

`temperature=0.2`，流式，`reasoning_content` 与 `content` 分开累计。

## 2. 结果

| 任务 | 指标 | SGLang | NInfer-NVFP4 | NInfer-MTP |
|---|---|---|---|---|
| **lru** | finish | length | length | length |
| | content 字数 | **2261** | **0** ❌ | **0** ❌ |
| | reasoning 字数 | 1078 | 4229 | 3894 |
| **bugfix** | finish | length | stop | stop |
| | content 字数 | 1235 | 577 | 612 |
| | reasoning 字数 | 1090 | 1891 | 1592 |
| **toolcall** | finish | tool_calls | tool_calls | tool_calls |
| | 工具调用数 | **2** ✅ | **2** ✅ | **2** ✅ |
| | 解析结果 | get_weather(北京/上海) | get_weather(北京/上海) | get_weather(北京/上海) |
| **design** | finish | length | length | length |
| | content 字数 | **4149** | **0** ❌ | **0** ❌ |
| | reasoning 字数 | 607 | 5781 | 6402 |

## 3. 关键发现

### 3.1 【最重要】NInfer 默认"只思考、不产出"
两个 NInfer 实例在 `lru` 与 `design` 上 **content 字数为 0**：模型把 1024 / 1536 的全部输出预算花在 `reasoning_content` 上（4229 / 5781 字），**从未关闭思考、没有输出正文**。原始 reasoning 尾部仍是 `from collections import OrderedDict...` 之类的草稿，属"想到一半撞上限"。

这不是权重问题（NVFP4 与 groupwise-int 两个 NInfer 变体表现一致），而是 **NInfer 的思考策略默认无界**：compose 里设了 `--preserve-thinking` 但**没有设 `--default-thinking-budget`**，导致思考不收敛。

已在 NInfer-NVFP4 上做控制变量验证（`bench/_test_thinking.py`，max_tokens=1024）：

| 请求侧设置 | finish | content 字数 | reasoning 字数 |
|---|---|---|---|
| 默认 | length | **0** | 4131 |
| `enable_thinking=false` | stop | **3015** | 0 |
| `reasoning_effort=none` | stop | **3201** | 0 |
| `reasoning_effort=low` | length | 275 | 2631 |
| `chat_template_kwargs.enable_thinking=false` | stop | **3773** | 0 |

**结论**：NInfer 需在请求侧关闭/限制思考（`enable_thinking=false` 或 `reasoning_effort=none`），或在服务端设 `--default-thinking-budget` / `--no-thinking`，否则 coding 任务会拿不到代码。SGLang 因 compose 里设了 `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`，所以能正常产出正文。

### 3.2 工具调用三者都正确
`toolcall` 上三个实例都产出 **2 个** `get_weather` 调用（北京、上海），流式 tool_call 分片可被标准解析器正确拼接。NInfer 虽不执行工具、也不做约束解码，但**解析出的 JSON 参数正确**。

### 3.3 `bugfix` 三者都能正常完成
唯一一个 NInfer 也能正常收敛并输出正文的任务（`finish=stop`）。说明问题不是"模型不会做"，而是**思考预算/收敛**。

### 3.4 全部代码任务都撞了输出上限
除 toolcall 外，所有任务 `finish_reason=length`——包括 SGLang。**该模型在 coding 任务上偏好长输出 + 长思考**，客户端 `max_tokens` 需要给足（建议 ≥ 4096），否则正文会被截断。

## 4. 对用户的意义

- 若采用 NInfer：**必须**先解决思考预算问题（否则复杂 coding 任务会返回空正文），建议服务端加 `--default-thinking-budget` 或客户端固定 `reasoning_effort`。
- 若采用 SGLang：当前 `reasoning_effort=medium` 已能稳定产出正文，仅需注意 `max_tokens` 给足。
- 三者的工具调用能力都可用，满足 agent 化 coding 的前置条件。

## 5. 修复验证：服务端 `--default-thinking-budget`（2026-09-25 补充）

在 NInfer-NVFP4 上以同一 LRU 提示词、**请求侧不加任何覆盖**，仅改服务端 budget 重启对比（脚本 `bench/_test_thinking2.py`）：

| 服务端 budget | 客户端设置 | max_tokens | finish | content 字数 | reasoning 字数 |
|---|---|---|---|---|---|
| 未设 | 默认 | 1024 | length | **0** | 3894 |
| 1024 | 默认 | 1024 | length | **0** | 3894 |
| 1024 | 默认 | 4096 | stop | 2121 | 4498 |
| **512** | 默认 | 1024 | length | **1417** | 2041 |
| **512** | 默认 | 4096 | stop | **2902** | 2074 |
| 512 | `reasoning_effort=none` | 1024 | stop | 2945 | 0 |
| 512 | `enable_thinking=false` | 1024 | stop | **3553** | 0 |

**要点**：
- `--default-thinking-budget` 单位是 **token**，且**思考 token 与正文共享客户端的 `max_tokens`**。
- budget=1024 时若客户端 `max_tokens=1024`，思考吃满预算 → 仍然没有正文；**budget=512 才留出正文空间**。
- **最稳做法（coding 场景）：请求侧 `enable_thinking=false`（或 `reasoning_effort=none`）** —— 正文最长（3553 / 2945 字）、token 最省（672–836）、且 `finish=stop` 自然收敛。
- 次选：服务端 `--default-thinking-budget 512`，并保证客户端 `max_tokens ≥ 2048`。

## 6. 配置建议：NInfer 对齐 SGLang 的 `reasoning_effort=medium`（2026-09-25 实测）

**不用关闭思考。** SGLang 用 `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`；NInfer **没有**这个参数（CLI 只有 `--default-thinking-budget` / `--no-thinking` / `--preserve-thinking`），所以用**思考 token 预算**达成等效。

同 LRU 提示词对照：

| 配置 | max_tokens | finish | content | reasoning |
|---|---|---|---|---|
| NInfer 默认（**无预算**） | 4096 | length | **0** | 16311 |
| **NInfer + `--default-thinking-budget 192`**（默认请求） | 1024 | length | 2437 | 770 |
| **NInfer + `--default-thinking-budget 256`** | 1024 | **stop** | **2245** | 1035 |
| 同上 | 4096 | **stop** | 2120 | 1041 |
| **SGLang `reasoning_effort=medium`（对照）** | 1024 | length | 2261 | 1078 |

→ **`--default-thinking-budget 256` 让 NInfer 的思考/正文分配与 SGLang `medium` 基本一致**（reasoning 1035 vs 1078，content 2245 vs 2261），且变为自然 `stop`。

**有效档位**：`reasoning_effort` 只接受 `none` / `low` / `medium`；`minimal` / `high` 会被 chat_template 拒绝（HTTP 400 `invalid_prompt`）。

已应用到 `Qwen3.8-27B-NInfer/docker-compose.yaml`（追加 `--default-thinking-budget 256`）。回退：删掉该参数即可。

### 工具调用（opencode 接入）

- **不需要** `--tool-call-parser` 之类的参数 —— NInfer 的 CLI 里**根本没有** tool/parser 开关（`--help` 仅有 `--chat-template FILE`）。工具调用由 `.ninfer` 内置的 chat_template 处理。
- 实测（`bench/_test_tools.py`）：单轮流式返回 **2 个正确 `tool_calls`**；回传 tool 结果后能给出最终文字回答（`finish=stop`）；`tool_choice: "auto"` 被接受。
- 限制：NInfer **不执行工具**（由客户端执行——正是 opencode 期望的模式）；拒绝 `strict:true` 与 required/具名 `tool_choice`；无 JSON-schema 约束解码。
