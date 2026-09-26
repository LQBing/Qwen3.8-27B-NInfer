# 质量对比报告：Qwen3.8-27B abliterated（uncensored）vs official

对比对象（同一 NInfer 服务配置：`--max-context 262144 --kv-capacity 262144 --max-concurrency 2 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --default-thinking-budget 256`，temperature 0.2）：

- `official` = `models/qwen3_8_27b_nvfp4.ninfer`（基座 Qwen3.8-27B）
- `abliterated` = `models/qwen3_8_27b_abliterated_v3.ninfer`（huihui-ai/Huihui-Qwen3.8-27B-abliterated）

数据来源：`quality/compare-side-by-side.md`（harness：`bench/quality_probe.py`）。性能（速度）前文已测且一致，本报告**只评答案质量**。

---

## (a) 逐任务判定表

| 任务 | 胜者 | official 分 /10 | abliterated 分 /10 | 一句话理由 |
|---|---|---|---|---|
| `lru` (max_tokens 1024) | **official** | 9 | 5 | official 完整交付 `get/put/peek` 且全程持锁；abliterated 被 `length` 截断，停在 `_pop_back` docstring 中间，`get/put/peek` 三个要求的方法一个都没写出来。 |
| `bugfix` (max_tokens 1024) | **tie** | 10 | 10 | 两者都精确指出 read-modify-write 非原子这一根因，并用 `threading.Lock` 包裹 `self.n += 1`，修复代码逐行一致。 |
| `toolcall` (max_tokens 512) | **tie** | 10 | 10 | 两者同块并行发出两条 `get_weather` 调用，参数 JSON 完全一致且均合法。 |
| `design` (max_tokens 1536) | **official** | 7 | 5 | 两者都被 `length` 截断；official 核心语义正确（含 `n > capacity` 拒绝、惰性 refill、key 隔离），abliterated 有 `n <= 0` 抛 ValueError 的边界 bug 且被截得更早。 |

---

## (b) 逐任务细节

### `lru` — official 胜（9 vs 5）

要求：`get/put/peek` 三方法、O(1) 时间复杂度、线程安全、类型注解、完整 docstring、直接输出代码。

**official（finish_reason=stop，完整 2209 字符）— 交付完整**
- 三个方法全部存在且均 O(1)，每个方法体都包在 `with self._lock:` 下（`threading.RLock`），线程安全真实成立：

  ```python
  with self._lock:
      if key not in self._data:
          return None
      self._data.move_to_end(key)
      return self._data[key]
  ```
  ```python
  with self._lock:
      if len(self._data) >= self.capacity:
          self._data.popitem(last=False)
      self._data[key] = value
  ```
- `peek` 正确地不改变使用顺序：`return self._data.get(key, None)`（无 `move_to_end`）。
- 类型注解齐全（`Hashable`/`Any`/`Optional`），docstring 完整（含 Args/Returns/Raises）。
- 无 bug、无幻觉 API；`OrderedDict.move_to_end` / `popitem(last=False)` 用法正确。
- 小瑕疵：`put` 的 docstring 首行被压进缩进（第 73 行 `       缓存值，并将该键标记为最近使用。`），属排版瑕疵，不影响代码可运行。

**abliterated（finish_reason=length，2788 字符处截断）— 未完成**
- 选择了更"工程化"的双向链表 + dict 方案，风格更精致（`__slots__`、`Generic[K, V]`、TypeVar），但 1024 token 预算不够，输出在 `_pop_back` 的 Returns 段中断：

  ```python
      def _pop_back(self) -> Optional[_Node[K, V]]:
          """Remove and return the least recently used node.

          Returns:
          ```
- 截断后缺失：`_pop_back` 方法体、`get`、`put`、`peek` —— 即 prompt 点名的三个公开方法一个都没写。
- 已写出部分本身无 bug：`_remove`/`_add_to_front` 的链表指针维护正确；`capacity < 0` 校验（允许 capacity=0，docstring 也声明 "non-negative"，自洽）。
- 结论：方案更优雅但**没写完**，作为"请实现 LRU"的答案是不完整的（truncation=yes）。

**冗余度**：abliterated 输出更长（2788 vs 2209 字符）却产出更少可运行代码 —— 双向链表方案在 1024 token 预算下明显 over-engineered。

### `bugfix` — 平局（10 vs 10）

要求：指出根因、最小修复、给出完整修复后代码；修复必须使增量原子化并解释原代码为何不安全。

两者（均 finish_reason=stop）几乎逐字等价：

- 根因解释（两者一致，中文）：

  > 根因：`incr()` 中的 `cur = self.n`、`cur += 1`、`self.n = cur` 是一个"读取—修改—写回"的复合操作，不是原子操作。多线程并发执行时，两个线程可能同时读到同一个 `self.n`，然后各自加 1 再写回，导致一次自增丢失。

- 修复代码（两者逐行相同）：

  ```python
  class Counter:
      def __init__(self):
          self.n = 0
          self._lock = threading.Lock()

      def incr(self):
          with self._lock:
              self.n += 1
  ```

- 修复确实使增量原子（`threading.Lock` 包裹读改写），解释正确且切中 GIL 下字节码交错导致的 lost update。
- 无 bug、无幻觉；无截断（official 651 字符、abliterated 528 字符，均 stop）。
- 唯一差异：official 多一句 `with self._lock:` 的作用说明（651 vs 528 字符），属冗余度微差，不足以分出胜负。

### `toolcall` — 平局（10 vs 10）

要求：对北京、上海各调用一次 `get_weather`。两者均 finish_reason=tool_calls、content 为空、同块发出 2 个调用。

实际发出的 arguments（原文 JSON 字符串，文件为 UTF-8 误读为 latin1 的显示，实际字节即"北京"/"上海"的 UTF-8）：

- official：
  ```json
  "arguments": "{\"city\":\"北京\"}"
  "arguments": "{\"city\":\"上海\"}"
  ```
- abliterated：
  ```json
  "arguments": "{\"city\":\"北京\"}"
  "arguments": "{\"city\":\"上海\"}"
  ```

- 两组调用**完全一致**：函数名、参数键（`city`）、城市值均正确；JSON 均合法（单键对象，无多余键、无格式错误）。
- 两者都合理地在同一 assistant 消息中并行发起两个独立调用（reasoning 中均提到"独立/parallel"）。
- 无冗余差异：content 都是空，仅 reasoning 语言不同（official 中文 171 字符、abliterated 英文 156 字符）。

### `design` — official 胜（7 vs 5）

要求：token bucket 限流器，capacity + refill rate 可设、`try_acquire(n)` 非阻塞、按 key 多租户隔离、单元 test、完整可运行代码。max_tokens 1536，两者均 finish_reason=length（都截断）。

**official（4784 字符）— 核心语义正确**
- 正确的惰性 refill 数学（注意其 refill 行有一个中文标点残留的小瑕疵，但数学正确）：

  ```python
  self._tokens = min(
      self.capacity,
      self._tokens + elapsed * self.refill_rate
  )
  ```
- `try_acquire` 在锁内 refill 后判断扣减，非阻塞、语义正确：

  ```python
  if self._tokens >= n:
      self._tokens -= n
      return True
  return False
  ```
- 多租户隔离正确实现：`TokenBucketRateLimiter` 按 key 惰性创建独立 bucket，支持 per-key `key_config`（`{"user1": (10, 2.0), ...}`）。
- 边界处理合理：`n < 0` 抛 ValueError、capacity/refill_rate 负数校验；隐含地 `n > capacity` 时返回 False（因 tokens ≤ capacity）。
- 缺陷：
  1. 被 `length` 截断 —— 开头列的 bullet 里承诺的 `unittest` 和 `__main__` demo 都没交付（`import unittest` 已写但无测试类）；
  2. 截断处 `TokenBucketRateLimiterWithConfig.__init__` 中 `self._lock = threading.Lock`（漏了括号，未调用），且该冗余类本身未写完；
  3. `_refill` docstring 有中文残片（"根据时间流逝...注意：调用方持有 self._lock"）。

**abliterated（4573 字符）— 设计更讲究，但有一个边界 bug**
- 设计说明更专业：lazy refill、两级锁（bucket 创建锁 + 每桶独立锁）、per-key 锁避免多租户互相阻塞、clock 可注入便于测试 —— 这些都在输出中明确写出。
- refill 数学与锁内扣减逻辑正确：

  ```python
  with self._lock:
      now = self.clock()
      self._refill_locked(now)
      if self._tokens >= n:
          self._tokens -= n
          return True
      return False
  ```
- **边界 bug**：`try_acquire` 对 `n <= 0` 抛 ValueError：

  ```python
  if n <= 0:
      raise ValueError(f"n must be > 0, got {n}")
  ```

  `try_acquire(0)` 语义上应恒为 True（拿 0 个令牌总该成功），这里把合法请求变成异常 —— 一个真实的 API 误用/语义瑕疵（对比 official 用 `n < 0` 才拒绝，`n=0` 走 `self._tokens >= 0` 返回 True，正确）。
- 多租户隔离正确：`_get_bucket` 双检查加锁创建，每 key 独立桶；额外提供 `reset`。
- 被 `length` 截断：单元 test 部分在分隔注释 `# ... python -m unittest -v` 处中断，**同样没交付任何测试**；`available` property docstring 有中文残片（"（近似值，触发一次..."）。
- 冗余度：4573 字符中约 1/4 是 Markdown 设计说明（"设计要点"小节），官方则直接进代码 —— 在 1536 token 预算下，abliterated 把更多预算花在散文上，留给测试的空间更少。

---

## (c) 总体判定

**总分：official 36/40 vs abliterated 30/40，差距 6 分（官方领先 15%）。**

| 任务 | official | abliterated |
|---|---|---|
| lru | 9 | 5 |
| bugfix | 10 | 10 |
| toolcall | 10 | 10 |
| design | 7 | 5 |
| **合计** | **36** | **30** |

判定：**存在可测量的质量下降，但幅度有限且机制单一。**

- 两个"小而完整"的任务（bugfix、toolcall）上 abliterated 与官方**完全持平**，无幻觉、无 API 误用、工具调用格式分毫不差 —— 说明 abliteration 没有破坏指令遵循与代码语法能力。
- 差距全部来自两个被 `length` 截断的任务，且方向一致：**abliterated 在有限 token 预算下"写得更大"（更工程化的方案 + 更多散文铺垫），导致核心交付物（三个公开方法 / 单元测试）没能在预算内写完**：
  - lru：选双向链表而非 `OrderedDict`，1024 token 不够 → 公开 API 缺失（5 分）；
  - design：先写 Markdown 设计说明再写代码，1536 token 不够 → 测试缺失且边界处理更差（`n <= 0` 抛异常，5 分）。
- 若把截断因素剥离（给足 max_tokens），abliterated 的 design 方案质量（per-key 锁、clock 注入）甚至可能反超官方 —— 目前的差距更多是"预算效率/风格"问题，而非"能力退化"。

一句话：abliteration 没有让模型"变笨"，但让它**变啰嗦、爱铺陈**，在 token 受限场景下交付完整度的期望值下降；对短任务无感，对长任务有实际可测的损失（本样本 6/40）。

---

## (d) Caveats（注意事项）

1. **finish_reason 不对称（最重要的混杂因素）**：
   | 任务 | official | abliterated |
   |---|---|---|
   | lru | stop（完整） | **length（1024 截断）** |
   | bugfix | stop | stop |
   | toolcall | tool_calls | tool_calls |
   | design | **length（1536 截断）** | **length（1536 截断）** |
   lru 任务上 official 完整而 abliterated 被截断，是 lru 4 分差的主要来源 —— 严格说这是在比较"完成版 vs 半成品"，不完全公平。
2. **design 任务双方都被截断**，所以该任务比较的是"同样不完整下谁的核心部分更对"，而非"完整答案"对比。
3. **样本量 = 1**：每个任务每模型仅一次采样（temperature 0.2 非 0），单次结果不能排除采样噪声；lru 上 abliterated 若恰好选 `OrderedDict` 方案就不会被截断。
4. **编码显示问题**：源对照文件以 UTF-8 写入但显示为 latin-1 乱码（如 `åäº¬` = 北京），本报告的引文已按正确解码转写；不影响内容判断。
5. **思考预算固定 256**（`--default-thinking-budget 256`）：所有 reasoning 块都在同一预算下截断（末尾均有 "Considering the limited time..." 标记），两模型思考深度可比。
6. **冗余度/风格 ≠ 质量**：本报告按"是否满足 prompt + 正确性 + 完整性"打分；若业务场景偏好更详尽的设计说明，design 任务上 abliterated 的"设计要点"小节是加分而非减分。
7. 未做 jailbreak/有害内容探测（本样本为纯代码任务）；abliteration 的主要风险场景（角色扮演、格式放飞）未覆盖。
