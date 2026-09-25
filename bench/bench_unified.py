# -*- coding: utf-8 -*-
"""
bench_unified.py —— 跨引擎统一 benchmark（OpenAI 兼容接口）

同一套场景 / 指标口径，可原样跑 SGLang 或 NInfer：

    python bench_unified.py --label sglang --out data\\sglang.jsonl
    python bench_unified.py --label ninfer --out data\\ninfer.jsonl

设计要点：
  - 仅依赖 requests + 标准库（无 openai SDK / asyncio / tqdm）。
  - 每个场景输出一行 JSON，追加到 --out，同时打印到 stdout。
  - 冷启动公平性：每轮给 prompt 追加唯一 salt（uuid），使前缀缓存无法"作弊"；
    并 best-effort 调用 POST /flush_cache（SGLang 有该非标准端点，NInfer 返回 404
    也直接忽略，不影响流程）。
  - 长上下文 prompt 先发 max_tokens=1 探针，读 usage.prompt_tokens 估算
    chars/token，再按目标长度线性切片构造，避免硬编码模型名或 tokenizer。
  - 指标口径（TTFT / decode_tok_s / total_s）与 bench.py、concurrent_bench.py
    一致；唯一修正是 TTFT 改记"第一个非空 token"（NInfer 会先发空 role chunk，
    见 stream_once 说明），以保证 SGLang / NInfer 之间可比。
  - 模型名通过 /v1/models 自动发现，绝不硬编码。
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid

import requests

# ---------------------------------------------------------------- 常量

DEFAULT_BASE = "http://localhost:30000"

# 场景清单（按输出顺序）；longctx_32k_longgen 仅在 --longgen 或显式 --scenarios 时运行
ALL_SCENARIOS = [
    "short_coding_gen",
    "longctx_32k_prefill",
    "longctx_32k_longgen",
    "longctx_128k_prefill",
    "conc2_short",
    "conc2_long_distinct",
]
# 默认（不带 --longgen）运行的场景
BASE_SCENARIOS = [s for s in ALL_SCENARIOS if s != "longctx_32k_longgen"]

# 短 prompt：与参考脚本保持完全一致
CODE_TASK = """你是一个资深 Python 工程师。请实现一个 LRU 缓存类，要求：
1. 支持 get(key) / put(key, value) / peek(key) 三个方法
2. 线程安全
3. 时间复杂度 O(1)
4. 用类型注解，写完整的 docstring
请直接输出代码，不要解释。"""

# 生成合成 "代码库" 时的默认字符量：约 3MB，足够覆盖 128K token 目标
DEFAULT_LONG_BODY_CHARS = 3_000_000


# ---------------------------------------------------------------- prompt 构造
# _func / long_code_prompt 的逻辑直接复用自 benchmark/bench.py 与 concurrent_bench.py

def _func(i: int) -> str:
    """生成第 i 个不同的代码函数，保证长上下文内容多样（避免退化 token）。"""
    name = f"process_record_{i:04d}"
    n = (i * 37) % 30 + 8  # 每个函数 8-37 行
    lines = [f"def {name}(records, config=None):", '    """Handle the data pipeline for chunk %d."""' % i]
    for j in range(n):
        k = (i * 31 + j * 17) % 1000
        if j % 5 == 0:
            lines.append(f"    # step {j}: normalize the {k}th field of each record")
        elif j % 5 == 1:
            lines.append(f"    acc_{j} = sum(x.get('value', 0) * {k} for x in records)")
        elif j % 5 == 2:
            lines.append(f"    if acc_{j} > {k}:")
            lines.append(f"        acc_{j} -= len(records)")
        elif j % 5 == 3:
            lines.append(f"    buf_{j} = [x for x in records if x.get('id') % {k + 1} == 0]")
        else:
            lines.append(f"    stats_{j} = {{'mean': acc_{j} / max(len(records), 1), 'bucket': {k}}}")
    lines.append(f"    return locals().get('stats_{n - 1}', {{'done': True}})")
    return "\n".join(lines) + "\n"


def build_long_body(min_chars: int = DEFAULT_LONG_BODY_CHARS) -> str:
    """生成足够长的合成代码库文本（至少 min_chars 个字符），供后续按需切片。"""
    parts = [
        "### 项目代码库（仅供上下文参考，包含多个模块）\n",
        "```python\n",
    ]
    size = sum(len(p) for p in parts)
    m = 0
    while size < min_chars:
        head = f"\n# ============ module_{m}: data ingestion utilities ============\n"
        parts.append(head)
        size += len(head)
        for i in range(m * 20, m * 20 + 20):
            fn = _func(i)
            parts.append(fn)
            size += len(fn)
        m += 1
    parts.append("```\n\n")
    return "".join(parts)


def slice_long_prompt(body: str, target_tokens: int, chars_per_token: float, offset: int = 0) -> str:
    """按 chars/token 比例线性外推到 target_tokens 长度并切片。

    offset 让不同请求从 body 不同位置起始 → 前缀不同，模拟多个独立会话。
    """
    cut = int(target_tokens * chars_per_token * 1.02)
    avail = max(len(body) - offset, 0)
    cut = min(cut, avail)
    return body[offset:offset + cut]


# ---------------------------------------------------------------- 网络基础

def chat_url(base: str) -> str:
    return f"{base}/v1/chat/completions"


def discover_model(base: str, explicit) -> str:
    """优先 GET /v1/models 取 data[0].id；失败则用 --model；再兜底 'default'。"""
    try:
        r = requests.get(f"{base}/v1/models", timeout=15)
        r.raise_for_status()
        data = r.json().get("data") or []
        if data and data[0].get("id"):
            return data[0]["id"]
    except Exception as e:  # noqa: BLE001
        print(f"[info] model discovery failed: {type(e).__name__}: {e}", file=sys.stderr)
    if explicit:
        return explicit
    return "default"


_flush_warned = {"done": False}


def flush_cache(base: str) -> None:
    """best-effort 清缓存。SGLang 有 /flush_cache；NInfer 无（404）也必须继续。"""
    try:
        requests.post(f"{base}/flush_cache", timeout=10)
    except Exception as e:  # noqa: BLE001
        if not _flush_warned["done"]:
            _flush_warned["done"] = True
            print(f"[info] /flush_cache unavailable ({type(e).__name__}); continuing", file=sys.stderr)


def calibrate_chars_per_token(base: str, model: str, sample_body: str) -> float:
    """max_tokens=1 探针读 usage.prompt_tokens，估算 chars/token。

    探针本身也带 salt，避免污染被测 prompt 的前缀缓存。
    """
    probe_body = sample_body + f"\n# probe-salt: {uuid.uuid4()}\n"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": probe_body}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    r = requests.post(chat_url(base), json=payload, timeout=300)
    r.raise_for_status()
    p_tokens = r.json()["usage"]["prompt_tokens"]
    return len(probe_body) / max(p_tokens, 1)


def probe_prompt_tokens(base: str, model: str, text: str):
    """对指定文本发 max_tokens=1 探针，返回真实 prompt_tokens（失败返回 None）。"""
    body = text + f"\n# probe-salt: {uuid.uuid4()}\n"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": body}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    r = requests.post(chat_url(base), json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("usage", {}).get("prompt_tokens")


# ---------------------------------------------------------------- 流式请求（单次）

def stream_once(base: str, model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    """一次流式请求，返回单轮指标（不含 'round' 字段）。

    解析逻辑与 bench.py / concurrent_bench.py 基本一致，但对 TTFT 做了一处
    跨引擎必要的修正：
      - 逐行读 SSE，跳过空行 / [DONE] / 非 JSON 行；
      - **TTFT = 第一个产出非空 token 的 chunk 到达时间**（content 或
        reasoning_content 或 tool_calls）。原因：NInfer 会先发一个仅含
        `role:"assistant"`、content 为空的 delta，若按"第一个含 choices 的
        chunk"计时，TTFT 会假性接近 0，跨引擎不可比。SGLang 同样适用此规则。
      - decode_tok_s = completion_tokens / (total_s - ttft_s)；
      - 用 stream_options.include_usage 拿 usage.completion_tokens / prompt_tokens；
      - 额外捕获 NInfer 原生 timings（draft_n / draft_n_accepted / predicted_ms）。
    任何异常都收敛到 error 字段，绝不抛出。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    first_token_kind = None
    first_choice_t = None
    total_out = None
    prompt_tokens = None
    finish_reason = None
    engine_timings = None
    err = None
    try:
        with requests.post(chat_url(base), json=payload, stream=True, timeout=900) as r:
            # 400/context 超限等错误在这里被捕获并记录，不中断整体流程
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                line = line.strip()
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    if first_choice_t is None:
                        first_choice_t = time.perf_counter() - t0
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    tool_calls = delta.get("tool_calls")
                    # 只有真正产出 token 的 chunk 才算首 token（见 docstring）
                    if ttft is None and (content or reasoning or tool_calls):
                        ttft = time.perf_counter() - t0
                        first_token_kind = (
                            "reasoning" if reasoning else ("content" if content else "tool_call")
                        )
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                if obj.get("usage"):
                    total_out = obj["usage"].get("completion_tokens")
                    prompt_tokens = obj["usage"].get("prompt_tokens")
                if obj.get("timings"):
                    engine_timings = obj["timings"]  # NInfer 原生 timings
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    t_end = time.perf_counter()

    # 兜底：若全程没有非空 delta（极少见），退回首个 choices chunk 的时间
    if ttft is None and first_choice_t is not None:
        ttft = first_choice_t
        first_token_kind = "empty"

    gen_time = (t_end - t0) - ttft if ttft is not None else 0.0
    return {
        "ttft_s": ttft,
        "first_token_kind": first_token_kind,
        "total_s": t_end - t0,
        "gen_s": gen_time,
        "completion_tokens": total_out,
        "prompt_tokens": prompt_tokens,
        "decode_tok_s": (total_out / gen_time) if (total_out and gen_time > 0) else None,
        "finish_reason": finish_reason,
        "engine_timings": engine_timings,
        "error": err,
    }


# ---------------------------------------------------------------- 统计辅助

def _median(rows: list, key: str):
    """对非空值取 median；无有效值时返回 None。"""
    vals = [r.get(key) for r in rows if r and r.get(key) is not None]
    return statistics.median(vals) if vals else None


def _join_finish(rows: list):
    vals = sorted({r.get("finish_reason") for r in rows if r and r.get("finish_reason")})
    return ",".join(vals) if vals else None


def _join_errors(rows: list):
    errs = [r.get("error") for r in rows if r and r.get("error")]
    return " | ".join(errs) if errs else None


def _salt(label: str, scenario: str, rnd: int) -> str:
    """每轮唯一 salt：即使引擎支持前缀缓存，也无法命中上一轮的前缀。"""
    return f"\n# run-salt: {label}-{scenario}-{rnd}-{uuid.uuid4()}\n"


def build_record(label, scenario, endpoint, model, prompt_tokens_est, rounds, wall_s=None) -> dict:
    return {
        "label": label,
        "scenario": scenario,
        "endpoint": endpoint,
        "model": model,
        "prompt_tokens_est": prompt_tokens_est,
        "rounds": rounds,
        "ttft_s_med": _median(rounds, "ttft_s"),
        "decode_tok_s_med": _median(rounds, "decode_tok_s"),
        "total_s_med": _median(rounds, "total_s"),
        "wall_s": wall_s,
        "error": None,
    }


# ---------------------------------------------------------------- 场景执行

def run_single(base, model, label, scenario, prompt, max_tokens, rounds, temperature) -> list:
    """串行场景：跑 rounds 轮，每轮 flush + salt 后测一次。"""
    rows = []
    for i in range(rounds):
        flush_cache(base)
        time.sleep(0.5)  # 与参考脚本一致，给清缓存留出时间
        salted = prompt + _salt(label, scenario, i)
        m = stream_once(base, model, salted, max_tokens, temperature)
        m["round"] = i
        rows.append(m)
    return rows


def _worker(base, model, prompt, max_tokens, temperature, start_event, results, idx):
    """并发线程体：等 start gate 后发起请求，结果写回 results[idx]。"""
    try:
        start_event.wait()
        results[idx] = stream_once(base, model, prompt, max_tokens, temperature)
    except Exception as e:  # noqa: BLE001  # 双重保险
        results[idx] = {
            "ttft_s": None, "first_token_kind": None, "total_s": 0.0, "gen_s": 0.0,
            "completion_tokens": None, "prompt_tokens": None, "decode_tok_s": None,
            "finish_reason": None, "engine_timings": None, "error": f"{type(e).__name__}: {e}",
        }


def run_concurrent(base, model, label, scenario, prompts, max_tokens, rounds, temperature):
    """并发场景：每轮用 start gate 让 N 个请求同时发出，测墙钟 + 每请求指标。

    返回 (rows, wall_s_median)。每轮 rows[i] 为本轮 N 个请求的 median 汇总，
    并额外附带 'requests' 字段保留每请求明细。
    """
    n = len(prompts)
    rows = []
    walls = []
    for i in range(rounds):
        flush_cache(base)
        time.sleep(0.5)
        salted = [p + _salt(label, scenario, i) for p in prompts]
        start_event = threading.Event()
        results = [None] * n
        threads = [
            threading.Thread(
                target=_worker,
                args=(base, model, salted[k], max_tokens, temperature, start_event, results, k),
            )
            for k in range(n)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        start_event.set()  # 放行：所有线程几乎同时开始计时
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        walls.append(wall)

        rows.append({
            "round": i,
            "ttft_s": _median(results, "ttft_s"),
            "total_s": _median(results, "total_s"),
            "gen_s": _median(results, "gen_s"),
            "completion_tokens": _median(results, "completion_tokens"),
            "decode_tok_s": _median(results, "decode_tok_s"),
            "finish_reason": _join_finish(results),
            "error": _join_errors(results),
            "requests": results,  # 额外：并发下每条请求的明细，便于排查
        })
    wall_med = statistics.median(walls) if walls else None
    return rows, wall_med


# ---------------------------------------------------------------- token 估算

_token_cache = {}


def estimate_prompt_tokens(base, model, text, chars_per_token):
    """短文本用真实探针；长文本用校准比例外推。结果缓存避免重复探针。"""
    if text in _token_cache:
        return _token_cache[text]
    est = None
    if len(text) < 5000:
        try:
            est = probe_prompt_tokens(base, model, text)
        except Exception as e:  # noqa: BLE001
            print(f"[info] short probe failed: {type(e).__name__}: {e}", file=sys.stderr)
    if est is None:
        est = int(len(text) / chars_per_token) if chars_per_token else None
    _token_cache[text] = est
    return est


# ---------------------------------------------------------------- CLI / 主流程

def select_scenarios(args) -> list:
    """决定本次要跑哪些场景（保持 ALL_SCENARIOS 的顺序）。"""
    if args.scenarios:
        tokens = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if "all" in tokens:
            wanted = list(BASE_SCENARIOS)
            if args.longgen:
                wanted.append("longctx_32k_longgen")
        else:
            wanted = tokens
    else:
        wanted = list(BASE_SCENARIOS)
        if args.longgen:
            wanted.append("longctx_32k_longgen")

    unknown = [s for s in wanted if s not in ALL_SCENARIOS]
    if unknown:
        print(f"[info] unknown scenarios ignored: {unknown}", file=sys.stderr)
    wanted_set = set(wanted)
    return [s for s in ALL_SCENARIOS if s in wanted_set]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="跨引擎统一 benchmark（OpenAI 兼容接口，适用于 SGLang / NInfer）",
    )
    p.add_argument("--label", required=True, help="本次运行的标签，写入每条记录")
    p.add_argument("--out", required=True, help="JSONL 输出路径（追加写入）")
    p.add_argument("--base-url", default=DEFAULT_BASE, help=f"服务地址（默认 {DEFAULT_BASE}）")
    p.add_argument("--model", default=None, help="模型名；省略则通过 /v1/models 自动发现")
    p.add_argument("--rounds", type=int, default=3, help="每个场景的轮数（默认 3）")
    p.add_argument("--quick", action="store_true", help="快速模式：rounds 强制为 2")
    p.add_argument("--scenarios", default=None, help="逗号分隔的场景子集（默认全部）")
    p.add_argument("--longgen", action="store_true", help="额外包含 longctx_32k_longgen 场景")
    p.add_argument("--temp", type=float, default=0.2, help="采样温度（默认 0.2）")
    return p.parse_args(argv)


def main():
    args = parse_args()

    rounds = 2 if args.quick else max(1, args.rounds)
    base = args.base_url.rstrip("/")
    endpoint = chat_url(base)
    model = discover_model(base, args.model)
    print(f"[info] model={model}", file=sys.stderr)

    selected = select_scenarios(args)

    # 需要长上下文时才构造大 body + 校准
    need_long = any(s.startswith("longctx") or s == "conc2_long_distinct" for s in selected)
    chars_per_token = 3.5  # 兜底值（code 域经验值）
    body = ""
    if need_long:
        body = build_long_body()
        try:
            chars_per_token = calibrate_chars_per_token(base, model, body[:80000])
            # [info] 前缀的校准行允许打印到 stdout
            print(f"[info] calibration chars_per_token={chars_per_token:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"[info] calibration failed ({type(e).__name__}: {e}); "
                  f"fallback chars_per_token={chars_per_token}", file=sys.stderr)

    def p32(offset=0):
        return slice_long_prompt(body, 32000, chars_per_token, offset)

    def p128():
        return slice_long_prompt(body, 128000, chars_per_token, 0)

    # 输出路径：确保父目录存在
    out_path = os.path.abspath(args.out)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    def emit(record):
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)  # 每个场景一行 JSON 到 stdout
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    for sc in selected:
        try:
            if sc == "short_coding_gen":
                rows = run_single(base, model, args.label, sc, CODE_TASK, 512, rounds, args.temp)
                pte = estimate_prompt_tokens(base, model, CODE_TASK, chars_per_token)
                rec = build_record(args.label, sc, endpoint, model, pte, rows)

            elif sc == "longctx_32k_prefill":
                text = p32(0)
                rows = run_single(base, model, args.label, sc, text, 256, rounds, args.temp)
                pte = int(len(text) / chars_per_token)
                rec = build_record(args.label, sc, endpoint, model, pte, rows)

            elif sc == "longctx_32k_longgen":
                text = p32(0)
                rows = run_single(base, model, args.label, sc, text, 1024, rounds, args.temp)
                pte = int(len(text) / chars_per_token)
                rec = build_record(args.label, sc, endpoint, model, pte, rows)

            elif sc == "longctx_128k_prefill":
                text = p128()
                rows = run_single(base, model, args.label, sc, text, 128, rounds, args.temp)
                pte = int(len(text) / chars_per_token)
                rec = build_record(args.label, sc, endpoint, model, pte, rows)

            elif sc == "conc2_short":
                prompts = [CODE_TASK, CODE_TASK]
                rows, wall = run_concurrent(base, model, args.label, sc, prompts, 512, rounds, args.temp)
                pte = estimate_prompt_tokens(base, model, CODE_TASK, chars_per_token)
                rec = build_record(args.label, sc, endpoint, model, pte, rows, wall_s=wall)

            elif sc == "conc2_long_distinct":
                # 两个不同起始位置 → 前缀不同，模拟两个独立长会话
                prompts = [p32(0), p32(300000)]
                rows, wall = run_concurrent(base, model, args.label, sc, prompts, 256, rounds, args.temp)
                pte = int(len(prompts[0]) / chars_per_token)
                rec = build_record(args.label, sc, endpoint, model, pte, rows, wall_s=wall)

            else:  # 理论上不会到达（select_scenarios 已过滤）
                continue
        except Exception as e:  # noqa: BLE001  # 单个场景级异常也不能中断整体
            rec = build_record(args.label, sc, endpoint, model, None, [], wall_s=None)
            rec["error"] = f"{type(e).__name__}: {e}"

        emit(rec)


if __name__ == "__main__":
    main()
