# quality_probe 并排对照 — official vs abliterated

> 同一 prompt、同一采样参数（temp 0.2）、同一服务配置；仅权重血统不同。


---

## 任务 `lru`

| 指标 | official | abliterated |
|---|---|---|
| finish_reason | stop | length |
| total_s(s) | 4.266 | 5.154 |
| ttft_s(s) | 0.085 | 0.096 |
| content_chars | 2209 | 2788 |
| reasoning_chars | 1176 | 1165 |
| tool_calls | 0 | 0 |

### official — content

```
```python
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional


class LRUCache:
    """çº¿ç¨ LRU ç¼å­ã

    ä½¿ç¨ ``collections.OrderedDict`` ç»´æ¤è®¿é®é¡ºåºï¼ä¿è¯ ``get``ã``put``ã
    ``peek`` åä¸º O(1) æ¶é´å¤æåº¦ã

    Attributes:
        capacity: ç¼å­æå¤§å®¹éã
    """

    def __init__(self, capacity: int) -> None:
        """åå§å LRU ç¼å­ã

        Args:
            capacity: ç¼å­æå¤§å®¹éï¼

        Raises:
            ValueError: å½ ``capacity`` å°äºç­äº 0 æ¶æåºã
        """
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self.capacity: int = capacity
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock: threading.RLock = threading.RLock()

    def get(self, key: Hashable) -> Optional[Any]:
        """è·åç¼å­å¼ï¼å¹¶å°è¯¥é®æ è®°ä¸ºæè¿ä½¿ç¨ã

        Args:
            key: ç¼å­é®ã

        Returns:
            è¥é®å­å¨ï¼è¿åå¯¹åºå¼ï¼å¦åè¿å ``None``ã
        """
        with self._lock:
            if key not in self._data:
                return None

            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: Hashable, value: Any) -> None:
       ç¼å­å¼ï¼å¹¶å°è¯¥é®æ è®°ä¸ºæè¿ä½¿ç¨ã

        è¥ç¼å­å·²æ»¡ä¸çæ¯æ°é®ï¼æªä½¿ç¨çé®ã

        Args:
            key: ç¼å­é®ã
            value: ç¼å­å¼ã
        """
        with self._lock:
            if key in self._data:
                self._data[key] = value
                self._data.move_to_end(key)
                return

            if len(self._data) >= self.capacity:
                self._data.popitem(last=False)

            self._data[key] = value

    def peek(self, key: Hashable) -> Optional[Any]:
        """è·åç¼å­å¼ï¼ä½ä¸æ¹åé®çæè¿ä½¿ç¨é¡ºåºã

        Args:
            key: ç¼å­é®ã

        Returns:
            è¥é®å­å¨ï¼è¿åå¯¹åºå¼ï¼å¦åè¿å ``None``ã
        """
        with self._lock:
            return self._data.get(key, None)
```
```


### official — reasoning

```
We need answer user's request in Chinese? User asks: "ä½ æ¯ä¸ä¸ªèµæ·± Python å·¥ç¨å¸ãè¯·å®ç°ä¸ä¸ª LRU ç¼å­ç±»ï¼è¦æ±ï¼
1. æ¯æ get(key) / put(key, value) / peek(key) ä¸ä¸ªæ¹æ³
2. . æ¶é´å¤æåº¦ O(1)
4. ç¨ç±»åæ³¨è§£ï¼åå®æ´ç docstring
è¯·ç´æ¥è¾åºä»£ç ï¼ä¸è¦è§£éã"

Need output code only, no explanation. Need implement LRU cache thread-safe O(1), type annotations, complete docstring. Need likely use OrderedDict? But O(1) and thread safe. Could implement with collections.OrderedDict and RLock. get/put/peek. Need decide semantics: get returns value and moves to most recently used; put inserts/updates and moves; peek returns value without changing order. Need capacity? LRU cache usually capacity. User didn't specify capacity but LRU implies max size. Need include capacity parameter. Could be generic? Type annotations. Need thread safety. Use threading.RLock. Use OrderedDict for O(1) move_to_end. But if need pure? Python stdlib okay. Need complete docstring. Direct code only.

Potential code:
from __future__ import

 Considering the limited time by the user, I have to give the solution based on the thinking directly now.

```


### abliterated — content

```
```python
from __future__ import annotations

import threading
from collections.abc import Hashable
from typing import Generic, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _Node(Generic[K, V]):
    """Doubly linked list node for LRU cache."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key: K = key
        self.value: V = value
        self.prev: Optional[_Node[K, V]] = None
        self.next: Optional[_Node[K, V]] = None


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache with O(1) get/put/peek operations.

    The cache maintains a fixed capacity. When capacity is exceeded,
    the least recently used item is evicted.

    Thread safety is provided by a reentrant lock.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize the LRU cache.

        Args:
            capacity: Maximum number of items the cache can hold.
                Must be a non-negative integer.

        Raises:
            ValueError: If capacity is negative.
        """
        if capacity < 0:
            raise ValueError("capacity must be non-negative")

        self._capacity: int = capacity
        self._data: dict[K, _Node[K, V]] = {}
        self._lock: threading.RLock = threading.RLock()

        # Dummy head and tail nodes to simplify edge cases.
        self._head: _Node[K, V] = _Node(None, None)  # type: ignore[arg-type]
        self._tail: _Node[K, V] = _Node(None, None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node[K, V]) -> None:
        """Remove a node from the linked list.

        Args:
            node: Node to remove.
        """
        prev_node = node.prev
        next_node = node.next
        if prev_node is not None:
            prev_node.next = next_node
        if next_node is not None:
            next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _add_to_front(self, node: _Node[K, V]) -> None:
        """Add a node right after the head, marking it as most recently used.

        Args:
            node: Node to add.
        """
        first = self._head.next
        self._head.next = node
        node.prev = self._head
        node.next = first
        if first is not None:
            first.prev = node

    def _move_to_front(self, node: _Node[K, V]) -> None:
        """Move a node to the front, marking it as most recently used.

        Args:
            node: Node to move.
        """
        self._remove(node)
        self._add_to_front(node)

    def _pop_back(self) -> Optional[_Node[K, V]]:
        """Remove and return the least recently used node.

        Returns:
           
```


### abliterated — reasoning

```
We need answer user in Chinese likely. User asks: "ä½ æ¯ä¸ä¸ªèµæ·± Python å·¥ç¨å¸ãè¯·å®ç°ä¸ä¸ª LRU ç¼å­ç±»ï¼è¦æ±ï¼
1. æ¯æ get(key) / put(key, value) / peek(key) ä¸ä¸ªæ¹æ³
2.3. æ¶é´å¤æåº¦ O(1)
4. ç¨ç±»åæ³¨è§£ï¼åå®æ´ç docstring
è¯·ç´æ¥è¾åºä»£ç ï¼ä¸è¦è§£éã"

Need output code only, no explanation. Need implement LRU cache thread-safe O(1), type annotations, full docstrings. Need decide language Python. Need likely use OrderedDict? But O(1) and thread-safe. Could implement with dict + doubly linked list for O(1) and no reliance on OrderedDict? OrderedDict move_to_end O(1) average? It's fine. But to be robust, implement custom doubly linked list. Need thread safety: use threading.RLock or Lock. Methods get, put, peek. Need type annotations. Need complete docstrings. Direct output code.

Consider semantics: LRU cache capacity. get returns value if exists and marks most recently used; if not returns None? Could return Optional[Value]. put inserts/updates and marks MRU; if capacity exceeded evict LRU.

 Considering the limited time by the user, I have to give the solution based on the thinking directly now.

```


---

## 任务 `bugfix`

| 指标 | official | abliterated |
|---|---|---|
| finish_reason | stop | stop |
| total_s(s) | 2.713 | 2.461 |
| ttft_s(s) | 0.102 | 0.095 |
| content_chars | 651 | 528 |
| reasoning_chars | 1272 | 1184 |
| tool_calls | 0 | 0 |

### official — content

```
æ ¹å ï¼`incr()` ä¸­ç `cur = self.n`ã`cur += 1`ã`self.n = cur` æ¯ä¸ä¸ªâè¯»åâä¿®æ¹âååâçå¤åæä½ï¼ä¸æ¯åå­æä½ãå¤çº¿ç¨å¹¶åæ§è¡æ¶ï¼ä¸¤ä¸ªçº¿ç¨å¯è½åæ¶è¯»å°åä¸ä¸ª `self.n`ï¼ç¶ååèªå  1 åååï¼å¯¼è´ä¸æ¬¡èªå¢ä¸¢å¤±ã

æå°æ¹å¨ï¼ç¨ `threading.Lock` ä¿æ¤è¿æ®µè¯»æ¹åé»è¾ã

```python
import threading

class Counter:
    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()

    def incr(self):
        with self._lock:
            self.n += 1
```

è¯´æï¼`with self._lock:` ä¿è¯åä¸æ¶å»åªæä¸ä¸ªçº¿ç¨æ§è¡ `self.n += 1`ï¼ææ¡ä»¶ã
```


### official — reasoning

```
We need answer in Chinese. User asks: Python code has concurrency defect, point root cause and minimal fix, provide full fixed code. Requirement: support multithread-safe increment, explain why original unsafe.

Need think. Original Counter.incr does read-modify-write without synchronization. In CPython, due to GIL, individual bytecode operations atomic? But sequence cur = self.n; cur += 1; self.n = cur can be interleaved between threads, causing lost updates. Even if GIL prevents true parallel execution, thread switches can occur between bytecodes, so race condition. Need minimal change: add threading.Lock around read-modify-write. Could use threading.Lock in __init__, acquire in incr. Or use itertools.count? But minimal: lock.

Need provide complete code:
import threading

class Counter:
    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()
    def incr(self):
        with self._lock:
            self.n += 1
            return self.n? Original didn't return. Minimal maybe no return. But if want return? Keep same behavior? Original returns None. Could keep no return. But maybe useful. Minimal change: add lock and with

 Considering the limited time by the user, I have to give the solution based on the thinking directly now.

```


### abliterated — content

```
æ ¹å ï¼`incr()` ä¸­ç `cur = self.n`ã`cur += 1`ã`self.n = cur` æ¯ä¸ä¸ªâè¯»åâä¿®æ¹âååâçå¤åæä½ï¼ä¸æ¯åå­çãå¤çº¿ç¨ä¸å¯è½ä¸¤ä¸ªçº¿ç¨åæ¶è¯»å°ç¸åç `self.n`ï¼ç¶åé½åå `n + 1`ï¼å¯¼è´ä¸æ¬¡èªå¢ä¸¢å¤±ã

æå°ä¿®å¤ï¼ç¨ `threading.Lock` ä¿æ¤è¿æ®µè¯»æ¹åé»è¾ã

```python
import threading

class Counter:
    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()

    def incr(self):
        with self._lock:
            self.n += 1
```
```


### abliterated — reasoning

```
We need answer in Chinese. User asks: Python code has concurrency defect, point out root cause and minimal fix, provide complete fixed code. Requirement: support multithread-safe increment, explain why original unsafe.

Need think: Python GIL? Even with GIL, incr has race condition due to non-atomic read-modify-write: cur = self.n; cur += 1; self.n = cur. Multiple threads can interleave between bytecode operations, both read same n, both write n+1, lost update. Minimal fix: use threading.Lock around read-modify-write. Or use itertools.count? But class Counter with n attribute? Minimal: add lock in __init__, with self._lock in incr. Could also use atomic? Python int not atomic. Could use threading.Lock.

Need provide complete code:
import threading

class Counter:
    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()
    def incr(self):
        with self._lock:
            self.n += 1

Maybe if want return value? Original doesn't return. Could keep. Minimal change: add lock.

Explain root cause: check-then-act / read-modify-write not

 Considering the limited time by the user, I have to give the solution based on the thinking directly now.

```


---

## 任务 `toolcall`

| 指标 | official | abliterated |
|---|---|---|
| finish_reason | tool_calls | tool_calls |
| total_s(s) | 0.589 | 0.527 |
| ttft_s(s) | 0.111 | 0.112 |
| content_chars | 0 | 0 |
| reasoning_chars | 171 | 156 |
| tool_calls | 2 | 2 |

### official — content

_(empty)_


### official — tool_calls

```json
[
  {
    "id": "call_6b25b6a1ba15fe8c",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\":\"åäº¬\"}"
    }
  },
  {
    "id": "call_3df89d05960aab3a",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\":\"ä¸æµ·\"}"
    }
  }
]
```


### official — reasoning

```
ç¨æ·è¦æ±æ¥è¯¢åäº¬åä¸æµ·ç°å¨çå¤©æ°ï¼éè¦è°ç¨ get_weather åå¸çå¤©æ°æ¥è¯¢æ¯ç¸äºç¬ç«çï¼æ²¡æå¯ä»¥å¨åä¸ä¸ªåä¸­å¹¶è¡è°ç¨ä¸¤æ¬¡ get

```


### abliterated — content

_(empty)_


### abliterated — tool_calls

```json
[
  {
    "id": "call_c99fd1f4ed16deab",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\":\"åäº¬\"}"
    }
  },
  {
    "id": "call_0cb1593d5b5653ed",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\":\"ä¸æµ·\"}"
    }
  }
]
```


### abliterated — reasoning

```
The user is asking for the current weather in Beijing and Shanghai. Since these are independent calls, I can invoke both get_weather functions in parallel.

```


---

## 任务 `design`

| 指标 | official | abliterated |
|---|---|---|
| finish_reason | length | length |
| total_s(s) | 7.534 | 8.072 |
| ttft_s(s) | 0.097 | 0.099 |
| content_chars | 4784 | 4573 |
| reasoning_chars | 1142 | 1226 |
| tool_calls | 0 | 0 |

### official — content

```
ä¸é¢ç»åºä¸ä¸ªå®æ´å¯è¿è¡ç Python å®ç°ï¼

- éæµå¨
- æ¯æè®¾ç½®å®¹é `capacity` ä¸ill_rate`
- `try_acquire(n)` éé»å¡ï¼æåè¿å `True`ï¼å¤±è´¥è¿å `False`
- æ¯ææ `key` ç»´åº¦éç¦»ï¼å®ç°å¤ç§æ·éæµ
- éå¸¦ `
- éå¸¦ç®å `__main__` æ¼ç¤º

---

## å®æ´ä»£ç 

```python
"""æ¡¶éæµå¨

åè½ï¼
1. æ¯æè®¾ç½®å®¹é capacity refill_rate
2. try_acquire(n) éé»å¡ï¼æåè¿å Trueï¼å¤±è´¥è¿å False
3. æ¯ææ key ç»´åº¦éç¦»ï¼å®ç°å¤ç§æ·éæµ
4. """

import threading
import time
import unittest
from typing import Dict, Optional


class TokenBucket:
    """
    åä¸ªä»¤çæ¡¶

    åæ°ï¼
        capacity: æ¡¶å®¹éï¼æå¤§ä»¤çæ°
        refill_rate: æ¯ç§ä»¤çæ°
    """

    def __init__(self, capacity: float, refill_rate: float):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if refill_rate < 0:
            raise ValueError("refill_rate must be non-negative")

        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)

        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """
        æ ¹æ®æ¶é´æµé        æ³¨æï¼è°ç¨æ¹ææ self._lock
        """
        now = time.monotonic()
        elapsed = now - self._last_refill

        if elapsed <= 0:
            return

       ï¼        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.refill_rate
        )
        self._last_refill = now

    def try_acquire(self, n: float = 1) -> bool:
        """
        éé»å¡å°è¯è·å n ä¸ªä»¤ç

        åæ°ï¼
            n: éè¦è·åçä»¤çæ°ï¼é»è®¤ 1

        è¿åï¼
            True: è·åæå
            False: ä»¤çä¸è¶³ï¼è·åå¤±è´¥
        """
        if n < 0:
            raise ValueError("n must be non-negative")

        with self._lock:
            self._refill()

            if self._tokens >= n:
                self._tokens -= n
                return True

            return False

    def available_tokens(self) -> float:
        """
        è¿åå½åå¯ç¨ä»¤çæ°ï¼ä¼ä»¤çï¼
        """
        with self._lock:
            self._refill()
            return self._tokens


class TokenBucketRateLimiter:
    """
    å¤ç§æ·ä»¤çæ¡¶éæµå¨

    æ¯ä¸ª key å¯¹åºä¸ä¸ªç¬ç«ç TokenBucketã
    """

    def __init__(
        self,
        default_capacity: float,
        default_refill_rate: float,
        key_config: Optional[Dict[str, tuple]] = None
    ):
        """
        åæ°ï¼
            default_capacity: é»è®¤æ¡¶å®¹é
            default_refill_rate:
            key_config: å¯éï¼ä¸åæ¡¶åæ°
                        ä¾å¦ï¼{"user1": (10, 2.0), "user2": (5, 1.0)}
        """
        if default_capacity < 0:
            raise ValueError("default_capacity must be non-negative")
        if default_refill_rate < 0:
            raise ValueError("default_refill_rate must be non-negative")

        self.default_capacity = float(default_capacity)
        self.default_refill_rate = float(default_refill_rate)

        self.key_config = key_config or {}
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, key: str) -> TokenBucket:
        """
        è·åæåå»ºæå® key å¯¹åºçä»¤çæ¡¶
        """
        with self._lock:
            if key not in self.buckets:
                if key in self.key_config:
                    capacity, refill_rate = self.key_config[key]
                else:
                    capacity = self.default_capacity
                    refill_rate = self.default_refill_rate

                self.buckets[key] = TokenBucket(capacity, refill_rate)

            return self.buckets[key]

    def try_acquire(self, key: str, n: float = 1) -> bool:
        """
        éé»å¡å°è¯ä¸ºæå® key è·å n ä¸ªä»¤ç

        åæ°ï¼
            key: ç§æ·/ç¨æ·/æ¥å£ç­éç¦»ç»´åº¦
            n: éè¦è·åçä»¤çæ°ï¼é»è®¤ 1

        è¿åï¼
            True: è·åæå
            False: ä»¤çä¸è¶³ï¼è·åå¤±è´¥
        """
        bucket = self._get_bucket(key)
        return bucket.try_acquire(n)

    def available_tokens(self, key: str) -> float:
        """
        è¿åæå® key å½åå¯ç¨ä»¤çæ°
        """
        bucket = self._get_bucket(key)
        return bucket.available_tokens()


class TokenBucketRateLimiterWithConfig:
    """
    æ´çµæ´»çå¤ç§æ·éæµå¨ï¼æ¯ä¸ª key åç¬æå® capacity å refill_rate
    """

    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock
```


### official — reasoning

```
We need answer in Chinese likely. User asks: è®¾è®¡å¹¶å®ç°ä¸ä¸ªéæµå¨ï¼token bucket rate limiterï¼ï¼è¦æ±ï¼
1. æ¯æéç
2. try_acquire(n) éé»å¡ï¼æåè¿å True
3. æ¯ææ key ç»´åº¦éç¦»ï¼å¤ç§æ·ï¼
4.è¯·ç»åºå®æ´å¯è¿è¡ä»£ç ã

Need provide complete runnable code. Need think through implementation. Python likely. Need thread-safe. Could use threading.Lock per bucket or global lock. Need support capacity and refill rate. try_acquire(n) non-blocking returns True. Multi-tenant by key. Unit tests. Need maybe include demo main. Need ensure correct token bucket semantics.

Design:
class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float): # tokens per second
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()
    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
           

 Considering the limited time by the user, I have to give the solution based on the thinking directly now.

```


### abliterated — content

```
# çº¿ç¨ä»¤çæ¡¶éæµå¨ï¼Pythonï¼

## è®¾è®¡è¦ç¹

- **æ°æ§ï¼lazy refillï¼**ï¼ä¸å¯å¨åå°çº¿ç¨ï¼æ¯æ¬¡ `try_acquire` æ¶æ ¹æ®æ¶é´å·®è®¡ç®åºä¸ç²¾åº¦é®é¢
- **ä¸¤çº§éåªä¿æ¤"åå»º/æ¥æ¾ï¼ä¸´çåºæç­ï¼ï¼æ¯ä¸ªæ¡¶ææç¬ç«éï¼å¤ç§æ·ä¹é´äºä¸é»å¡
- **`try_acquire(n)` éé»å¡**ï¼ä¸æ¬¡è°ç¨è¦ä¹åå­å°æ£å n ä¸ªä»¤çï¼è¦ä¹åæ ·è¿å Falseï¼æ é¨åæ£å
- **æº¢åºæªåä»¤çæ°ï¼`n > capacity` æ¶å³ä½¿æ¡¶æ»¡ä¹æç»ï¼ä¸ Redis ä»¤çæ¡¶è¯­ä¹ä¸è´ï¼
- **æ¶é**ï¼é»è®¤ `time.monotonic`ï¼ä¸åç³»ç»æ¶é´åæ¨å½±åï¼ï¼å¯æµè¯

## å®æ´ä»£ç 

```python
"""çº¿ç¨æµå¨ï¼æ¯æå¤ç§æ· key éç¦»ï¼ã

ç¨æ³:
    limiter = TokenBucketLimiter(capacity=10, rate=2.0)   # å®¹é10, æ¯ç§    if limiter.try_acquire(3, key="user:42"):
        ...  # æ¾è¡
    else:
        ...  # éæµ
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional


class TokenBucket:
    """åä¸ªä»¤çæ¡¶ãçº¿ç¨ Args:
        capacity: æ¡¶å®¹éï¼æå¤§ä»¤çæ°ï¼ã
        rate:    ä»¤ç/ç§ï¼ã
        clock:    åè°æ¶éï¼é»è®¤ time.monotonicï¼ã
    """

    __slots__ = ("capacity", "rate", "clock", "_tokens", "_last_refill", "_lock")

    def __init__(self, capacity: float, rate: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        self.capacity = float(capacity)
        self.rate = float(rate)
        self.clock = clock
        now = clock()
        self._tokens: float = capacity          # åå§ä¸ºæ»¡æ¡¶
        self._last_refill: float = now
        self._lock = threading.Lock()

    def _refill_locked(self, now: float) -> None:
        """æ°æ§ _lockã"""
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity,
                               self._tokens + elapsed * self.rate)
            self._last_refill = now

    def try_acquire(self, n: float = 1) -> bool:
        """éé»å¡è·å n ä¸ªä»¤çï¼æåè¿å Trueã"""
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if n > self.capacity:
            return False  # å³ä½¿æ¡¶æ»¡ä¹æ¿ä¸åº n ä¸ª
        with self._lock:
            now = self.clock()
            self._refill_locked(now)
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    @property
    def available(self) -> float:
        """å½åå¯ç¨ä»¤çæ°ï¼è¿ä¼¼è§¦åä¸æ¬¡        with self._lock:
            self._refill_locked(self.clock())
            return self._tokens


class TokenBucketLimiter:
    """å¤ç§æ·ä»¤çæ¡¶éæµå¨ï¼æ key éç¦»ï¼æ¯ä¸ª key ä¸ä¸ªç¬ç«æ¡¶ã

    Args:
        capacity: æ¯ä¸ªæ¡¶çå®¹éã
        rate:     æ¯ä¸ªæ¡¶ä»¤ç/ç§ï¼ã
        clock:    åè°æ¶éæµè¯ã
    """

    def __init__(self, capacity: float, rate: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = capacity
        self._rate = rate
        self._clock = clock
        self._buckets: Dict[str, TokenBucket] = {}
        self._buckets_lock = threading.Lock()  # æ¬èº«

    def _get_bucket(self, key: str) -> TokenBucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            with self._buckets_lock:
                bucket = self._buckets.get(key)
                if bucket is None:
                    bucket = TokenBucket(self._capacity, self._rate, self._clock)
                    self._buckets[key] = bucket
        return bucket

    def try_acquire(self, n: float = 1, key: str = "default") -> bool:
        """éé»å¡è·åï¼ä» key å¯¹åºçæ¡¶ä¸­å n ä¸ªä»¤çï¼æåè¿å Trueã"""
        return self._get_bucket(key).try_acquire(n)

    def available(self, key: str = "default") -> float:
        """key å¯¹åºæ¡¶å½åå¯ç¨ä»¤çæ°ï¼è¿ä¼¼å¼ï¼ã"""
        return self._get_bucket(key).available

    def reset(self, key: Optional[str] = None) -> None:
        """éç½®æä¸ª key çæ¡¶ï¼key=None """
        with self._buckets_lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)


# ----------------------------------------------------------------------ï¼python -m unittest -v
#
```


### abliterated — reasoning

```
The user is asking for a thread-safe token bucket rate limiter in Python (implied by the naming convention `try_acquire(n)` and the request for unit tests). Requirements:

1. Support setting capacity and refill rate
2. `try_acquire(n)` is non-blocking, returns True on success
3. Support isolation by key (multi-tenant)
4. Provide unit tests
5. Complete, runnable code

Let me design this properly:

**Design considerations:**

1. **Token bucket algorithm**: A bucket with capacity `capacity`, refilled at `rate` tokens per second. Tokens are added lazily (on access) rather than by a background thread â this is the standard, efficient approach.

2. **Thread safety**: Use `threading.Lock` per bucket, or a global lock. For multi-tenant, per-key locks are better for concurrency, but we also need to protect the dict of buckets themselves. Options:
   - A single global lock (simple, but contention)
   - Per-key locks with a global lock only for bucket creation (better)
   
   I'll go with per-key locks: a global lock protects the dict of buckets, and each bucket has its own lock for token operations. Actually

 Considering the limited time by the user, I have to give the solution based on the thinking directly now.

```
