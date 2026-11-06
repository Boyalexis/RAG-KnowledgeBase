"""单进程 LLM 请求速率控制：所有 Planner/Generator 调用共享同一节流器。"""
from __future__ import annotations

import logging
import threading
import time

from openai import RateLimitError

logger = logging.getLogger(__name__)


class RequestGate:
    def __init__(self, min_interval_seconds: float = 25.0):
        if min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be positive")
        self._interval = min_interval_seconds
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait_turn(self) -> None:
        with self._lock:
            delay = self._next_allowed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_allowed = time.monotonic() + self._interval


# 注意：只在当前 Python 进程内共享；多进程/多副本需使用外部限流器。
_gate = RequestGate(min_interval_seconds=25.0)


def limited_chat_completion(client, *, max_attempts: int = 2, **kwargs):
    """两次请求至少间隔 25 秒；遇到 429 最多重试一次。"""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    for attempt in range(max_attempts):
        _gate.wait_turn()
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt + 1 >= max_attempts:
                raise
            logger.warning("LLM API 返回 429，按全局速率限制重试（第 %d 次）", attempt + 2)

    raise AssertionError("unreachable")
