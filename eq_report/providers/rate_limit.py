"""Process-wide request throttle shared by every MegadataAPI caller.

Both the async acquisition providers (megadata.py, whose three branches run
concurrently) and the technical appendix's direct, synchronous fetch
(rendering/technical_appendix.py) hit the same MegadataAPI host. Without a
shared gate, a run can burst several requests at once. This is a single
process-wide, thread-safe minimum-interval limiter: every caller blocks until
at least ``interval_seconds`` has elapsed since the last request returned,
regardless of which thread or provider instance is asking.
"""

from __future__ import annotations

import threading
import time


class MinIntervalLimiter:
    def __init__(self, interval_seconds: float) -> None:
        self._interval = max(0.0, interval_seconds)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
                now += delay
            self._next_allowed = now + self._interval


_limiters: dict[float, MinIntervalLimiter] = {}
_limiters_lock = threading.Lock()


def megadata_limiter(interval_seconds: float) -> MinIntervalLimiter:
    """Return the process-wide limiter for a given interval, creating it once.

    Keyed by interval so every caller configured with the same
    ``EQR_MEGADATA_MIN_REQUEST_INTERVAL_SECONDS`` shares one limiter, however
    many provider instances or threads end up calling it.
    """
    with _limiters_lock:
        limiter = _limiters.get(interval_seconds)
        if limiter is None:
            limiter = MinIntervalLimiter(interval_seconds)
            _limiters[interval_seconds] = limiter
        return limiter
