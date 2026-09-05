"""
Phase 8: Rate limiting, per API key.

Approach: a sliding window counter. For each API key, we keep a list
of timestamps of its recent requests. When a new request comes in,
we drop timestamps older than the window, then check if the count
of what remains exceeds the key's limit.
"""

import time
import threading
from collections import defaultdict
from fastapi import HTTPException, status

RATE_LIMITS = {
    "free": {"max_requests": 5, "window_seconds": 60},
    "unlimited": {"max_requests": 1000, "window_seconds": 60},
}

_request_log = defaultdict(list)
_lock = threading.Lock()


def check_rate_limit(auth: dict):
    """
    Call this AFTER verify_api_key() succeeds. Raises 429 if the key
    has exceeded its allowed requests in the current window.
    """
    api_key = auth["api_key"]
    tier = auth.get("tier", "free")
    limit_config = RATE_LIMITS.get(tier, RATE_LIMITS["free"])

    max_requests = limit_config["max_requests"]
    window_seconds = limit_config["window_seconds"]

    now = time.time()

    with _lock:
        _request_log[api_key] = [
            t for t in _request_log[api_key] if now - t < window_seconds
        ]

        current_count = len(_request_log[api_key])

        if current_count >= max_requests:
            oldest = min(_request_log[api_key])
            retry_after = round(window_seconds - (now - oldest), 1)

            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds}s for tier '{tier}'.",
                headers={"Retry-After": str(retry_after)}
            )

        _request_log[api_key].append(now)
