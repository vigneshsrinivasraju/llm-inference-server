"""
Phase 9: Request logging + basic metrics.

Every request gets recorded in memory as a simple dict. The /metrics
endpoint aggregates these into a summary - total requests, per-key
breakdown, average latency, and error/rate-limit counts.

In a real production system this would go to a proper time-series
store (Prometheus, etc.) - but the CONCEPT (record what happened,
then aggregate for visibility) is identical.
"""

import time
import threading
from collections import defaultdict

_lock = threading.Lock()
_request_log = []  # list of dicts, one per request

MAX_LOG_SIZE = 1000  # cap memory usage - keep only the most recent N


def log_request(api_key: str, endpoint: str, status_code: int,
                 latency_sec: float, tokens_generated: int = 0):
    """Call this after every request completes, success or failure."""
    entry = {
        "timestamp": time.time(),
        "api_key": api_key,
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_sec": round(latency_sec, 3),
        "tokens_generated": tokens_generated,
    }
    with _lock:
        _request_log.append(entry)
        if len(_request_log) > MAX_LOG_SIZE:
            _request_log.pop(0)  # drop oldest to cap memory


def get_metrics_summary() -> dict:
    """Aggregates the in-memory log into a summary dict for /metrics."""
    with _lock:
        log_copy = list(_request_log)

    if not log_copy:
        return {
            "total_requests": 0,
            "message": "No requests recorded yet."
        }

    total_requests = len(log_copy)
    successful = [r for r in log_copy if r["status_code"] == 200]
    rate_limited = [r for r in log_copy if r["status_code"] == 429]
    unauthorized = [r for r in log_copy if r["status_code"] == 401]
    other_errors = [
        r for r in log_copy
        if r["status_code"] not in (200, 429, 401)
    ]

    avg_latency = (
        round(sum(r["latency_sec"] for r in successful) / len(successful), 3)
        if successful else 0
    )
    total_tokens = sum(r["tokens_generated"] for r in successful)

    per_key_counts = defaultdict(int)
    for r in log_copy:
        per_key_counts[r["api_key"]] += 1

    return {
        "total_requests": total_requests,
        "successful_requests": len(successful),
        "rate_limited_requests": len(rate_limited),
        "unauthorized_requests": len(unauthorized),
        "other_errors": len(other_errors),
        "avg_latency_sec": avg_latency,
        "total_tokens_generated": total_tokens,
        "requests_per_api_key": dict(per_key_counts),
    }
