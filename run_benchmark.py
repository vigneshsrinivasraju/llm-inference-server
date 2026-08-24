"""
Phase 4: Benchmarking script.

What this does:
  1. Fires multiple requests at the server AT THE SAME TIME (concurrently)
  2. Measures latency and tokens/sec for each one
  3. Repeats this at different concurrency levels (1, 5, 10 simultaneous users)
  4. Saves everything to a JSON file so we can chart it later

Why concurrency matters:
  Testing "1 request, then wait, then next request" tells you almost nothing
  about how a server behaves in the real world, where many users hit it at
  the same time. This script simulates that.
"""

import requests
import time
import json
import concurrent.futures
from statistics import mean

SERVER_URL = "http://127.0.0.1:8000/generate"

# The prompts we'll cycle through - using a few different ones
# so results aren't skewed by one lucky/unlucky prompt
TEST_PROMPTS = [
    "Explain what machine learning is in two sentences.",
    "What is the capital of France and why is it famous?",
    "Describe how a car engine works briefly.",
    "What are the benefits of regular exercise?",
    "Explain photosynthesis in simple terms.",
]

# We'll test the server under these different levels of simultaneous load
CONCURRENCY_LEVELS = [1, 5, 10]

# How many total requests to send at EACH concurrency level
REQUESTS_PER_LEVEL = 10


def send_single_request(prompt: str) -> dict:
    """
    Sends one request to the server and measures how long it takes
    from THIS SIDE (the client) - this is important because it captures
    real-world latency, including network overhead, not just the
    server's internal generation time.
    """
    start = time.time()
    try:
        response = requests.post(
            SERVER_URL,
            json={"prompt": prompt, "max_tokens": 100, "temperature": 0.7},
            timeout=120  # generous, since requests now queue safely under load
        )
        elapsed = time.time() - start
        data = response.json()

        return {
            "success": True,
            "client_side_latency_sec": round(elapsed, 3),
            "tokens_generated": data.get("tokens_generated", 0),
            "server_reported_tokens_per_sec": data.get("tokens_per_second", 0)
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "client_side_latency_sec": round(elapsed, 3),
            "error": str(e)
        }


def run_benchmark_at_concurrency(concurrency: int, total_requests: int) -> dict:
    """
    Fires `total_requests` requests using `concurrency` workers running
    at the same time. This is the core of the benchmark.

    ThreadPoolExecutor is what actually makes requests run CONCURRENTLY
    instead of one after another - it manages a pool of worker threads,
    each independently sending a request and waiting for its response.
    """
    print(f"\n--- Running benchmark: concurrency={concurrency}, total_requests={total_requests} ---")

    prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(total_requests)]

    overall_start = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # This submits all requests to the thread pool. With max_workers=concurrency,
        # at most `concurrency` requests are ever in-flight to the server at once.
        futures = [executor.submit(send_single_request, p) for p in prompts]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result["success"] else "FAILED"
            print(f"  Request done [{status}] - latency: {result['client_side_latency_sec']}s")

    overall_elapsed = time.time() - overall_start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        avg_latency = mean(r["client_side_latency_sec"] for r in successful)
        avg_tokens_per_sec = mean(r["server_reported_tokens_per_sec"] for r in successful)
        total_tokens = sum(r["tokens_generated"] for r in successful)
    else:
        avg_latency = 0
        avg_tokens_per_sec = 0
        total_tokens = 0

    # Overall throughput of the SYSTEM (not just one request) -
    # this is what matters when many users hit it at once
    system_throughput = total_tokens / overall_elapsed if overall_elapsed > 0 else 0

    summary = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_wall_clock_time_sec": round(overall_elapsed, 3),
        "avg_client_latency_sec": round(avg_latency, 3),
        "avg_server_tokens_per_sec": round(avg_tokens_per_sec, 2),
        "system_throughput_tokens_per_sec": round(system_throughput, 2)
    }

    print(f"  Summary: {summary}")
    return summary


def main():
    print("Starting benchmark suite...")
    print(f"Target server: {SERVER_URL}")

    # Quick check that the server is actually up before we start
    try:
        requests.get(SERVER_URL.replace("/generate", "/"), timeout=5)
    except Exception:
        print("ERROR: Could not reach the server. Is it running? (uvicorn server.main:app --reload)")
        return

    all_results = []
    for concurrency in CONCURRENCY_LEVELS:
        summary = run_benchmark_at_concurrency(concurrency, REQUESTS_PER_LEVEL)
        all_results.append(summary)

    # Save everything to a JSON file for later charting
    output_path = "benchmarks/results/fp16_cpu_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll benchmarks complete. Results saved to {output_path}")
    print("\n--- FINAL COMPARISON TABLE ---")
    print(f"{'Concurrency':<12} {'Avg Latency(s)':<16} {'Avg Tok/s':<12} {'System Tok/s':<14}")
    for r in all_results:
        print(f"{r['concurrency']:<12} {r['avg_client_latency_sec']:<16} {r['avg_server_tokens_per_sec']:<12} {r['system_throughput_tokens_per_sec']:<14}")


if __name__ == "__main__":
    main()
