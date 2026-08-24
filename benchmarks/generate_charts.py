"""
Phase 5: Turn benchmark JSON results into charts.

Reads the JSON file(s) produced by run_benchmark.py and produces
PNG charts showing:
  1. Latency vs Concurrency
  2. Per-request tokens/sec vs Concurrency
  3. System throughput vs Concurrency

These three charts together tell the full story: as more users hit
the server at once, individual experience gets slower (chart 1 & 2),
while total system capacity stays roughly fixed (chart 3) - proving
the CPU/no-batching bottleneck we discussed.
"""

import json
import matplotlib.pyplot as plt
import os

RESULTS_FILE = "benchmarks/results/fp16_cpu_results.json"
OUTPUT_DIR = "benchmarks/results"


def load_results(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def make_charts(results: list, label: str = "FP16 (CPU)"):
    concurrency_levels = [r["concurrency"] for r in results]
    avg_latency = [r["avg_client_latency_sec"] for r in results]
    avg_tokens_per_sec = [r["avg_server_tokens_per_sec"] for r in results]
    system_throughput = [r["system_throughput_tokens_per_sec"] for r in results]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Chart 1: Latency vs Concurrency ---
    plt.figure(figsize=(7, 5))
    plt.plot(concurrency_levels, avg_latency, marker="o", color="#d62728")
    plt.title(f"Average Request Latency vs Concurrency ({label})")
    plt.xlabel("Concurrent Requests")
    plt.ylabel("Average Latency (seconds)")
    plt.grid(True, alpha=0.3)
    plt.xticks(concurrency_levels)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/latency_vs_concurrency.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/latency_vs_concurrency.png")

    # --- Chart 2: Per-request tokens/sec vs Concurrency ---
    plt.figure(figsize=(7, 5))
    plt.plot(concurrency_levels, avg_tokens_per_sec, marker="o", color="#1f77b4")
    plt.title(f"Per-Request Throughput vs Concurrency ({label})")
    plt.xlabel("Concurrent Requests")
    plt.ylabel("Tokens/sec (per request)")
    plt.grid(True, alpha=0.3)
    plt.xticks(concurrency_levels)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/per_request_throughput.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/per_request_throughput.png")

    # --- Chart 3: System throughput vs Concurrency ---
    plt.figure(figsize=(7, 5))
    plt.plot(concurrency_levels, system_throughput, marker="o", color="#2ca02c")
    plt.title(f"Total System Throughput vs Concurrency ({label})")
    plt.xlabel("Concurrent Requests")
    plt.ylabel("System Throughput (tokens/sec, total)")
    plt.grid(True, alpha=0.3)
    plt.xticks(concurrency_levels)
    plt.ylim(0, max(system_throughput) * 1.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/system_throughput.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/system_throughput.png")

    # --- Combined summary chart (all 3 side by side) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(concurrency_levels, avg_latency, marker="o", color="#d62728")
    axes[0].set_title("Avg Latency vs Concurrency")
    axes[0].set_xlabel("Concurrent Requests")
    axes[0].set_ylabel("Latency (sec)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(concurrency_levels)

    axes[1].plot(concurrency_levels, avg_tokens_per_sec, marker="o", color="#1f77b4")
    axes[1].set_title("Per-Request Tok/s vs Concurrency")
    axes[1].set_xlabel("Concurrent Requests")
    axes[1].set_ylabel("Tokens/sec")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(concurrency_levels)

    axes[2].plot(concurrency_levels, system_throughput, marker="o", color="#2ca02c")
    axes[2].set_title("System Throughput vs Concurrency")
    axes[2].set_xlabel("Concurrent Requests")
    axes[2].set_ylabel("Tokens/sec (total)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(concurrency_levels)
    axes[2].set_ylim(0, max(system_throughput) * 1.3)

    fig.suptitle(f"Benchmark Summary - {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/summary_combined.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/summary_combined.png")


def main():
    print(f"Loading results from {RESULTS_FILE}...")
    results = load_results(RESULTS_FILE)
    make_charts(results, label="TinyLlama Q4_K_M (CPU)")
    print("\nAll charts generated. Check benchmarks/results/ for PNG files.")


if __name__ == "__main__":
    main()
