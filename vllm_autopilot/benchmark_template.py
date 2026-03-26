"""
vLLM Benchmark Template - Like train.py in Karpathy's autoresearch.

This file is MODIFIED by the agent each iteration.
The agent changes the configuration parameters below to test different settings.
"""

import subprocess
import sys
import re
from pathlib import Path

# ============================================================================
# CONFIGURATION PARAMETERS - AGENT MODIFIES THESE
# ============================================================================

# Model configuration (usually fixed by user)
MODEL_NAME = "Qwen/Qwen3.5-9B"
CONTEXT_LENGTH = 16384
PRECISION = "fp16"

# vLLM parameters to optimize
GPU_MEMORY_UTILIZATION = 0.90  # Agent modifies: [0.80 - 0.98]
MAX_NUM_SEQS = 256             # Agent modifies: [64 - 16384]
TENSOR_PARALLEL_SIZE = 1       # Agent modifies: [1, 2, 4, 8, 16]
BLOCK_SIZE = 16                # Agent modifies: [8, 16, 32]
SWAP_SPACE = 4                 # Agent modifies: [0 - 16] GB

# Benchmark parameters (usually fixed)
INPUT_LENGTH = 1024
OUTPUT_LENGTH = 512
NUM_PROMPTS = 50
TIMEOUT = 300  # 5 minutes

# ============================================================================
# BENCHMARK EXECUTION - FIXED (never modified by agent)
# ============================================================================


def run_benchmark() -> dict:
    """
    Run vLLM throughput benchmark with current configuration.

    Returns:
        dict with keys:
            - throughput: float (tokens per second)
            - status: str ("success", "oom", "timeout", "error")
            - error: str (if failed)
    """
    cmd = [
        sys.executable,
        "-m",
        "vllm.benchmarks.benchmark_throughput",
        "--model",
        MODEL_NAME,
        "--input-len",
        str(INPUT_LENGTH),
        "--output-len",
        str(OUTPUT_LENGTH),
        "--num-prompts",
        str(NUM_PROMPTS),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--block-size",
        str(BLOCK_SIZE),
        "--swap-space",
        str(SWAP_SPACE),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=None,  # Inherit environment
        )

        # Parse throughput from output
        for line in result.stdout.split("\n"):
            if "Throughput:" in line:
                # Example: "Throughput: 1234.56 tokens/s"
                match = re.search(r"Throughput:\s+([\d.]+)", line)
                if match:
                    throughput = float(match.group(1))
                    return {
                        "throughput": throughput,
                        "status": "success",
                    }

        # Throughput not found in output
        return {
            "throughput": 0.0,
            "status": "error",
            "error": "Could not parse throughput from benchmark output",
        }

    except subprocess.TimeoutExpired:
        return {
            "throughput": 0.0,
            "status": "timeout",
            "error": f"Benchmark exceeded {TIMEOUT}s timeout",
        }

    except Exception as e:
        error_msg = str(e).lower()

        # Check if OOM
        if "out of memory" in error_msg or "oom" in error_msg:
            return {
                "throughput": 0.0,
                "status": "oom",
                "error": "Out of memory (OOM) error",
            }

        return {
            "throughput": 0.0,
            "status": "error",
            "error": str(e),
        }


def get_current_config() -> dict:
    """Return current configuration as dict."""
    return {
        "model_name": MODEL_NAME,
        "context_length": CONTEXT_LENGTH,
        "precision": PRECISION,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_seqs": MAX_NUM_SEQS,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "block_size": BLOCK_SIZE,
        "swap_space": SWAP_SPACE,
    }


if __name__ == "__main__":
    # Run benchmark and print result
    print("\n" + "=" * 60)
    print("vLLM BENCHMARK")
    print("=" * 60)

    print("\nConfiguration:")
    config = get_current_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nRunning benchmark...")
    result = run_benchmark()

    print("\nResult:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("=" * 60 + "\n")

    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)
