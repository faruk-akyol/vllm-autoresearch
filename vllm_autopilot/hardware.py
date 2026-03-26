"""
Hardware detection - Like prepare.py in Karpathy's autoresearch.

This file is FIXED and never modified by the agent.
Handles GPU detection, CUDA version checking, vLLM validation.
"""

import subprocess
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU hardware information."""

    name: str
    vram_gb: int
    compute_capability: str
    index: int


@dataclass
class HardwareConfig:
    """Complete hardware configuration."""

    gpus: List[GPUInfo]
    num_gpus: int
    cuda_version: str
    vllm_version: str
    total_vram_gb: int


def detect_gpus() -> List[GPUInfo]:
    """Detect all NVIDIA GPUs in the system."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            index, name, vram_mb, compute_cap = line.split(", ")
            gpus.append(
                GPUInfo(
                    name=name.strip(),
                    vram_gb=int(float(vram_mb) / 1024),
                    compute_capability=compute_cap.strip(),
                    index=int(index),
                )
            )

        return gpus

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to detect GPUs. Is nvidia-smi available? {e}")


def detect_cuda_version() -> str:
    """Detect CUDA version."""
    try:
        # Try nvcc first
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        # Fallback to nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        pass

    return "unknown"


def detect_vllm_version() -> str:
    """Detect installed vLLM version."""
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        raise RuntimeError(
            "vLLM not installed. Please install with: pip install vllm"
        )


def detect_hardware() -> HardwareConfig:
    """Detect complete hardware configuration."""
    gpus = detect_gpus()

    if not gpus:
        raise RuntimeError("No NVIDIA GPUs detected!")

    return HardwareConfig(
        gpus=gpus,
        num_gpus=len(gpus),
        cuda_version=detect_cuda_version(),
        vllm_version=detect_vllm_version(),
        total_vram_gb=sum(gpu.vram_gb for gpu in gpus),
    )


def print_hardware_info(hw: HardwareConfig) -> None:
    """Print hardware information in a nice format."""
    print("=" * 60)
    print("HARDWARE DETECTION")
    print("=" * 60)
    print(f"\nGPUs Detected: {hw.num_gpus}")
    for gpu in hw.gpus:
        print(f"  [{gpu.index}] {gpu.name}")
        print(f"      VRAM: {gpu.vram_gb} GB")
        print(f"      Compute: {gpu.compute_capability}")

    print(f"\nTotal VRAM: {hw.total_vram_gb} GB")
    print(f"CUDA Version: {hw.cuda_version}")
    print(f"vLLM Version: {hw.vllm_version}")
    print("=" * 60 + "\n")


def can_fit_on_single_gpu(model_size_gb: float, gpu_vram_gb: int) -> bool:
    """Check if model can fit on a single GPU."""
    # Leave 20% headroom for activations, KV cache, etc.
    usable_vram = gpu_vram_gb * 0.8
    return model_size_gb <= usable_vram


def estimate_model_size(model_name: str, precision: str = "fp16") -> float:
    """Estimate model size in GB based on name and precision."""
    # Extract parameter count from model name
    match = re.search(r"(\d+\.?\d*)[Bb]", model_name)
    if match:
        params_billions = float(match.group(1))
    else:
        # Common model sizes
        size_map = {
            "7b": 7,
            "8b": 8,
            "9b": 9,
            "13b": 13,
            "14b": 14,
            "32b": 32,
            "34b": 34,
            "70b": 70,
            "72b": 72,
        }
        model_lower = model_name.lower()
        for key, val in size_map.items():
            if key in model_lower:
                params_billions = val
                break
        else:
            # Default to 14B if unknown
            params_billions = 14
            print(f"⚠️  Unknown model size for {model_name}, assuming {params_billions}B")

    # Bytes per parameter
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    bpp = bytes_per_param.get(precision, 2)
    size_gb = (params_billions * 1e9 * bpp) / (1024**3)

    return size_gb


if __name__ == "__main__":
    # Test hardware detection
    hw = detect_hardware()
    print_hardware_info(hw)

    # Test model size estimation
    test_models = [
        "Qwen/Qwen3.5-9B",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
    ]

    print("\nModel Size Estimates:")
    for model in test_models:
        size = estimate_model_size(model)
        print(f"  {model}: ~{size:.1f} GB (fp16)")
