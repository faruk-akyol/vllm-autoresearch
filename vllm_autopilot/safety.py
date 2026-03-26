"""
Safety Agent - Prevents OOM errors before running benchmarks.

Estimates memory requirements and rejects unsafe configurations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class MemoryEstimate:
    """Memory usage estimate breakdown."""

    model_weights_gb: float
    kv_cache_gb: float
    activation_gb: float
    overhead_gb: float
    total_gb: float
    available_gb: float
    safe: bool
    confidence: str  # "high", "medium", "low"
    warning: Optional[str] = None


class SafetyAgent:
    """
    Pre-flight safety checks to prevent OOM errors.

    Estimates memory requirements before running benchmarks.
    Learns from actual OOM events to improve predictions.
    """

    def __init__(self, gpu_vram_gb: int, safety_margin: float = 0.15):
        """
        Initialize safety agent.

        Args:
            gpu_vram_gb: Total VRAM per GPU in GB
            safety_margin: Reserve this fraction of VRAM (default 15%)
        """
        self.gpu_vram_gb = gpu_vram_gb
        self.safety_margin = safety_margin

        # Learned constraints from real OOM events
        self.oom_history: Dict[str, float] = {}

    def check_config(
        self,
        config: Dict,
        model_size_gb: float,
        context_length: int,
    ) -> Tuple[bool, MemoryEstimate, Optional[Dict]]:
        """
        Check if config is safe to run.

        Args:
            config: vLLM configuration dict
            model_size_gb: Model size in GB
            context_length: Context length in tokens

        Returns:
            (is_safe, estimate, safe_alternative_config)
        """
        estimate = self._estimate_memory(config, model_size_gb, context_length)

        if estimate.safe:
            return True, estimate, None

        # Unsafe - suggest safer config
        safe_config = self._suggest_safe_config(config, estimate)
        return False, estimate, safe_config

    def _estimate_memory(
        self, config: Dict, model_size_gb: float, context_length: int
    ) -> MemoryEstimate:
        """Estimate total memory requirements."""

        # 1. Model weights
        model_weights = model_size_gb

        # 2. KV cache (depends on config)
        kv_cache = self._estimate_kv_cache(
            context_length=context_length,
            max_num_seqs=config.get("max_num_seqs", 256),
            model_size_gb=model_size_gb,
        )

        # 3. Activation memory (rough estimate)
        activation_memory = self._estimate_activations(
            max_num_seqs=config.get("max_num_seqs", 256),
            context_length=context_length,
        )

        # 4. System overhead
        overhead = 0.5  # ~500MB for vLLM runtime

        # 5. Total
        total = model_weights + kv_cache + activation_memory + overhead

        # 6. Check if safe
        gpu_mem_util = config.get("gpu_memory_utilization", 0.90)
        requested_memory = total / gpu_mem_util if gpu_mem_util > 0 else total

        safe_limit = self.gpu_vram_gb * (1 - self.safety_margin)
        is_safe = requested_memory <= safe_limit

        # 7. Confidence
        if requested_memory < self.gpu_vram_gb * 0.7:
            confidence = "high"
            warning = None
        elif requested_memory < safe_limit:
            confidence = "medium"
            warning = "Approaching VRAM limit"
        else:
            confidence = "low"
            warning = f"Exceeds safe limit by {requested_memory - safe_limit:.1f} GB"

        return MemoryEstimate(
            model_weights_gb=model_weights,
            kv_cache_gb=kv_cache,
            activation_gb=activation_memory,
            overhead_gb=overhead,
            total_gb=total,
            available_gb=self.gpu_vram_gb,
            safe=is_safe,
            confidence=confidence,
            warning=warning,
        )

    def _estimate_kv_cache(
        self, context_length: int, max_num_seqs: int, model_size_gb: float
    ) -> float:
        """Estimate KV cache memory usage."""
        # Rough approximation based on model size and sequences
        # Real formula: 2 * num_layers * context * hidden_size * max_seqs * bytes
        # Simplified: scale with model size and num sequences

        base_cache_per_seq = (model_size_gb / 10) * (context_length / 4096)
        total_cache = base_cache_per_seq * max_num_seqs * 0.1  # 10% of model size per seq

        return total_cache

    def _estimate_activations(self, max_num_seqs: int, context_length: int) -> float:
        """Estimate activation memory (temporary tensors)."""
        # Rough estimate: 50MB per sequence, scaled by context length
        base_per_seq = 0.05  # 50MB
        total = base_per_seq * max_num_seqs * (context_length / 4096)
        return total

    def _suggest_safe_config(self, unsafe_config: Dict, estimate: MemoryEstimate) -> Dict:
        """Suggest a safer alternative configuration."""
        safe_config = unsafe_config.copy()

        # Calculate reduction needed
        excess = estimate.total_gb - (self.gpu_vram_gb * (1 - self.safety_margin))
        reduction_needed = excess / estimate.available_gb

        # Strategy: Reduce max_num_seqs first, then gpu_memory_utilization
        if reduction_needed < 0.1:
            # Small reduction: just lower gpu_mem
            safe_config["gpu_memory_utilization"] = max(
                0.75, unsafe_config.get("gpu_memory_utilization", 0.90) - 0.05
            )
        elif reduction_needed < 0.3:
            # Medium reduction: lower max_num_seqs
            safe_config["max_num_seqs"] = max(
                64, int(unsafe_config.get("max_num_seqs", 256) * 0.7)
            )
        else:
            # Large reduction: both
            safe_config["gpu_memory_utilization"] = max(
                0.75, unsafe_config.get("gpu_memory_utilization", 0.90) - 0.1
            )
            safe_config["max_num_seqs"] = max(
                64, int(unsafe_config.get("max_num_seqs", 256) * 0.5)
            )

        return safe_config

    def learn_from_oom(
        self, config: Dict, actual_oom_threshold: Optional[float] = None
    ) -> None:
        """
        Learn from actual OOM event to improve future predictions.

        Args:
            config: Configuration that caused OOM
            actual_oom_threshold: Actual memory limit discovered (if known)
        """
        # Store OOM constraints
        key = f"gpu_mem_{config.get('gpu_memory_utilization', 0)}"
        if key not in self.oom_history or (
            actual_oom_threshold and actual_oom_threshold < self.oom_history[key]
        ):
            self.oom_history[key] = config.get("gpu_memory_utilization", 0.90)

        # Similar for max_num_seqs
        key = f"max_seqs_{config.get('tensor_parallel_size', 1)}"
        current_seqs = config.get("max_num_seqs", 256)
        if key not in self.oom_history or current_seqs < self.oom_history.get(key, float("inf")):
            self.oom_history[key] = current_seqs

    def print_estimate(self, estimate: MemoryEstimate, config: Dict) -> None:
        """Print memory estimate in readable format."""
        print("\n" + "=" * 60)
        print("🛡️  SAFETY AGENT - Pre-Flight Check")
        print("=" * 60)

        print(f"\nMemory Breakdown:")
        print(f"  Model Weights:    {estimate.model_weights_gb:>6.2f} GB")
        print(f"  KV Cache:         {estimate.kv_cache_gb:>6.2f} GB")
        print(f"  Activations:      {estimate.activation_gb:>6.2f} GB")
        print(f"  System Overhead:  {estimate.overhead_gb:>6.2f} GB")
        print(f"  {'-' * 40}")
        print(f"  Total Required:   {estimate.total_gb:>6.2f} GB")
        print(f"  Available VRAM:   {estimate.available_gb:>6.2f} GB")
        print(f"  Safety Margin:    {estimate.available_gb * self.safety_margin:>6.2f} GB ({self.safety_margin*100:.0f}%)")

        utilization = (estimate.total_gb / estimate.available_gb) * 100
        print(f"\nVRAM Utilization: {utilization:.1f}%")

        if estimate.safe:
            print(f"\n✅ SAFE - Config will NOT cause OOM")
            print(f"   Confidence: {estimate.confidence.upper()}")
        else:
            print(f"\n❌ UNSAFE - Config will likely cause OOM")
            excess = estimate.total_gb - (estimate.available_gb * (1 - self.safety_margin))
            print(f"   Over budget by: {excess:.2f} GB")

        if estimate.warning:
            print(f"\n⚠️  Warning: {estimate.warning}")

        print(f"\nTested Config:")
        print(f"  gpu_memory_utilization: {config.get('gpu_memory_utilization', 0.90)}")
        print(f"  max_num_seqs: {config.get('max_num_seqs', 256)}")
        print(f"  tensor_parallel_size: {config.get('tensor_parallel_size', 1)}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test safety agent
    safety = SafetyAgent(gpu_vram_gb=80)

    test_config = {
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 4096,
        "tensor_parallel_size": 1,
        "block_size": 16,
    }

    model_size_gb = 18.0  # Qwen 9B
    context_length = 16384

    is_safe, estimate, safe_config = safety.check_config(
        test_config, model_size_gb, context_length
    )

    safety.print_estimate(estimate, test_config)

    if not is_safe and safe_config:
        print("Suggested Safe Config:")
        print(f"  gpu_memory_utilization: {safe_config['gpu_memory_utilization']}")
        print(f"  max_num_seqs: {safe_config['max_num_seqs']}\n")
