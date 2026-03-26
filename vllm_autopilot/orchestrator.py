"""
Orchestrator - Main optimization loop.

Like Karpathy's autoresearch loop, but for vLLM optimization.
"""

import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import subprocess
import sys

from vllm_autopilot.hardware import detect_hardware, estimate_model_size, can_fit_on_single_gpu
from vllm_autopilot.agent import ResearchAgent
from vllm_autopilot.safety import SafetyAgent
from vllm_autopilot.database import ConfigDatabase


@dataclass
class ExperimentResult:
    """Single experiment result."""

    iteration: int
    gpu_id: int
    config: Dict
    result: Dict  # {"throughput": float, "status": str}
    timestamp: str


class VLLMOrchestrator:
    """
    Main orchestrator for vLLM optimization.

    Coordinates agents, runs experiments, tracks convergence.
    """

    def __init__(
        self,
        model_name: str,
        context_length: int = 16384,
        max_iterations: int = 25,
        target_percentile: float = 0.95,
        output_dir: Path = Path("./vllm_optimization"),
    ):
        """
        Initialize orchestrator.

        Args:
            model_name: HuggingFace model name
            context_length: Context length in tokens
            max_iterations: Maximum optimization iterations
            target_percentile: Stop at this percentile (e.g., 0.95 = 95%)
            output_dir: Output directory for results
        """
        self.model_name = model_name
        self.context_length = context_length
        self.max_iterations = max_iterations
        self.target_percentile = target_percentile
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect hardware
        print("Detecting hardware...")
        self.hardware = detect_hardware()
        self.model_size_gb = estimate_model_size(model_name)

        # Initialize agents
        self.agent = ResearchAgent()
        self.safety = SafetyAgent(gpu_vram_gb=self.hardware.gpus[0].vram_gb)
        self.database = ConfigDatabase()

        # State
        self.history: List[ExperimentResult] = []
        self.best_config: Optional[Dict] = None
        self.best_throughput: float = 0.0
        self.estimated_max_throughput: Optional[float] = None
        self.convergence_count = 0

        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"

    def run(self) -> Dict:
        """
        Run complete optimization.

        Returns:
            Optimal configuration dict
        """
        print("\n" + "=" * 70)
        print("vLLM AUTOPILOT - Automatic Configuration Optimization")
        print("=" * 70)

        print(f"\nModel: {self.model_name} (~{self.model_size_gb:.1f} GB)")
        print(f"Context: {self.context_length} tokens")
        print(f"Hardware: {self.hardware.num_gpus}× {self.hardware.gpus[0].name}")
        print(f"Target: {self.target_percentile*100:.0f}th percentile")
        print(f"Max iterations: {self.max_iterations}")

        # Check database first
        print("\nQuerying database for existing config...")
        existing = self.database.query(
            gpu_model=self.hardware.gpus[0].name,
            model_name=self.model_name,
            context_length=self.context_length,
            num_gpus=self.hardware.num_gpus,
        )

        if existing:
            print("✓ Found existing config in database!")
            self.database.print_config(existing)
            user_input = input("Use this config? (y/n): ")
            if user_input.lower() == "y":
                return existing["config"]
            print("Starting fresh optimization...")

        # Calculate parallelism
        can_parallel = can_fit_on_single_gpu(self.model_size_gb, self.hardware.gpus[0].vram_gb)
        num_parallel = self.hardware.num_gpus if can_parallel else 1

        print(f"\nParallel experiments: {num_parallel} (model fits on 1 GPU: {can_parallel})")
        print("\n" + "=" * 70 + "\n")

        # Main optimization loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'=' * 70}")
            print(f"Iteration {iteration}/{self.max_iterations}")
            print(f"{'=' * 70}\n")

            # Generate experiments
            print(f"Agent: Generating {num_parallel} experiments...")
            agent_output = self.agent.generate_experiments(
                iteration=iteration,
                history=[asdict(h) for h in self.history],
                num_experiments=num_parallel,
                hardware_info={
                    "num_gpus": self.hardware.num_gpus,
                    "gpu_model": self.hardware.gpus[0].name,
                    "vram_per_gpu": self.hardware.gpus[0].vram_gb,
                },
                model_info={
                    "model_name": self.model_name,
                    "size_gb": self.model_size_gb,
                    "context_length": self.context_length,
                },
            )

            print(f"\nHypothesis: {agent_output['hypothesis']}")
            print(f"Prediction: {agent_output['prediction']}\n")

            # Safety check experiments
            safe_experiments = []
            for exp_id, config in enumerate(agent_output["experiments"]):
                is_safe, estimate, safe_config = self.safety.check_config(
                    config, self.model_size_gb, self.context_length
                )

                if not is_safe:
                    print(f"⚠️  Experiment {exp_id}: UNSAFE config detected")
                    self.safety.print_estimate(estimate, config)
                    print(f"   Using safer alternative...")
                    config = safe_config

                safe_experiments.append(config)

            # Run experiments
            print(f"\nRunning {len(safe_experiments)} benchmarks...")
            results = self._run_parallel_experiments(safe_experiments, iteration)

            # Analyze results
            for result in results:
                if result.result["status"] == "success":
                    throughput = result.result["throughput"]
                    print(
                        f"  GPU {result.gpu_id}: {throughput:.1f} tok/s "
                        f"(seqs={result.config['max_num_seqs']}, "
                        f"gpu_mem={result.config['gpu_memory_utilization']})"
                    )

                    # Update best
                    if throughput > self.best_throughput:
                        self.best_throughput = throughput
                        self.best_config = result.config
                        print(f"    🚀 New best!")
                        self.convergence_count = 0
                    else:
                        improvement = (
                            (throughput - self.best_throughput) / self.best_throughput * 100
                        )
                        if abs(improvement) < 1.0:
                            self.convergence_count += 1

                elif result.result["status"] == "oom":
                    print(f"  GPU {result.gpu_id}: OOM - config too aggressive")
                    self.safety.learn_from_oom(result.config)
                else:
                    print(f"  GPU {result.gpu_id}: {result.result['status']}")

            # Save checkpoint
            self._save_checkpoint(iteration)

            # Check 95th percentile target
            if iteration >= 10 and self.estimated_max_throughput is None:
                # Estimate max based on growth rate
                recent_improvements = [
                    (r.result["throughput"] - self.history[i - 1].result["throughput"])
                    / self.history[i - 1].result["throughput"]
                    for i, r in enumerate(self.history[-5:], start=len(self.history) - 4)
                    if i > 0
                    and r.result["status"] == "success"
                    and self.history[i - 1].result["status"] == "success"
                ]

                if recent_improvements:
                    avg_improvement = sum(recent_improvements) / len(recent_improvements)
                    self.estimated_max_throughput = self.best_throughput * (
                        1 + avg_improvement * 3
                    )
                    print(f"\n📊 Estimated max throughput: {self.estimated_max_throughput:.0f} tok/s")
                    print(f"   95th percentile target: {self.estimated_max_throughput * self.target_percentile:.0f} tok/s")

            # Check if reached target
            if (
                self.estimated_max_throughput
                and self.best_throughput >= self.estimated_max_throughput * self.target_percentile
            ):
                print(f"\n✓ Reached {self.target_percentile*100:.0f}th percentile target!")
                print(f"   Current: {self.best_throughput:.0f} tok/s")
                print(f"   Target: {self.estimated_max_throughput * self.target_percentile:.0f} tok/s")
                break

            # Check convergence
            if self.convergence_count >= 5:
                print(f"\n✓ Converged after {iteration} iterations")
                print(f"   Last 5 iterations showed < 1% improvement")
                break

        # Final results
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        print(f"\nOptimal Configuration:")
        print(json.dumps(self.best_config, indent=2))
        print(f"\nThroughput: {self.best_throughput:.1f} tokens/sec")
        print(f"Iterations: {len(self.history)}")

        # Save to database
        save_input = input("\nSave this config to community database? (y/n): ")
        if save_input.lower() == "y":
            self.database.save_config(
                gpu_model=self.hardware.gpus[0].name,
                num_gpus=self.hardware.num_gpus,
                vram_per_gpu=self.hardware.gpus[0].vram_gb,
                model_name=self.model_name,
                model_size_gb=self.model_size_gb,
                context_length=self.context_length,
                config=self.best_config,
                throughput=self.best_throughput,
            )

        # Save final config
        final_config_file = self.output_dir / "optimal_config.json"
        with open(final_config_file, "w") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "config": self.best_config,
                    "throughput": self.best_throughput,
                },
                f,
                indent=2,
            )
        print(f"\n✓ Saved optimal config to {final_config_file}")

        return self.best_config

    def _run_parallel_experiments(
        self, configs: List[Dict], iteration: int
    ) -> List[ExperimentResult]:
        """Run experiments in parallel across GPUs."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        # Determine number of parallel workers
        num_workers = min(len(configs), self.hardware.num_gpus)

        print(f"Running {len(configs)} experiments in parallel on {num_workers} GPUs...")

        # Create process pool
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all experiments
            futures = {}
            for gpu_id, config in enumerate(configs):
                future = executor.submit(
                    self._run_single_experiment_worker,
                    gpu_id % self.hardware.num_gpus,  # Assign to GPU
                    config,
                    iteration,
                    self.model_name,
                    self.context_length,
                    self.output_dir,
                )
                futures[future] = (gpu_id, config)

            # Collect results as they complete
            for future in as_completed(futures):
                gpu_id, config = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Print progress
                    if result.result["status"] == "success":
                        print(f"  ✓ GPU {result.gpu_id}: {result.result['throughput']:.1f} tok/s")
                    else:
                        print(f"  ✗ GPU {result.gpu_id}: {result.result['status']}")

                except Exception as e:
                    print(f"  ✗ GPU {gpu_id}: Exception - {e}")
                    # Create failed result
                    result = ExperimentResult(
                        iteration=iteration,
                        gpu_id=gpu_id,
                        config=config,
                        result={"throughput": 0.0, "status": "error", "error": str(e)},
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    results.append(result)

        return results

    @staticmethod
    def _run_single_experiment_worker(
        gpu_id: int,
        config: Dict,
        iteration: int,
        model_name: str,
        context_length: int,
        output_dir: Path,
    ) -> ExperimentResult:
        """
        Worker function to run single experiment on specific GPU.

        This is a static method so it can be pickled for multiprocessing.
        """
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        # Set CUDA_VISIBLE_DEVICES to assign this process to specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Create modified benchmark file for this GPU
        template_path = Path(__file__).parent / "benchmark_template.py"
        benchmark_path = output_dir / f"benchmark_gpu{gpu_id}_iter{iteration}.py"

        # Read template
        template_code = template_path.read_text()

        # Replace model and context
        template_code = template_code.replace(
            'MODEL_NAME = "Qwen/Qwen3.5-9B"',
            f'MODEL_NAME = "{model_name}"'
        )
        template_code = template_code.replace(
            'CONTEXT_LENGTH = 16384',
            f'CONTEXT_LENGTH = {context_length}'
        )

        # Replace configuration parameters
        for key, value in config.items():
            pattern = f"{key.upper()} = "
            if pattern in template_code:
                lines = template_code.split("\n")
                new_lines = []
                for line in lines:
                    if line.strip().startswith(pattern) and not line.strip().startswith("#"):
                        new_lines.append(f"{key.upper()} = {value}")
                    else:
                        new_lines.append(line)
                template_code = "\n".join(new_lines)

        # Write modified benchmark
        benchmark_path.write_text(template_code)

        # Run benchmark
        try:
            result = subprocess.run(
                [sys.executable, str(benchmark_path)],
                capture_output=True,
                text=True,
                timeout=360,  # 6 min timeout
                env=os.environ.copy(),  # Pass environment with CUDA_VISIBLE_DEVICES
            )

            # Parse throughput from output
            for line in result.stdout.split("\n"):
                if "throughput:" in line.lower():
                    import re
                    match = re.search(r"([\d.]+)", line)
                    if match:
                        throughput = float(match.group(1))
                        return ExperimentResult(
                            iteration=iteration,
                            gpu_id=gpu_id,
                            config=config,
                            result={"throughput": throughput, "status": "success"},
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )

            # Throughput not found
            return ExperimentResult(
                iteration=iteration,
                gpu_id=gpu_id,
                config=config,
                result={"throughput": 0.0, "status": "error", "error": "Could not parse output"},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        except subprocess.TimeoutExpired:
            return ExperimentResult(
                iteration=iteration,
                gpu_id=gpu_id,
                config=config,
                result={"throughput": 0.0, "status": "timeout"},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                status = "oom"
            else:
                status = "error"

            return ExperimentResult(
                iteration=iteration,
                gpu_id=gpu_id,
                config=config,
                result={"throughput": 0.0, "status": status, "error": str(e)},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )


    def _save_checkpoint(self, iteration: int) -> None:
        """Save checkpoint after each iteration."""
        checkpoint = {
            "iteration": iteration,
            "best_config": self.best_config,
            "best_throughput": self.best_throughput,
            "estimated_max": self.estimated_max_throughput,
            "convergence_count": self.convergence_count,
            "history": [asdict(h) for h in self.history],
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)


def optimize(
    model: str,
    context_length: int = 16384,
    max_iterations: int = 25,
) -> Dict:
    """
    Main entry point for optimization.

    Args:
        model: HuggingFace model name
        context_length: Context length in tokens
        max_iterations: Maximum iterations

    Returns:
        Optimal configuration dict
    """
    orchestrator = VLLMOrchestrator(
        model_name=model,
        context_length=context_length,
        max_iterations=max_iterations,
    )

    return orchestrator.run()


if __name__ == "__main__":
    # Test orchestrator
    config = optimize(
        model="Qwen/Qwen3.5-9B",
        context_length=16384,
        max_iterations=5,  # Short test
    )

    print("\nFinal config:")
    print(json.dumps(config, indent=2))
