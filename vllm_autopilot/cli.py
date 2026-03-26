"""
Command-line interface for vLLM Autopilot.
"""

import argparse
import sys
from pathlib import Path

from vllm_autopilot.orchestrator import optimize
from vllm_autopilot.database import ConfigDatabase
from vllm_autopilot.hardware import detect_hardware, print_hardware_info


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="vLLM Autopilot - Automatic vLLM configuration optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  vllm-autopilot --model Qwen/Qwen3.5-9B

  # Custom context length
  vllm-autopilot --model meta-llama/Llama-3.1-8B --context-length 8192

  # Quick mode (fewer iterations)
  vllm-autopilot --model Qwen/Qwen3.5-9B --max-iterations 10

  # Query database
  vllm-autopilot --query --model Qwen/Qwen3.5-9B

  # List all configs
  vllm-autopilot --list-configs
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name (e.g., Qwen/Qwen3.5-9B)",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=16384,
        help="Context length in tokens (default: 16384)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Maximum optimization iterations (default: 25)",
    )

    parser.add_argument(
        "--target-percentile",
        type=float,
        default=0.95,
        help="Target percentile to stop optimization (default: 0.95)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./vllm_optimization"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--query",
        action="store_true",
        help="Query database for existing config (no optimization)",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all configs in database",
    )

    parser.add_argument(
        "--hardware-info",
        action="store_true",
        help="Print hardware information and exit",
    )

    args = parser.parse_args()

    # Hardware info
    if args.hardware_info:
        hw = detect_hardware()
        print_hardware_info(hw)
        return 0

    # List configs
    if args.list_configs:
        db = ConfigDatabase()
        configs = db.list_configs()
        print(f"\nFound {len(configs)} configurations in database:\n")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config['model_name']} on {config['num_gpus']}× {config['gpu_model']}")
            print(f"   Context: {config['context_length']}, Throughput: {config['throughput_tokens_per_sec']:.1f} tok/s")
            print(f"   Verified by: {config['verified_count']} user(s)\n")
        return 0

    # Query mode
    if args.query:
        if not args.model:
            print("Error: --model required for query mode")
            return 1

        hw = detect_hardware()
        db = ConfigDatabase()

        result = db.query(
            gpu_model=hw.gpus[0].name,
            model_name=args.model,
            context_length=args.context_length,
            num_gpus=hw.num_gpus,
        )

        if result:
            print("✓ Found config in database:")
            db.print_config(result)
        else:
            print(f"✗ No config found for:")
            print(f"  Model: {args.model}")
            print(f"  Context: {args.context_length}")
            print(f"  GPU: {hw.gpus[0].name}")
            print(f"\nRun without --query to start optimization.")

        return 0

    # Optimization mode
    if not args.model:
        print("Error: --model required")
        parser.print_help()
        return 1

    print("\n🚀 Starting vLLM Autopilot optimization...\n")

    try:
        config = optimize(
            model=args.model,
            context_length=args.context_length,
            max_iterations=args.max_iterations,
        )

        print("\n✓ Optimization complete!")
        print(f"  Results saved to: {args.output_dir / 'optimal_config.json'}")

        return 0

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
