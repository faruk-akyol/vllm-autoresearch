"""
Example: Optimize Qwen3.5-9B model

This script demonstrates how to optimize vLLM configuration
for Qwen3.5-9B model using vLLM Autopilot.
"""

from vllm_autopilot import optimize
from vllm_autopilot.database import ConfigDatabase
from vllm_autopilot.hardware import detect_gpus
import json


def main():
    # 1. Check hardware
    print("🔍 Detecting hardware...")
    gpus = detect_gpus()
    print(f"✓ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - GPU {gpu.index}: {gpu.name} ({gpu.vram_gb} GB)")
    print()

    # 2. Check database first
    print("📊 Checking database for existing config...")
    db = ConfigDatabase()

    existing_config = db.query(
        gpu_model=gpus[0].name,
        model_name="Qwen/Qwen3.5-9B",
        context_length=16384,
        num_gpus=len(gpus),
        exact_match=False  # Allow similar configs
    )

    if existing_config:
        print(f"✓ Found existing config!")
        print(f"  Throughput: {existing_config['throughput']:.2f} tokens/sec")
        print(f"  Config: {json.dumps(existing_config['config'], indent=2)}")

        use_existing = input("\n Use this config? (y/n): ").lower().strip()
        if use_existing == 'y':
            print("\n✓ Using existing config from database.")
            return existing_config
    else:
        print("✗ No existing config found.")

    print()

    # 3. Run optimization
    print("🚀 Starting optimization...")
    print("   This will take approximately 15-25 minutes.")
    print("   You can stop (Ctrl+C) and resume later.\n")

    optimal_config = optimize(
        model="Qwen/Qwen3.5-9B",
        context_length=16384,
        max_iterations=25,
        target_percentile=0.95,
        output_dir="./vllm_optimization",
        log_file="optimization.log",
        resume_from_checkpoint=True
    )

    # 4. Display results
    print("\n" + "="*60)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\nOptimal Configuration:")
    print(json.dumps(optimal_config['config'], indent=2))
    print(f"\nThroughput: {optimal_config['throughput']:.2f} tokens/sec")
    print(f"Iterations: {optimal_config['iterations']}")
    print(f"Total time: {optimal_config['total_time_seconds']:.0f} seconds")

    # 5. Show how to use the config
    print("\n" + "="*60)
    print("📝 HOW TO USE THIS CONFIG")
    print("="*60)

    config = optimal_config['config']

    print("\nOption 1: vLLM CLI")
    print("-" * 60)
    print(f"""vllm serve Qwen/Qwen3.5-9B \\
  --gpu-memory-utilization {config['gpu_memory_utilization']} \\
  --max-num-seqs {config['max_num_seqs']} \\
  --tensor-parallel-size {config['tensor_parallel_size']} \\
  --block-size {config['block_size']} \\
  --swap-space {config['swap_space']} \\
  --port 8000
""")

    print("\nOption 2: Python API")
    print("-" * 60)
    print(f"""from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3.5-9B",
    gpu_memory_utilization={config['gpu_memory_utilization']},
    max_num_seqs={config['max_num_seqs']},
    tensor_parallel_size={config['tensor_parallel_size']},
    block_size={config['block_size']},
    swap_space={config['swap_space']}
)

prompts = ["Your prompt here"]
outputs = llm.generate(prompts)
""")

    # 6. Ask to save to database
    save_to_db = input("\n💾 Save to community database? (y/n): ").lower().strip()
    if save_to_db == 'y':
        db.save_config(
            gpu_model=gpus[0].name,
            num_gpus=len(gpus),
            model_name="Qwen/Qwen3.5-9B",
            context_length=16384,
            config=config,
            throughput=optimal_config['throughput']
        )
        print("✓ Saved to database! Others can now benefit from your optimization.")

    print("\n✨ Done! Enjoy your optimized vLLM setup.")

    return optimal_config


if __name__ == "__main__":
    # Make sure ANTHROPIC_API_KEY is set
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ERROR: ANTHROPIC_API_KEY not set!")
        print("\nPlease set your API key:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    result = main()
