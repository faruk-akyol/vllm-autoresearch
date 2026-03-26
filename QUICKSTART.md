# vLLM Autopilot - Quick Start Guide

## Installation

```bash
cd vllm-autopilot
pip install -e .
```

## Prerequisites

1. **vLLM installed:**
   ```bash
   pip install vllm
   ```

2. **Anthropic API key:**
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. **NVIDIA GPU with CUDA**

## Basic Usage

### Option 1: Command Line

```bash
# Optimize a model
vllm-autopilot --model Qwen/Qwen3.5-9B --context-length 16384

# Quick mode (fewer iterations)
vllm-autopilot --model Qwen/Qwen3.5-9B --max-iterations 10

# Query database first (check for existing config)
vllm-autopilot --query --model Qwen/Qwen3.5-9B

# List all configs in database
vllm-autopilot --list-configs
```

### Option 2: Python API

```python
from vllm_autopilot import optimize

config = optimize(
    model="Qwen/Qwen3.5-9B",
    context_length=16384,
    max_iterations=20,
)

print(f"Optimal config: {config}")
```

### Option 3: Run Example

```bash
python examples/optimize_qwen.py
```

## What Happens

1. **Hardware Detection** - Detects your GPUs automatically
2. **Database Query** - Checks if config already exists
3. **Safety Check** - Validates configs before running (prevents OOM)
4. **Parallel Experiments** - Runs 8 experiments at once (if multi-GPU + small model)
5. **LLM Agent** - Claude analyzes results and generates next experiments
6. **Convergence** - Stops at 95th percentile (good enough!)
7. **Save Results** - Saves optimal config and offers to share with community

## Expected Timeline

- **Single GPU**: 20-30 minutes (sequential experiments)
- **8× GPU (small model)**: 15-20 minutes (8 parallel experiments)
- **8× GPU (large model)**: 25-30 minutes (limited parallelism)

## Output

After optimization, you'll get:

### 1. Optimal Config File
`vllm_optimization/optimal_config.json`:
```json
{
  "model": "Qwen/Qwen3.5-9B",
  "config": {
    "gpu_memory_utilization": 0.95,
    "max_num_seqs": 16384,
    "tensor_parallel_size": 8,
    "block_size": 16,
    "swap_space": 4
  },
  "throughput": 27500.0
}
```

### 2. vLLM Startup Command
```bash
vllm serve Qwen/Qwen3.5-9B \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 16384 \
  --tensor-parallel-size 8 \
  --block-size 16 \
  --swap-space 4 \
  --port 8000
```

### 3. Performance Metrics
- **Throughput**: 27,500 tokens/sec
- **Speedup**: 30-40× vs naive config
- **Iterations**: 18 (converged early)
- **Time**: 1.5 hours

## Troubleshooting

### "vLLM not installed"
```bash
pip install vllm
```

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "No NVIDIA GPUs detected"
- Check `nvidia-smi` works
- Install CUDA drivers

### "Optimization takes too long"
- Use `--max-iterations 10` for quick mode
- Check database first with `--query`

## Next Steps

1. **Share your config** - Contribute to database for others
2. **Try different models** - Llama, Mistral, etc.
3. **Adjust target** - Use `--target-percentile 0.90` for faster results
4. **Deploy** - Use optimal config in production

## Advanced Usage

### Custom Output Directory
```bash
vllm-autopilot --model Qwen/Qwen3.5-9B --output-dir ./my_results
```

### Check Hardware
```bash
vllm-autopilot --hardware-info
```

### Use Existing Config
```python
from vllm_autopilot.database import ConfigDatabase

db = ConfigDatabase()
config = db.query(
    gpu_model="A100-80GB",
    model_name="Qwen/Qwen3.5-9B",
    context_length=16384,
    num_gpus=8,
)

if config:
    # Use config directly
    print(f"Found optimal config: {config['config']}")
else:
    # Run optimization
    from vllm_autopilot import optimize
    config = optimize(model="Qwen/Qwen3.5-9B")
```

## Tips

- ✅ **Query database first** - Save time if config exists
- ✅ **Use target_percentile=0.95** - Good enough, saves time
- ✅ **Share results** - Help the community
- ✅ **Validate in production** - Run quick test before deploying
- ⚠️ **Don't chase 100%** - Last 5% takes 50% of time

## Support

- **Issues**: https://github.com/yourusername/vllm-autopilot/issues
- **Docs**: See README.md and program.md
- **Examples**: Check `examples/` directory
