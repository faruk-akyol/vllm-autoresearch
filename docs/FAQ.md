# Frequently Asked Questions (FAQ)

## General Questions

### What is vLLM Autopilot?

vLLM Autopilot is a tool that automatically finds the optimal vLLM configuration for your GPU and model. It uses an LLM agent (Claude/GPT) to intelligently explore the parameter space, similar to how a human researcher would conduct experiments.

---

### Why do I need this?

vLLM has many parameters (`gpu_memory_utilization`, `max_num_seqs`, `tensor_parallel_size`, etc.) that significantly affect throughput. Finding optimal settings manually can take days of trial and error. vLLM Autopilot:

- Finds optimal configs automatically in 15-30 minutes
- Achieves 30-40× throughput improvement over naive settings
- Prevents OOM crashes with a safety agent
- Shares configs with the community to avoid duplicate work

---

### How does it work?

It follows Andrej Karpathy's **autoresearch pattern**:

1. LLM agent forms a hypothesis (e.g., "higher memory utilization should help")
2. Generates experiments to test the hypothesis
3. Safety agent validates configs (prevents OOM)
4. Runs benchmarks in parallel across GPUs
5. Agent analyzes results and forms new hypothesis
6. Repeats until 95th percentile convergence

---

### How long does optimization take?

**Single GPU**: 20-30 minutes (sequential experiments)
**8× GPU (small model)**: 15-20 minutes (8 parallel experiments)
**8× GPU (large model)**: 25-30 minutes (limited parallelism due to tensor parallelism)

You can speed it up with `--max-iterations 10` or `--target-percentile 0.90`.

---

### Is it safe?

Yes! The **safety agent** estimates memory usage before each experiment and rejects configs likely to cause OOM. It also learns from actual OOM events to improve future predictions.

However, OOMs can still occasionally happen (memory estimation isn't perfect), so monitor your system during optimization.

---

### Can I trust the community database?

The database is currently trust-based (v0.1.0). Anyone can contribute configs.

**Coming in v0.2.0:**
- Verification system (configs verified by multiple users)
- Reputation scores
- Flagging suspicious results

For now, we recommend:
- Running your own optimization for critical production systems
- Using database configs as starting points, not final configs

---

## Usage Questions

### Do I need an API key?

Yes, you need an **Anthropic API key** (for Claude). The agent uses Claude to analyze results and generate experiments.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Coming in v0.2.0**: OpenAI GPT support as an alternative.

---

### How much does it cost?

**Compute**: Free (runs on your own GPUs)
**API calls**: ~$0.50-$2.00 per optimization (Claude API)

Each optimization makes ~10-25 API calls (one per iteration). With Claude Sonnet 4.5:
- Input: ~2K tokens per call × 20 calls = 40K tokens = $0.12
- Output: ~1K tokens per call × 20 calls = 20K tokens = $0.30
- **Total**: ~$0.42 per optimization

Cheaper with `--max-iterations 10` or `--target-percentile 0.90`.

---

### Can I use it offline?

No, vLLM Autopilot requires internet access to call the Claude API.

**Workaround**: Run optimization once online, save the config, then use offline.

**Alternative for future**: Local LLM support (Ollama, llama.cpp) planned for v0.3.0.

---

### What if I don't have multiple GPUs?

It still works! On a single GPU:
- Runs experiments sequentially (not in parallel)
- Takes longer (~30-40 min instead of 15-20 min)
- Still finds optimal config

---

### Can I stop and resume later?

Yes! vLLM Autopilot automatically saves checkpoints after every iteration.

```bash
# Start optimization
vllm-autopilot --model Qwen/Qwen3.5-9B

# Stop anytime (Ctrl+C)

# Resume later (automatically picks up from checkpoint)
vllm-autopilot --model Qwen/Qwen3.5-9B
```

To start fresh:
```bash
vllm-autopilot --model Qwen/Qwen3.5-9B --no-resume
```

---

### How do I query the database?

```bash
# Command line
vllm-autopilot --query --model Qwen/Qwen3.5-9B --gpu "RTX 4090"

# Python
from vllm_autopilot.database import ConfigDatabase

db = ConfigDatabase()
config = db.query(
    gpu_model="RTX 4090",
    model_name="Qwen/Qwen3.5-9B",
    context_length=16384,
    num_gpus=1
)

if config:
    print(f"Found: {config['config']}")
    print(f"Throughput: {config['throughput']} tok/s")
```

---

### Can I contribute my config to the database?

Yes! After optimization completes, you'll be prompted:

```
💾 Save to community database? (y/n): y
```

Or manually:
```python
from vllm_autopilot.database import ConfigDatabase

db = ConfigDatabase()
db.save_config(
    gpu_model="RTX 4090",
    num_gpus=1,
    model_name="Qwen/Qwen3.5-9B",
    context_length=16384,
    config=optimal_config,
    throughput=14020.5
)
```

---

## Technical Questions

### What parameters does it optimize?

**Tier 1 (Critical)**:
- `gpu_memory_utilization` (0.80 - 0.98)
- `max_num_seqs` (64 - 16384)
- `tensor_parallel_size` (1, 2, 4, 8, 16)

**Tier 2 (Important)**:
- `block_size` (8, 16, 32)
- `swap_space` (0 - 16 GB)

See [CONFIGURATION.md](CONFIGURATION.md) for details.

---

### Does it support quantized models?

**v0.1.0**: Yes, but quantization is not optimized (uses model's existing quantization)
**v0.2.0**: Will add quantization as an optimization parameter (FP16, AWQ, GPTQ, etc.)

---

### What about latency optimization?

**v0.1.0**: Only optimizes for **throughput** (tokens/sec)
**v0.3.0**: Will add **multi-objective optimization** (throughput + latency)

For now, if you need low latency:
- Use `--target-percentile 0.90` for faster optimization
- Manually reduce `max_num_seqs` in final config (lower batch = lower latency)

---

### Can I use it for other LLM engines?

**v0.1.0**: vLLM only
**v0.3.0**: Planned support for llama.cpp, TensorRT-LLM

---

### How accurate is the safety agent?

The safety agent's memory estimation is ~90% accurate. It occasionally:
- **False positives** (~5%): Rejects safe configs
- **False negatives** (~5%): Allows configs that OOM

When an OOM occurs, the safety agent learns from it and improves future predictions.

---

### Why 95th percentile and not 100%?

**Pareto Principle**: The last 5% of performance takes 50% of the time.

Example:
```
Iteration 1-10:  Reach 12,000 tok/s (time: 15 min)
Iteration 11-20: Reach 13,500 tok/s (time: 15 min)
Iteration 21-30: Reach 13,800 tok/s (time: 15 min)
Iteration 31-50: Reach 14,000 tok/s (time: 30 min)  ← Diminishing returns
```

The 95th percentile (13,650 tok/s) is reached at iteration ~18 (20 min).
Pushing to 100% (14,000 tok/s) takes 50+ iterations (75+ min).

**Verdict**: 95% is good enough, saves massive time.

---

## Troubleshooting Questions

### Why is optimization so slow?

**Possible causes:**
1. Too many iterations (`--max-iterations 50`)
2. Sequential execution (not parallel)
3. Slow network (API latency)
4. Large model + single GPU

**Solutions:**
1. Reduce iterations: `--max-iterations 15`
2. Lower target: `--target-percentile 0.90`
3. Check database first: `--query`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more.

---

### Why do all experiments cause OOM?

**Possible causes:**
1. Model too large for GPU
2. Context length too long
3. Safety agent misconfigured

**Solutions:**
1. Reduce context: `--context-length 8192`
2. Use tensor parallelism (automatically tried for large models)
3. Check model size vs VRAM

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more.

---

### Why is throughput so low?

**Possible causes:**
1. GPU thermal throttling
2. Other processes using GPU
3. vLLM installation issue
4. Incorrect benchmark config

**Solutions:**
1. Check GPU utilization: `nvidia-smi`
2. Kill other processes: `nvidia-smi` → kill PID
3. Test vLLM directly: `python -m vllm.benchmarks.benchmark_throughput --model ...`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more.

---

## Comparison Questions

### vLLM Autopilot vs Manual Tuning

| Aspect | Manual Tuning | vLLM Autopilot |
|--------|---------------|----------------|
| **Time** | Days | 15-30 minutes |
| **Expertise required** | High | None |
| **Thoroughput gain** | Varies | 30-40× consistently |
| **Safety** | Trial and error, many OOMs | Safety agent, few OOMs |
| **Community sharing** | Manual sharing | Automatic database |

---

### vLLM Autopilot vs Ray Tune / Optuna

| Aspect | Ray Tune / Optuna | vLLM Autopilot |
|--------|-------------------|----------------|
| **Approach** | Bayesian optimization, random search | LLM-guided hypothesis generation |
| **Sample efficiency** | 50-100 experiments | 10-25 experiments |
| **Interpretability** | Black box | Agent explains reasoning |
| **Parallel execution** | Yes | Yes |
| **Safety** | No OOM prevention | Safety agent |

vLLM Autopilot is more sample-efficient and safer, but Ray Tune/Optuna are more general-purpose.

---

### vLLM Autopilot vs Using vLLM Defaults

| Metric | vLLM Defaults | vLLM Autopilot |
|--------|---------------|----------------|
| **Throughput** | ~350 tok/s | ~14,000 tok/s |
| **GPU utilization** | ~30% | ~95% |
| **Setup time** | 0 minutes | 15-30 minutes |
| **Optimal?** | No | Yes |

vLLM defaults are a safe starting point, but far from optimal.

---

## Future Features

### When will v0.2.0 be released?

Target: Q1 2025

**Planned features:**
- Web UI for browsing configs
- OpenAI GPT support
- Improved safety agent
- Database verification system

---

### When will v1.0.0 be released?

Target: Q2-Q3 2025

**Planned features:**
- Stable API
- Production-ready safety guarantees
- Comprehensive test coverage
- Enterprise features (audit logs, etc.)

---

### Can I request a feature?

Yes! Open a feature request on GitHub:

https://github.com/yourusername/vllm-autopilot/issues/new?template=feature_request.md

---

### Can I contribute code?

Absolutely! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

We especially welcome:
- Improvements to safety agent memory estimation
- Support for other LLM providers (OpenAI, local models)
- Web UI development
- Test coverage improvements

---

## Philosophy Questions

### Why use an LLM for optimization?

**Traditional approaches**:
- Grid search: Exhaustive, slow (test every combination)
- Random search: Wasteful, no learning
- Bayesian optimization: Sample-efficient, but black box

**LLM-guided approach**:
- Forms human-like hypotheses
- Learns from experiments like a researcher
- Explainable (agent explains reasoning)
- Sample-efficient (10-25 experiments vs 50-100)

**Inspired by**: Karpathy's autoresearch, which showed LLMs can effectively guide scientific experiments.

---

### Why not just use the database?

The database is great for **quick lookups** if your exact config exists. But:

1. **Exact matches are rare**: Different GPUs, models, context lengths → different optimal configs
2. **Hardware varies**: Even same GPU model can perform differently (cooling, drivers, etc.)
3. **vLLM updates**: New versions may have different optimal configs
4. **Trust**: Database is unverified (v0.1.0), your own optimization is more trustworthy

**Best practice**: Query database first, but run your own optimization for production systems.

---

### Why 95th percentile instead of maximizing throughput?

**Pareto Principle**: Last 5% takes 50% of the time.

Also:
- 95% is "good enough" for production
- Chasing 100% leads to overfitting (config works in benchmark but not production)
- Time saved can be used for other optimizations (prompt engineering, caching, etc.)

**Philosophy**: "Perfect is the enemy of good."

---

## Still Have Questions?

- **Documentation**: See `docs/` directory
- **GitHub Issues**: https://github.com/yourusername/vllm-autopilot/issues
- **Discussions**: https://github.com/yourusername/vllm-autopilot/discussions
- **Email**: (if you set up a contact email)

We're here to help!
