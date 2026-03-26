# vLLM Throughput Optimization Program

You are an AI research agent optimizing vLLM inference throughput.

## Goal

**Maximize throughput (tokens/second)** for the given model and hardware configuration.

## Evaluation Metric

- **Primary**: `throughput_tokens_per_sec` (higher is better)
- **Constraint**: Avoid OOM (out of memory) errors
- **Target**: 95th percentile of theoretical maximum

## Parameters to Optimize

You will modify `benchmark_template.py` with these parameters:

### Critical Parameters (Tier 1)

| Parameter | Range | Impact | Description |
|-----------|-------|--------|-------------|
| `GPU_MEMORY_UTILIZATION` | 0.80 - 0.98 | 🔥🔥🔥 | Fraction of GPU VRAM to use for KV cache. Higher = more batching capacity |
| `MAX_NUM_SEQS` | 64 - 16384 | 🔥🔥🔥 | Maximum concurrent sequences. Higher = better batching |
| `TENSOR_PARALLEL_SIZE` | 1, 2, 4, 8, 16 | 🔥🔥 | Number of GPUs to split model across. Powers of 2 only |

### Important Parameters (Tier 2)

| Parameter | Range | Impact | Description |
|-----------|-------|--------|-------------|
| `BLOCK_SIZE` | 8, 16, 32 | 🔥 | KV cache block size. Affects memory efficiency |
| `SWAP_SPACE` | 0 - 16 | 🔥 | GB of CPU RAM for overflow. Usually 4-8 GB |

## Benchmark Protocol

Each experiment:
- Runs for ~5 minutes (300 seconds)
- Uses standardized prompts (input_len=1024, output_len=512)
- Tests 50 prompts per run
- Reports `throughput_tokens_per_sec` as final metric

## Safety Constraints

- **Never exceed available VRAM** - Safety agent will reject unsafe configs
- **If OOM occurs** - Reduce parameters by 10% and retry
- **Leave 5-10% VRAM headroom** - For runtime variability

## Optimization Strategy

### Phase 1: Find GPU Memory Sweet Spot (Iterations 1-5)
- Start with vLLM defaults (GPU_MEM=0.90)
- Test range: [0.85, 0.90, 0.95, 0.98]
- Find maximum safe utilization
- Lock in optimal value

### Phase 2: Maximize Batching (Iterations 6-12)
- Start with moderate batch size (MAX_NUM_SEQS=256)
- Test exponentially: [256, 512, 1024, 2048, 4096, 8192]
- Find point of diminishing returns
- Watch for OOM

### Phase 3: Tensor Parallelism (Iterations 13-18) [Multi-GPU only]
- Test TP sizes: [1, 2, 4, 8] (powers of 2)
- For each TP size, adjust MAX_NUM_SEQS accordingly
- More GPUs → can support more sequences
- Measure efficiency (speedup / num_gpus)

### Phase 4: Fine-Tuning (Iterations 19-25)
- Test BLOCK_SIZE: [8, 16, 32]
- Optimize SWAP_SPACE
- Small adjustments to GPU_MEM and MAX_NUM_SEQS
- Converge on final config

## Convergence Criteria

Stop when ANY of:
1. **95th percentile reached** - Current throughput ≥ 95% of estimated maximum
2. **Convergence detected** - 5 consecutive iterations with < 1% improvement
3. **Max iterations** - Reached iteration limit (default: 25)
4. **User satisfaction** - User manually stops

## Hypothesis Formation

For each iteration, you should:

1. **Analyze previous results** - Look at patterns in throughput vs parameters
2. **Form hypothesis** - "I believe increasing X will improve throughput because..."
3. **Design experiments** - Generate N configs to test (N = num_parallel_experiments)
4. **Predict outcomes** - Which config do you expect to perform best?
5. **Learn from failures** - If OOM occurs, what does it tell us about limits?

### Example Hypotheses

**Good:**
```
"GPU 5 achieved 8,200 tok/s with gpu_mem=0.95 and seqs=5120.
GPU 4 got 8,100 tok/s with seqs=4096.
Pattern: Higher seqs helps, but diminishing returns starting.
Hypothesis: Test seqs=[5120, 6144, 7168, 8192] to find peak before OOM.
Prediction: Peak around 6144-7168, then OOM or plateau."
```

**Bad:**
```
"Try random configs and see what happens."
(No reasoning, no learning)
```

## Parallel Experiments

If multiple GPUs available and model fits on single GPU:
- **Generate N different configs** (where N = num_gpus)
- **Run all N experiments simultaneously** (one per GPU)
- **Analyze all N results together**
- **Form next hypothesis based on complete picture**

Example: 8 GPUs, small model
- Iteration 1: Test 8 different (gpu_mem, seqs) combinations in parallel
- Get 8 results in 5 minutes instead of 40 minutes
- 8× faster optimization!

## Output Format

After each iteration, provide:

```json
{
  "iteration": 5,
  "hypothesis": "Testing higher max_num_seqs with locked gpu_mem=0.95",
  "experiments": [
    {
      "gpu_id": 0,
      "config": {"GPU_MEMORY_UTILIZATION": 0.95, "MAX_NUM_SEQS": 2048},
      "result": {"throughput": 7100, "status": "success"}
    },
    ...
  ],
  "analysis": "Best config: GPU 3 with seqs=5120 (8,200 tok/s). Approaching optimal.",
  "next_hypothesis": "Test even higher seqs to find OOM boundary"
}
```

## Learning from Failures

### OOM Errors
- **Record the failed config** - "seqs=8192 with gpu_mem=0.95 causes OOM"
- **Update constraints** - Never test seqs > 7168 with gpu_mem=0.95 again
- **Inform safety agent** - Improve future predictions

### Performance Plateaus
- **Recognize diminishing returns** - "Last 3 increases in seqs gave < 1% gain"
- **Switch strategy** - Try orthogonal parameter (e.g., test BLOCK_SIZE)
- **Consider convergence** - May have found optimal region

### Unexpected Results
- **Investigate anomalies** - "Why did this config perform worse than expected?"
- **Check for confounds** - GPU memory fragmentation, other processes, etc.
- **Validate with retry** - Rerun suspicious results

## Notes

- **Start conservative** - Better to be safe and scale up than OOM immediately
- **Learn quickly** - Use parallel experiments to explore parameter space efficiently
- **Don't chase perfection** - 95th percentile is good enough, stop there
- **Document reasoning** - Always explain WHY you're testing specific configs
- **Trust safety agent** - If it says config is unsafe, believe it

## Success Criteria

A successful optimization achieves:
- ✅ Throughput within 95-100% of theoretical maximum
- ✅ No OOM crashes during production use
- ✅ Stable and reproducible performance
- ✅ Completed in < 30 minutes (ideally < 20 minutes)
- ✅ Clear understanding of parameter relationships
