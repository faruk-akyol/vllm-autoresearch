# Configuration Guide

## vLLM Parameters Explained

This guide explains all vLLM parameters that affect throughput and how vLLM Autopilot optimizes them.

---

## Critical Parameters (Tier 1)

These have the highest impact on throughput and are always optimized.

### `gpu_memory_utilization`

**Range**: 0.80 - 0.98
**Default**: 0.90
**Impact**: 🔥🔥🔥 Extreme

**What it does:**
- Controls what fraction of GPU VRAM to use for KV cache
- Higher = more memory for batching = more concurrent requests
- Lower = safer but less throughput

**Example:**
```python
# Conservative (safe)
gpu_memory_utilization = 0.85  # Use 85% of VRAM

# Optimal (typical best)
gpu_memory_utilization = 0.94  # Use 94% of VRAM

# Aggressive (risky)
gpu_memory_utilization = 0.98  # Use 98% of VRAM
```

**Trade-offs:**
- **Too low** (0.80-0.85): Wastes GPU memory, lower throughput
- **Optimal** (0.90-0.95): Balanced, good throughput
- **Too high** (0.96-0.98): Risk of OOM, unstable

**How Autopilot optimizes:**
1. Starts at 0.90 (default)
2. Tests higher values (0.92, 0.94, 0.96)
3. Finds maximum safe value before OOM
4. Typically converges to 0.93-0.95

**Real-world impact:**
```
gpu_mem=0.80: 9,200 tokens/sec
gpu_mem=0.90: 11,500 tokens/sec (+25%)
gpu_mem=0.94: 13,800 tokens/sec (+20%)
gpu_mem=0.98: OOM
```

---

### `max_num_seqs`

**Range**: 64 - 16384
**Default**: 256
**Impact**: 🔥🔥🔥 Extreme

**What it does:**
- Maximum number of concurrent sequences (requests) to batch
- Higher = better GPU utilization = more throughput
- Constrained by available KV cache memory

**Example:**
```python
# Low batching
max_num_seqs = 128  # 128 concurrent requests

# Medium batching
max_num_seqs = 512  # 512 concurrent requests

# High batching (if memory allows)
max_num_seqs = 2048  # 2048 concurrent requests
```

**Trade-offs:**
- **Too low** (64-128): GPU underutilized, low throughput
- **Optimal** (256-1024): Depends on model size and VRAM
- **Too high** (2048+): May cause OOM

**Relationship to memory:**
```python
# Approximate KV cache memory:
kv_cache_gb = (max_num_seqs * context_length * 150 bytes) / 1e9

# Example: Qwen3.5-9B, 16k context, 24GB GPU
max_num_seqs = 256:  ~6.1 GB  ✓ Safe
max_num_seqs = 512:  ~12.3 GB ✓ Safe
max_num_seqs = 1024: ~24.6 GB ✗ OOM
```

**How Autopilot optimizes:**
1. Starts conservatively (256)
2. Tests exponentially: 256 → 384 → 512 → 768 → 1024
3. Finds maximum before OOM
4. Typical sweet spot: 384-768 for 9B models on 24GB GPU

**Real-world impact:**
```
max_num_seqs=128:  8,400 tokens/sec
max_num_seqs=256:  11,200 tokens/sec (+33%)
max_num_seqs=512:  13,900 tokens/sec (+24%)
max_num_seqs=1024: OOM
```

---

### `tensor_parallel_size`

**Range**: 1, 2, 4, 8, 16 (powers of 2)
**Default**: 1
**Impact**: 🔥🔥 Very High

**What it does:**
- Number of GPUs to split model across
- Required for models larger than single GPU VRAM
- Adds communication overhead, reduces efficiency

**Example:**
```python
# Single GPU (no parallelism)
tensor_parallel_size = 1  # Entire model on one GPU

# 2-way parallelism
tensor_parallel_size = 2  # Model split across 2 GPUs

# 8-way parallelism
tensor_parallel_size = 8  # Model split across 8 GPUs
```

**Trade-offs:**
- **TP=1**: Best efficiency, but model must fit in single GPU
- **TP=2,4**: Moderate overhead (~10-20% slower per GPU)
- **TP=8+**: High overhead (~30-40% slower per GPU)

**When to use:**
```python
# Small models (7B-13B on 24GB GPUs)
tensor_parallel_size = 1  # Model fits on single GPU

# Medium models (30B-40B on 24GB GPUs)
tensor_parallel_size = 2  # ~30GB model needs 2 GPUs

# Large models (70B on 24GB GPUs)
tensor_parallel_size = 4  # ~70GB model needs 4+ GPUs

# Huge models (405B on 80GB GPUs)
tensor_parallel_size = 8  # ~400GB model needs 8 GPUs
```

**How Autopilot optimizes:**
1. Checks if model fits on single GPU
2. If yes: uses TP=1 (best efficiency)
3. If no: tries minimum TP needed
4. Tests higher TP if available (may improve throughput via parallelism)

**Real-world impact:**
```
Qwen3.5-9B on 8× RTX 4090:
  TP=1: 14,000 tok/s (using 1 GPU)
  TP=2: 12,500 tok/s total (6,250 per GPU, overhead)
  TP=8: 9,600 tok/s total (1,200 per GPU, high overhead)

Qwen3.5-70B on 8× RTX 4090:
  TP=1: OOM (model too large)
  TP=2: OOM (still too large)
  TP=4: 11,200 tok/s (model fits, optimal)
  TP=8: 10,500 tok/s (unnecessary overhead)
```

---

## Important Parameters (Tier 2)

These have moderate impact and are optimized in later phases.

### `block_size`

**Range**: 8, 16, 32
**Default**: 16
**Impact**: 🔥 Moderate

**What it does:**
- KV cache block size (number of tokens per block)
- Affects memory fragmentation and allocation efficiency
- Smaller = less fragmentation, larger = less overhead

**Example:**
```python
block_size = 8   # Fine-grained allocation
block_size = 16  # Balanced (default)
block_size = 32  # Coarse-grained allocation
```

**Trade-offs:**
- **block_size=8**: Less memory waste, more allocation overhead
- **block_size=16**: Balanced (usually optimal)
- **block_size=32**: More memory waste, less allocation overhead

**How Autopilot optimizes:**
1. Starts with default (16)
2. Tests 8 and 32 in fine-tuning phase
3. Usually confirms 16 is optimal
4. Rarely changes from default

**Real-world impact:**
```
block_size=8:  13,200 tok/s
block_size=16: 13,800 tok/s (+4%)
block_size=32: 13,500 tok/s (-2%)
```

---

### `swap_space`

**Range**: 0 - 16 GB
**Default**: 4 GB
**Impact**: 🔥 Low-Moderate

**What it does:**
- Amount of CPU RAM to use for KV cache overflow
- When GPU memory full, swaps to CPU RAM
- Helps avoid OOM but adds latency

**Example:**
```python
swap_space = 0   # No swapping (fail if OOM)
swap_space = 4   # 4GB CPU RAM for overflow
swap_space = 16  # 16GB CPU RAM for overflow
```

**Trade-offs:**
- **swap_space=0**: Faster, but may OOM under load spikes
- **swap_space=4-8**: Good safety buffer, minimal overhead
- **swap_space=16+**: Rarely useful, wastes RAM

**How Autopilot optimizes:**
1. Usually keeps default (4GB)
2. May test 0 if seeking maximum throughput
3. May increase to 8GB if OOMs occur

**Real-world impact:**
```
swap_space=0:  14,100 tok/s (but risky)
swap_space=4:  14,000 tok/s (-0.7%)
swap_space=8:  13,950 tok/s (-1%)
```

---

## Parameters NOT Optimized

These are fixed or not relevant to throughput optimization.

### `model`
- Fixed by user
- Not optimized

### `tokenizer`
- Usually same as model
- Not optimized

### `max_model_len`
- Same as `context_length` (user input)
- Not optimized

### `dtype`
- Model's native dtype (e.g., float16, bfloat16)
- Not optimized (using wrong dtype breaks model)

### `quantization`
- If quantized (e.g., AWQ, GPTQ), fixed at load time
- Not optimized in current version
- Future: may add quantization as parameter

### `seed`
- For reproducibility, not performance
- Not optimized

---

## Parameter Interactions

### Memory Budget

Total GPU memory usage:
```python
total_memory = (
    model_size / tensor_parallel_size +           # Model weights
    kv_cache_size +                               # KV cache
    activation_memory +                           # Activations
    overhead                                      # vLLM overhead
)

kv_cache_size = (
    max_num_seqs *
    context_length *
    num_layers * 2 *  # Keys + Values
    hidden_size /
    1e9  # Convert to GB
)
```

**Constraints:**
```python
total_memory <= gpu_vram_gb * gpu_memory_utilization
```

**This means:**
- Higher `gpu_memory_utilization` → more room for `max_num_seqs`
- Higher `max_num_seqs` → need higher `gpu_memory_utilization`
- Larger `tensor_parallel_size` → less memory per GPU, but allows larger models

---

### Batching Efficiency

Throughput formula (simplified):
```python
throughput = (
    batch_size *                    # Concurrent requests
    gpu_compute_utilization *       # How busy is GPU
    tokens_per_second_per_request
)
```

**Key insights:**
- Larger batches (high `max_num_seqs`) = better GPU utilization
- But: larger batches need more memory
- Optimal: largest batch that fits in memory

---

### Tensor Parallelism Overhead

Efficiency formula:
```python
efficiency = 1 - (communication_overhead / compute_time)

communication_overhead ∝ tensor_parallel_size
```

**Trade-off:**
- TP=1: 100% efficiency (no communication)
- TP=2: ~90% efficiency (10% overhead)
- TP=4: ~80% efficiency (20% overhead)
- TP=8: ~70% efficiency (30% overhead)

**When TP is worth it:**
```python
# Only if model doesn't fit on single GPU
if model_size > gpu_vram:
    min_tp = ceil(model_size / gpu_vram)
    # Use min_tp, not higher
```

---

## Optimization Phases

### Phase 1: Memory Utilization (Iterations 1-5)

**Goal**: Find maximum safe `gpu_memory_utilization`

**Strategy:**
```python
test_values = [0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
optimal_gpu_mem = max(value for value in test_values if not OOM)
```

**Typical outcome**: 0.93-0.95

---

### Phase 2: Batch Size (Iterations 6-12)

**Goal**: Find maximum `max_num_seqs` that fits

**Strategy:**
```python
# Lock in optimal gpu_mem from Phase 1
gpu_memory_utilization = 0.94  # From Phase 1

# Test batch sizes
test_values = [256, 384, 512, 768, 1024, 1536, 2048]
optimal_batch = max(value for value in test_values if not OOM)
```

**Typical outcome**: 384-768 for 9B models

---

### Phase 3: Tensor Parallelism (Iterations 13-18)

**Goal**: Determine if TP helps (multi-GPU only)

**Strategy:**
```python
if model_fits_on_single_gpu:
    tensor_parallel_size = 1  # Best efficiency
else:
    # Test minimum required and higher
    test_values = [2, 4, 8]
    # Pick TP with highest total throughput
```

**Typical outcome**: TP=1 for small models, TP=4-8 for large models

---

### Phase 4: Fine-Tuning (Iterations 19-25)

**Goal**: Optimize `block_size` and minor tweaks

**Strategy:**
```python
# Test block sizes
test_block_sizes = [8, 16, 32]

# Small adjustments to gpu_mem and max_seqs
gpu_mem += [0.00, +0.01, -0.01]
max_seqs += [0, +32, -32]
```

**Typical outcome**: Confirms defaults (block_size=16)

---

## Configuration Examples

### Small Model (7B-13B) on Single GPU

**Hardware**: 1× RTX 4090 (24GB)
**Model**: Qwen/Qwen3.5-9B
**Context**: 16384

**Optimal Config:**
```python
{
  "gpu_memory_utilization": 0.94,
  "max_num_seqs": 420,
  "tensor_parallel_size": 1,
  "block_size": 16,
  "swap_space": 4
}

# Throughput: ~14,000 tokens/sec
```

---

### Small Model (7B-13B) on Multi-GPU

**Hardware**: 8× RTX 4090 (24GB each)
**Model**: meta-llama/Llama-3.1-8B
**Context**: 8192

**Optimal Config (per GPU):**
```python
{
  "gpu_memory_utilization": 0.95,
  "max_num_seqs": 512,
  "tensor_parallel_size": 1,  # Each GPU runs independently
  "block_size": 16,
  "swap_space": 4
}

# Throughput: ~16,000 tokens/sec per GPU
# Total: ~128,000 tokens/sec across 8 GPUs
```

---

### Large Model (70B) on Multi-GPU

**Hardware**: 8× RTX 4090 (24GB each)
**Model**: Qwen/Qwen3.5-70B
**Context**: 16384

**Optimal Config:**
```python
{
  "gpu_memory_utilization": 0.95,
  "max_num_seqs": 256,
  "tensor_parallel_size": 4,  # Model split across 4 GPUs
  "block_size": 16,
  "swap_space": 4
}

# Throughput: ~11,000 tokens/sec total
# Can run 2 independent instances (4 GPUs each) = 22,000 tok/s total
```

---

### Huge Model (405B) on Multi-GPU

**Hardware**: 8× A100 (80GB each)
**Model**: meta-llama/Llama-3.1-405B
**Context**: 8192

**Optimal Config:**
```python
{
  "gpu_memory_utilization": 0.96,
  "max_num_seqs": 128,
  "tensor_parallel_size": 8,  # Model split across all 8 GPUs
  "block_size": 16,
  "swap_space": 8
}

# Throughput: ~3,500 tokens/sec total
```

---

## How to Use Optimal Configs

### Option 1: vLLM CLI
```bash
vllm serve Qwen/Qwen3.5-9B \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 420 \
  --tensor-parallel-size 1 \
  --block-size 16 \
  --swap-space 4 \
  --port 8000
```

### Option 2: Python API
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.5-9B",
    gpu_memory_utilization=0.94,
    max_num_seqs=420,
    tensor_parallel_size=1,
    block_size=16,
    swap_space=4
)

prompts = ["Hello, how are you?"] * 100
outputs = llm.generate(prompts)
```

### Option 3: OpenAI-Compatible Server
```bash
vllm serve Qwen/Qwen3.5-9B \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 420 \
  --api-key your-api-key \
  --port 8000
```

Then use with OpenAI client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Advanced Topics

### Dynamic Batching

vLLM automatically batches requests. The `max_num_seqs` parameter sets the upper limit:

```python
# max_num_seqs = 420

# Low load (10 requests):
# Batches 10 requests together

# Medium load (200 requests):
# Batches 200 requests together

# High load (1000 requests):
# Batches up to 420 at a time (remaining 580 queued)
```

### Context Length Impact

Longer context = more KV cache memory needed:

```python
# Same model, same GPU, different context lengths:

context=4096,  max_num_seqs=1600  # ✓ Fits
context=8192,  max_num_seqs=800   # ✓ Fits
context=16384, max_num_seqs=420   # ✓ Fits
context=32768, max_num_seqs=210   # ✓ Fits
context=65536, max_num_seqs=105   # ✓ Fits
```

Rule of thumb: `max_num_seqs ∝ 1 / context_length`

### Quantization Impact

Quantized models use less memory → can fit larger batches:

```python
# FP16 (original):
model_size = 14 GB
max_num_seqs = 420

# AWQ 4-bit quantized:
model_size = 4 GB  # ~3.5× smaller
max_num_seqs = 1200  # ~3× larger batch
# Throughput: ~1.8× higher
```

Autopilot will support quantization in v0.2.0.
