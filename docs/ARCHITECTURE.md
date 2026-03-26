# Architecture

## System Overview

vLLM Autopilot follows the **autoresearch pattern** popularized by Andrej Karpathy. The system uses an LLM agent to intelligently explore the parameter space and find optimal configurations.

```
┌──────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                           │
│  (Main optimization loop, convergence tracking)          │
└────────┬───────────────────────────────────────┬─────────┘
         │                                       │
         ▼                                       ▼
┌────────────────────┐                 ┌────────────────────┐
│  HYPOTHESIS AGENT  │◄────────────────┤   SAFETY AGENT     │
│  (Claude/GPT LLM)  │                 │  (OOM Prevention)  │
│                    │                 │                    │
│ - Analyzes results │                 │ - Memory estimates │
│ - Forms hypothesis │                 │ - Validates configs│
│ - Designs exps     │                 │ - Learns from OOMs │
└────────┬───────────┘                 └────────┬───────────┘
         │                                       │
         │         ┌─────────────────────────────┘
         │         │
         ▼         ▼
┌──────────────────────────────────────────────────────────┐
│              PARALLEL SCHEDULER                          │
│  Distributes experiments across GPUs                     │
│  Uses ProcessPoolExecutor + CUDA_VISIBLE_DEVICES         │
└────────┬───────────────────────────────────────┬─────────┘
         │                                       │
         ▼                                       ▼
┌──────────────────┐                   ┌──────────────────┐
│  GPU 0           │ ...               │  GPU 7           │
│  Benchmark       │                   │  Benchmark       │
└──────────────────┘                   └──────────────────┘
         │                                       │
         └───────────────┬───────────────────────┘
                         ▼
                ┌────────────────────┐
                │   RESULTS DB       │
                │  (History + Best)  │
                └────────────────────┘
```

## Core Components

### 1. Orchestrator (`orchestrator.py`)

**Responsibility**: Coordinates the entire optimization process

**Key Functions:**
- Manages iteration loop
- Tracks convergence (95th percentile target)
- Handles checkpointing and resume
- Coordinates agents
- Logs progress

**Pseudocode:**
```python
while not converged and iterations < max_iterations:
    # 1. Generate experiments
    experiments = hypothesis_agent.generate(history)

    # 2. Validate safety
    safe_experiments = [e for e in experiments if safety_agent.check(e)]

    # 3. Run in parallel
    results = parallel_scheduler.run(safe_experiments)

    # 4. Update history
    history.append(results)

    # 5. Check convergence
    if reached_95p_percentile(history):
        converged = True

    # 6. Save checkpoint
    save_checkpoint(history, best_config)

return best_config
```

**State Management:**
- Checkpoints saved after every iteration
- History includes all experiments + results
- Best config tracked separately for fast access

---

### 2. Hypothesis Agent (`agent.py`)

**Responsibility**: LLM-powered experiment generation

**Key Functions:**
- Analyzes benchmark history
- Forms scientific hypotheses
- Designs N experiments (N = num_parallel_gpus)
- Predicts which configs will perform best

**Architecture:**
```
Input:
  - Hardware info (GPUs, VRAM)
  - Model info (name, size, context length)
  - Experiment history (configs, throughputs, OOMs)
  - Research instructions (program.md)

Processing:
  1. Analyze patterns in history
  2. Identify promising parameter ranges
  3. Form hypothesis about next experiments
  4. Generate N diverse configs to test

Output:
  - List of N experiment configs
  - Hypothesis explanation
  - Predicted performance ranking
```

**Prompt Engineering:**
The agent receives a structured prompt containing:
1. **Goal**: "Maximize throughput (tokens/sec)"
2. **Current state**: Best config so far, iteration number
3. **History**: Previous experiments and results
4. **Constraints**: Hardware limits, OOM boundaries
5. **Instructions**: Research strategy from `program.md`

**Example Reasoning:**
```
Iteration 5 Analysis:
- Experiments 1-4 with gpu_mem=0.90 averaged 11,200 tok/s
- Experiments 5-8 with gpu_mem=0.95 averaged 11,800 tok/s
- Pattern: Higher gpu_mem → better throughput
- But: gpu_mem=0.98 caused OOM in iteration 3

Hypothesis:
  Optimal gpu_mem is between 0.92-0.96

Next Experiments:
  Test: [0.92, 0.93, 0.94, 0.95, 0.96] with fixed seqs=256
  Prediction: Peak around 0.94-0.95
```

---

### 3. Safety Agent (`safety.py`)

**Responsibility**: Prevent OOM crashes

**Key Functions:**
- Estimate memory requirements before running
- Validate configs (reject if likely OOM)
- Suggest safe alternatives
- Learn from actual OOM events

**Memory Estimation Formula:**
```python
def estimate_memory(config, model_size_gb, context_length):
    # 1. Model weights
    model_memory = model_size_gb / config.tensor_parallel_size

    # 2. KV cache
    # Formula: (num_layers * 2 * hidden_size * num_heads * head_dim
    #           * max_num_seqs * context_length) / 1e9
    # Approximation: ~150 bytes per token per sequence
    kv_cache_gb = (config.max_num_seqs * context_length * 150) / 1e9

    # 3. Activations (estimated)
    activation_gb = 2.0  # Conservative estimate

    # 4. Overhead
    overhead_gb = 1.0

    total_gb = model_memory + kv_cache_gb + activation_gb + overhead_gb

    # Apply memory utilization
    available_gb = gpu_vram_gb * config.gpu_memory_utilization

    return {
        "total_estimated": total_gb,
        "available": available_gb,
        "safe": total_gb < available_gb - safety_margin
    }
```

**Learning from OOMs:**
```python
class SafetyAgent:
    def __init__(self):
        self.oom_history = []  # Track actual OOM events
        self.correction_factor = 1.0

    def record_oom(self, config, actual_memory_gb):
        estimated = self.estimate_memory(config)
        error = actual_memory_gb / estimated

        # Update correction factor (exponential moving average)
        self.correction_factor = 0.9 * self.correction_factor + 0.1 * error
        self.oom_history.append({
            "config": config,
            "estimated": estimated,
            "actual": actual_memory_gb
        })
```

---

### 4. Parallel Scheduler (`orchestrator.py`)

**Responsibility**: Execute experiments in parallel across GPUs

**Implementation:**
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def run_parallel_experiments(configs, num_gpus):
    num_workers = min(len(configs), num_gpus)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for i, config in enumerate(configs):
            gpu_id = i % num_gpus
            future = executor.submit(
                run_single_experiment_worker,
                gpu_id, config
            )
            futures[future] = (gpu_id, config)

        results = []
        for future in as_completed(futures):
            gpu_id, config = futures[future]
            result = future.result()
            results.append(result)

        return results

def run_single_experiment_worker(gpu_id, config):
    # Assign this process to specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Modify benchmark_template.py with config
    write_benchmark_config(config)

    # Run benchmark
    result = subprocess.run(
        ["python", "benchmark_template.py"],
        capture_output=True
    )

    return parse_result(result)
```

**Key Design Decisions:**

1. **ProcessPoolExecutor vs ThreadPoolExecutor**
   - Need separate processes to isolate GPU assignments
   - `CUDA_VISIBLE_DEVICES` is process-level, not thread-level

2. **Static Worker Method**
   - Worker must be picklable for multiprocessing
   - Use `@staticmethod` or top-level function

3. **GPU Assignment**
   - Round-robin assignment: `gpu_id = i % num_gpus`
   - Each process sees only its assigned GPU

---

### 5. Hardware Detection (`hardware.py`)

**Responsibility**: Detect GPUs and system capabilities

**Fixed Code** (never modified by agent):

```python
def detect_gpus():
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.total,compute_cap",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )

    gpus = []
    for line in result.stdout.strip().split('\n'):
        index, name, vram_mb, compute_cap = line.split(', ')
        gpus.append(GPUInfo(
            index=int(index),
            name=name.strip(),
            vram_gb=int(vram_mb) // 1024,
            compute_capability=compute_cap
        ))

    return gpus
```

---

### 6. Benchmark Template (`benchmark_template.py`)

**Responsibility**: Run vLLM benchmarks

**Modified by agent** each iteration:

```python
# AGENT MODIFIES THESE PARAMETERS
GPU_MEMORY_UTILIZATION = 0.90  # Changed by agent
MAX_NUM_SEQS = 256             # Changed by agent
TENSOR_PARALLEL_SIZE = 1       # Changed by agent
BLOCK_SIZE = 16                # Changed by agent
SWAP_SPACE = 4                 # Changed by agent

# FIXED CODE (never modified)
def run_benchmark():
    cmd = [
        "python", "-m", "vllm.benchmarks.benchmark_throughput",
        "--model", MODEL_NAME,
        "--input-len", "1024",
        "--output-len", "512",
        "--num-prompts", "50",
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--block-size", str(BLOCK_SIZE),
        "--swap-space", str(SWAP_SPACE)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_throughput(result.stdout)
```

**Pattern**: Similar to Karpathy's `train.py` - agent modifies config, fixed code runs benchmark

---

### 7. Database (`database.py`)

**Responsibility**: Store and query optimal configs

**Schema:**
```json
{
  "configs": [
    {
      "id": "uuid",
      "gpu_model": "RTX 4090",
      "num_gpus": 1,
      "model": "Qwen/Qwen3.5-9B",
      "context_length": 16384,
      "config": {
        "gpu_memory_utilization": 0.94,
        "max_num_seqs": 420,
        "tensor_parallel_size": 1,
        "block_size": 16
      },
      "throughput": 14020,
      "verified_count": 1,
      "timestamp": "2024-12-15T10:30:00Z",
      "user": "anonymous",
      "vllm_version": "0.6.0"
    }
  ]
}
```

**Query Logic:**
1. Try exact match first (gpu, model, context, num_gpus)
2. If not found, try similar configs (same GPU, similar model size)
3. Return config with highest verification count

**Verification System:**
- Users can verify existing configs
- `verified_count` increments with each verification
- Higher verification = more trustworthy

---

## Data Flow

### Iteration Lifecycle

```
1. START ITERATION
   ├─ Load checkpoint (if resume)
   └─ Initialize agents

2. GENERATE EXPERIMENTS
   ├─ Hypothesis agent reads history
   ├─ Forms hypothesis
   ├─ Generates N configs
   └─ Returns experiments

3. VALIDATE SAFETY
   ├─ For each config:
   │  ├─ Safety agent estimates memory
   │  ├─ If safe: keep
   │  └─ If unsafe: suggest alternative
   └─ Returns safe configs

4. RUN BENCHMARKS
   ├─ Parallel scheduler creates workers
   ├─ Each worker:
   │  ├─ Assigns to GPU
   │  ├─ Modifies benchmark_template.py
   │  ├─ Runs vLLM benchmark
   │  └─ Returns throughput
   └─ Collects all results

5. ANALYZE RESULTS
   ├─ Update history
   ├─ Update best config if improved
   ├─ Check for OOMs
   │  └─ If OOM: notify safety agent
   └─ Calculate convergence metrics

6. CHECK CONVERGENCE
   ├─ Calculate 95th percentile target
   ├─ Check if current best >= 95p
   ├─ Check diminishing returns
   └─ Decide: continue or stop?

7. CHECKPOINT
   ├─ Save history to disk
   ├─ Save best config
   └─ Save agent state

8. END ITERATION
   └─ If converged: exit
   └─ Else: go to step 2
```

---

## Convergence Algorithm

```python
def check_convergence(history, target_percentile=0.95):
    # Get all successful experiments (no OOM)
    successful = [e for e in history if e.status == "success"]
    throughputs = [e.throughput for e in successful]

    if len(throughputs) < 10:
        return False  # Need more data

    # Calculate 95th percentile target
    sorted_throughputs = sorted(throughputs, reverse=True)
    percentile_95 = sorted_throughputs[int(len(sorted_throughputs) * 0.05)]

    current_best = max(throughputs)

    # Check if within 95th percentile
    if current_best >= percentile_95:
        return True

    # Also check diminishing returns
    last_5 = throughputs[-5:]
    if len(last_5) == 5:
        improvement = (max(last_5) - min(last_5)) / min(last_5)
        if improvement < 0.01:  # <1% improvement
            return True

    return False
```

---

## Design Patterns

### 1. Agent Pattern
- Each component is an independent agent
- Agents communicate through well-defined interfaces
- Orchestrator coordinates but doesn't micromanage

### 2. Checkpoint Pattern
- State saved after every iteration
- Can resume from any point
- Never lose progress

### 3. Parallel Execution Pattern
- ProcessPoolExecutor for true parallelism
- GPU assignment via environment variables
- Results collected asynchronously

### 4. Safety First Pattern
- Validate before running
- Learn from failures
- Conservative estimates with safety margin

### 5. 95th Percentile Pattern
- Don't chase perfection
- Good enough beats perfect
- Diminishing returns detection

---

## Scalability Considerations

### Multi-GPU Scaling
- Linear speedup: 8 GPUs = 8× faster optimization
- Only if model fits on single GPU
- Tensor parallelism for larger models

### Large Model Support
- Tensor parallelism splits model across GPUs
- Reduces parallelism (8 GPUs → 4 experiments if TP=2)
- Safety agent accounts for distributed memory

### Memory Efficiency
- Checkpoint compression
- History pruning (keep best N)
- Database indexing for fast queries

---

## Error Handling

### OOM Recovery
```python
try:
    result = run_benchmark(config)
except OOMError:
    safety_agent.record_oom(config)
    # Retry with safer config
    safe_config = safety_agent.suggest_safe_config(config)
    result = run_benchmark(safe_config)
```

### Checkpoint Corruption
```python
try:
    state = load_checkpoint()
except CorruptedCheckpoint:
    # Rollback to previous checkpoint
    state = load_checkpoint(iteration - 1)
```

### Network Failures
```python
try:
    response = claude_api.call()
except NetworkError:
    # Retry with exponential backoff
    for attempt in range(3):
        time.sleep(2 ** attempt)
        try:
            response = claude_api.call()
            break
        except NetworkError:
            continue
```

---

## Future Architecture Improvements

### v0.2.0
- Ray integration for distributed execution
- Multi-objective optimization (throughput + latency)
- Bayesian optimization alongside LLM

### v0.3.0
- Streaming results (real-time updates)
- Web UI with live progress
- Multi-cloud support (AWS, GCP, Azure)

### v1.0.0
- Production monitoring and alerts
- A/B testing of configs
- Auto-tuning based on production metrics
