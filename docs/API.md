# API Documentation

## Python API Reference

### High-Level API

#### `optimize()`

Main optimization function for finding optimal vLLM configurations.

```python
from vllm_autopilot import optimize

config = optimize(
    model: str,
    context_length: int = 16384,
    max_iterations: int = 25,
    target_percentile: float = 0.95,
    output_dir: str = "./vllm_optimization",
    log_file: Optional[str] = None,
    api_key: Optional[str] = None,
    resume_from_checkpoint: bool = True
) -> dict
```

**Parameters:**

- `model` (str): HuggingFace model name (e.g., "Qwen/Qwen3.5-9B")
- `context_length` (int): Maximum context length to optimize for (default: 16384)
- `max_iterations` (int): Maximum optimization iterations (default: 25)
- `target_percentile` (float): Stop when reaching this percentile of max throughput (default: 0.95)
- `output_dir` (str): Directory for checkpoints and results (default: "./vllm_optimization")
- `log_file` (Optional[str]): Path to log file (default: None)
- `api_key` (Optional[str]): Anthropic API key (default: reads from ANTHROPIC_API_KEY env var)
- `resume_from_checkpoint` (bool): Resume from existing checkpoint if found (default: True)

**Returns:**

```python
{
    "model": "Qwen/Qwen3.5-9B",
    "context_length": 16384,
    "config": {
        "gpu_memory_utilization": 0.94,
        "max_num_seqs": 420,
        "tensor_parallel_size": 1,
        "block_size": 16,
        "swap_space": 4
    },
    "throughput": 14020.5,
    "iterations": 9,
    "total_time_seconds": 1320,
    "hardware": {
        "gpu_model": "RTX 4090",
        "num_gpus": 8,
        "vram_per_gpu": 24
    }
}
```

**Example:**

```python
from vllm_autopilot import optimize

# Basic usage
config = optimize(
    model="meta-llama/Llama-3.1-8B",
    context_length=8192
)

print(f"Optimal throughput: {config['throughput']} tokens/sec")
print(f"Config: {config['config']}")

# Advanced usage with custom settings
config = optimize(
    model="Qwen/Qwen3.5-9B",
    context_length=16384,
    max_iterations=30,
    target_percentile=0.98,  # Aim for 98th percentile
    output_dir="./my_optimization",
    log_file="optimization.log"
)
```

---

### Database API

#### `ConfigDatabase`

Manages the community configuration database.

```python
from vllm_autopilot.database import ConfigDatabase

db = ConfigDatabase(database_path: str = "configs/database.json")
```

##### Methods

**`query()`** - Search for existing configurations

```python
config = db.query(
    gpu_model: str,
    model_name: str,
    context_length: int,
    num_gpus: int = 1,
    exact_match: bool = False
) -> Optional[dict]
```

**Parameters:**
- `gpu_model`: GPU model name (e.g., "RTX 4090", "A100-80GB")
- `model_name`: HuggingFace model name
- `context_length`: Context length
- `num_gpus`: Number of GPUs (default: 1)
- `exact_match`: If True, only return exact matches (default: False, allows similar configs)

**Returns:** Config dict if found, None otherwise

**Example:**

```python
db = ConfigDatabase()

# Try exact match first
config = db.query(
    gpu_model="RTX 4090",
    model_name="Qwen/Qwen3.5-9B",
    context_length=16384,
    num_gpus=1,
    exact_match=True
)

if not config:
    # Try similar configs
    config = db.query(
        gpu_model="RTX 4090",
        model_name="Qwen/Qwen3.5-9B",
        context_length=16384,
        num_gpus=1,
        exact_match=False
    )

if config:
    print(f"Found config: {config['config']}")
    print(f"Throughput: {config['throughput']} tok/s")
else:
    print("No config found, run optimization")
```

---

**`save_config()`** - Save a new configuration

```python
db.save_config(
    gpu_model: str,
    num_gpus: int,
    model_name: str,
    context_length: int,
    config: dict,
    throughput: float,
    verify: bool = False
) -> None
```

**Parameters:**
- `gpu_model`: GPU model name
- `num_gpus`: Number of GPUs
- `model_name`: HuggingFace model name
- `context_length`: Context length
- `config`: vLLM configuration dict
- `throughput`: Measured throughput (tokens/sec)
- `verify`: If True, increments verification count for existing config (default: False)

**Example:**

```python
db = ConfigDatabase()

db.save_config(
    gpu_model="RTX 4090",
    num_gpus=1,
    model_name="Qwen/Qwen3.5-9B",
    context_length=16384,
    config={
        "gpu_memory_utilization": 0.94,
        "max_num_seqs": 420,
        "tensor_parallel_size": 1,
        "block_size": 16
    },
    throughput=14020.5
)
```

---

**`list_configs()`** - List all configurations

```python
configs = db.list_configs(
    gpu_model: Optional[str] = None,
    model_name: Optional[str] = None
) -> List[dict]
```

**Parameters:**
- `gpu_model`: Filter by GPU model (optional)
- `model_name`: Filter by model name (optional)

**Returns:** List of all matching configurations

**Example:**

```python
db = ConfigDatabase()

# List all configs
all_configs = db.list_configs()

# List configs for specific GPU
rtx_configs = db.list_configs(gpu_model="RTX 4090")

# List configs for specific model
qwen_configs = db.list_configs(model_name="Qwen/Qwen3.5-9B")

for config in rtx_configs:
    print(f"{config['model']} @ {config['context_length']}: {config['throughput']} tok/s")
```

---

### Hardware API

#### `detect_gpus()`

Detect available GPUs in the system.

```python
from vllm_autopilot.hardware import detect_gpus, GPUInfo

gpus: List[GPUInfo] = detect_gpus()
```

**Returns:** List of `GPUInfo` objects

**GPUInfo attributes:**
- `index` (int): GPU index
- `name` (str): GPU model name
- `vram_gb` (int): VRAM in GB
- `compute_capability` (str): CUDA compute capability

**Example:**

```python
from vllm_autopilot.hardware import detect_gpus

gpus = detect_gpus()

print(f"Found {len(gpus)} GPUs:")
for gpu in gpus:
    print(f"  GPU {gpu.index}: {gpu.name} ({gpu.vram_gb} GB)")
```

---

#### `get_cuda_version()`

Get CUDA version.

```python
from vllm_autopilot.hardware import get_cuda_version

cuda_version: str = get_cuda_version()
```

**Returns:** CUDA version string (e.g., "12.4")

---

#### `estimate_model_size()`

Estimate model size in GB.

```python
from vllm_autopilot.hardware import estimate_model_size

size_gb: float = estimate_model_size(model_name: str)
```

**Parameters:**
- `model_name`: HuggingFace model name

**Returns:** Estimated model size in GB

**Example:**

```python
from vllm_autopilot.hardware import estimate_model_size

size = estimate_model_size("Qwen/Qwen3.5-9B")
print(f"Model size: ~{size} GB")
```

---

### Safety Agent API

#### `SafetyAgent`

Validates configurations to prevent OOM errors.

```python
from vllm_autopilot.safety import SafetyAgent

safety = SafetyAgent(
    gpu_vram_gb: int,
    num_gpus: int = 1,
    safety_margin_gb: float = 2.0
)
```

**Parameters:**
- `gpu_vram_gb`: VRAM per GPU in GB
- `num_gpus`: Number of GPUs (default: 1)
- `safety_margin_gb`: Safety margin to leave free (default: 2.0)

##### Methods

**`check_config()`** - Validate a configuration

```python
is_safe, estimate, safe_config = safety.check_config(
    config: dict,
    model_size_gb: float,
    context_length: int
) -> Tuple[bool, MemoryEstimate, Optional[dict]]
```

**Parameters:**
- `config`: vLLM configuration dict
- `model_size_gb`: Model size in GB
- `context_length`: Context length

**Returns:**
- `is_safe` (bool): Whether config is safe to run
- `estimate` (MemoryEstimate): Memory usage breakdown
- `safe_config` (Optional[dict]): Suggested safe config if unsafe, None otherwise

**Example:**

```python
from vllm_autopilot.safety import SafetyAgent

safety = SafetyAgent(gpu_vram_gb=24, num_gpus=1)

config = {
    "gpu_memory_utilization": 0.95,
    "max_num_seqs": 512,
    "tensor_parallel_size": 1,
    "block_size": 16
}

is_safe, estimate, safe_config = safety.check_config(
    config=config,
    model_size_gb=14.0,
    context_length=16384
)

if is_safe:
    print(f"✓ Config is safe (estimated: {estimate.total_gb} GB)")
else:
    print(f"✗ Config unsafe! Estimated: {estimate.total_gb} GB > {24} GB")
    print(f"Suggested safe config: {safe_config}")
```

---

**`record_oom()`** - Record OOM event for learning

```python
safety.record_oom(
    config: dict,
    model_size_gb: float,
    context_length: int,
    actual_memory_used_gb: Optional[float] = None
) -> None
```

**Parameters:**
- `config`: Configuration that caused OOM
- `model_size_gb`: Model size in GB
- `context_length`: Context length
- `actual_memory_used_gb`: Actual memory used before OOM (optional)

**Example:**

```python
# After an OOM occurs
safety.record_oom(
    config=failed_config,
    model_size_gb=14.0,
    context_length=16384,
    actual_memory_used_gb=23.8  # From nvidia-smi
)

# Safety agent will improve future predictions
```

---

### Orchestrator API

#### `OptimizationOrchestrator`

Main orchestrator for the optimization loop.

```python
from vllm_autopilot.orchestrator import OptimizationOrchestrator

orchestrator = OptimizationOrchestrator(
    model: str,
    context_length: int = 16384,
    max_iterations: int = 25,
    target_percentile: float = 0.95,
    output_dir: str = "./vllm_optimization",
    api_key: Optional[str] = None
)
```

##### Methods

**`run()`** - Run the optimization loop

```python
result = orchestrator.run(resume: bool = True) -> dict
```

**Parameters:**
- `resume`: Resume from checkpoint if exists (default: True)

**Returns:** Optimization result dict

**Example:**

```python
from vllm_autopilot.orchestrator import OptimizationOrchestrator

orchestrator = OptimizationOrchestrator(
    model="Qwen/Qwen3.5-9B",
    context_length=16384,
    max_iterations=20
)

result = orchestrator.run(resume=True)
print(f"Optimal config: {result['config']}")
```

---

## CLI Reference

### Commands

#### Optimize

```bash
vllm-autopilot [OPTIONS]
```

**Options:**
- `--model MODEL`: HuggingFace model name (required)
- `--context-length LENGTH`: Context length (default: 16384)
- `--max-iterations N`: Max iterations (default: 25)
- `--target-percentile P`: Target percentile (default: 0.95)
- `--output-dir DIR`: Output directory (default: ./vllm_optimization)
- `--log-file FILE`: Log file path
- `--no-resume`: Don't resume from checkpoint

**Example:**
```bash
vllm-autopilot --model Qwen/Qwen3.5-9B --context-length 16384 --max-iterations 20
```

---

#### Query Database

```bash
vllm-autopilot --query --model MODEL --gpu GPU_MODEL [OPTIONS]
```

**Options:**
- `--model MODEL`: Model name
- `--gpu GPU_MODEL`: GPU model (e.g., "RTX 4090")
- `--context-length LENGTH`: Context length (default: 16384)
- `--num-gpus N`: Number of GPUs (default: 1)

**Example:**
```bash
vllm-autopilot --query --model Qwen/Qwen3.5-9B --gpu "RTX 4090" --context-length 16384
```

---

#### List Configs

```bash
vllm-autopilot --list-configs [--gpu GPU_MODEL] [--model MODEL]
```

**Example:**
```bash
vllm-autopilot --list-configs --gpu "RTX 4090"
```

---

#### Hardware Info

```bash
vllm-autopilot --hardware-info
```

**Example:**
```bash
vllm-autopilot --hardware-info
# Output:
# GPU 0: NVIDIA RTX 4090 (24 GB)
# GPU 1: NVIDIA RTX 4090 (24 GB)
# CUDA Version: 12.4
```

---

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for Claude (required)
- `OPENAI_API_KEY`: API key for GPT (future support)
- `CUDA_VISIBLE_DEVICES`: Set by orchestrator for parallel execution
