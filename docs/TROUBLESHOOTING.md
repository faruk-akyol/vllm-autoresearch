# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Problem: `pip install vllm-autopilot` fails

**Solution 1: Check Python version**
```bash
python --version
# Should be 3.10 or higher
```

**Solution 2: Upgrade pip**
```bash
pip install --upgrade pip
pip install vllm-autopilot
```

**Solution 3: Install from source**
```bash
git clone https://github.com/yourusername/vllm-autopilot
cd vllm-autopilot
pip install -e .
```

---

#### Problem: `ModuleNotFoundError: No module named 'vllm'`

**Solution:**
```bash
pip install vllm
```

If that fails (vLLM can be tricky):
```bash
# Try with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# Or build from source
git clone https://github.com/vllm-project/vllm
cd vllm
pip install -e .
```

---

### API Key Issues

#### Problem: `ANTHROPIC_API_KEY not set`

**Solution:**
```bash
# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (CMD)
set ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."

# Permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

**Alternative: Pass directly**
```python
from vllm_autopilot import optimize

config = optimize(
    model="Qwen/Qwen3.5-9B",
    api_key="sk-ant-..."  # Pass directly
)
```

---

#### Problem: `AuthenticationError: Invalid API key`

**Solution:**
1. Check API key is correct (starts with `sk-ant-`)
2. Verify API key is active at https://console.anthropic.com/
3. Check for extra spaces or quotes in environment variable

```bash
# Debug: Print current value
echo $ANTHROPIC_API_KEY

# Should NOT have quotes
# Wrong: "sk-ant-..."
# Right: sk-ant-...
```

---

### GPU Detection Issues

#### Problem: `No NVIDIA GPUs detected`

**Solution 1: Check nvidia-smi works**
```bash
nvidia-smi
# Should show GPU info
```

**Solution 2: Install/update NVIDIA drivers**
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install nvidia-driver-535

# Check installation
nvidia-smi
```

**Solution 3: Check CUDA installation**
```bash
nvcc --version
# Should show CUDA version

# If not installed:
# Download from https://developer.nvidia.com/cuda-downloads
```

---

#### Problem: `CUDA out of memory` during optimization

**Solution 1: Reduce safety margin**
```python
from vllm_autopilot.safety import SafetyAgent

# More aggressive memory usage
safety = SafetyAgent(
    gpu_vram_gb=24,
    safety_margin_gb=1.0  # Default is 2.0
)
```

**Solution 2: Kill other processes**
```bash
# Check GPU memory usage
nvidia-smi

# Kill processes using GPU
sudo kill -9 <PID>
```

**Solution 3: Restart with lower initial config**
```bash
# Start more conservatively
vllm-autopilot --model Qwen/Qwen3.5-9B --context-length 8192
```

---

### Optimization Issues

#### Problem: Optimization takes too long (>1 hour)

**Possible Causes:**
1. Too many iterations
2. Sequential execution (should be parallel)
3. Large model + single GPU
4. Slow network (API calls to Claude)

**Solution 1: Reduce iterations**
```bash
vllm-autopilot --model Qwen/Qwen3.5-9B --max-iterations 10
```

**Solution 2: Lower target percentile**
```bash
vllm-autopilot --model Qwen/Qwen3.5-9B --target-percentile 0.90
```

**Solution 3: Check parallel execution is working**
```python
# Check logs for parallel execution
# Should see: "Running 8 parallel experiments..."
# Not: "Running experiment 1/8... experiment 2/8..."
```

**Solution 4: Use faster model for agent**
```python
# Future feature - use GPT-4o-mini instead of Claude
# Currently only Claude Sonnet supported
```

---

#### Problem: Optimization converges too quickly (iteration 2-3)

**Possible Causes:**
1. Database has existing config (optimization skipped)
2. Model is very small and easy to optimize
3. 95th percentile target too low

**Solution 1: Check if database was queried**
```bash
# Look for this in output:
# "✓ Found existing config in database"
# If found, that's why it was fast!

# To force fresh optimization:
rm configs/database.json  # Delete database
vllm-autopilot --model Qwen/Qwen3.5-9B
```

**Solution 2: Increase target**
```bash
vllm-autopilot --model Qwen/Qwen3.5-9B --target-percentile 0.98
```

---

#### Problem: All experiments cause OOM

**Possible Causes:**
1. Model too large for GPU
2. Safety agent is misconfigured
3. Context length too long

**Solution 1: Reduce context length**
```bash
vllm-autopilot --model Qwen/Qwen3.5-9B --context-length 8192
# Instead of 16384 or 32768
```

**Solution 2: Use tensor parallelism**
```bash
# For large models, use multiple GPUs
# Safety agent will automatically suggest TP configs
```

**Solution 3: Check model size vs VRAM**
```python
from vllm_autopilot.hardware import estimate_model_size, detect_gpus

model_size = estimate_model_size("Qwen/Qwen3.5-70B")
gpus = detect_gpus()

print(f"Model: {model_size} GB")
print(f"GPU VRAM: {gpus[0].vram_gb} GB")

# If model > VRAM, need tensor parallelism
# Qwen 70B (~140GB) needs TP=8 on 8× 24GB GPUs
```

---

#### Problem: Throughput is very low (<100 tokens/sec)

**Possible Causes:**
1. Benchmark is misconfigured
2. GPU is throttling (thermal)
3. Other processes competing for GPU
4. vLLM version incompatibility

**Solution 1: Check GPU utilization**
```bash
nvidia-smi dmon -i 0
# Should show high GPU utilization (>80%)
# If low, something is wrong
```

**Solution 2: Check thermal throttling**
```bash
nvidia-smi
# Look at temperature and power usage
# Temp >85°C = may be throttling
```

**Solution 3: Test vLLM directly**
```bash
python -m vllm.benchmarks.benchmark_throughput \
  --model Qwen/Qwen3.5-9B \
  --input-len 1024 \
  --output-len 512 \
  --num-prompts 50

# Should get >1000 tok/s with default settings
# If not, vLLM installation issue
```

**Solution 4: Update vLLM**
```bash
pip install --upgrade vllm
```

---

### Database Issues

#### Problem: `configs/database.json` not found

**Solution:**
```bash
# Create empty database
mkdir -p configs
echo '{"configs": []}' > configs/database.json
```

**Or let the tool create it:**
```python
from vllm_autopilot.database import ConfigDatabase

db = ConfigDatabase()  # Auto-creates if missing
```

---

#### Problem: Database queries return no results

**Possible Causes:**
1. Exact match not found
2. GPU model name mismatch
3. Empty database

**Solution 1: Check database contents**
```bash
cat configs/database.json | jq '.configs | length'
# Shows number of configs

cat configs/database.json | jq '.configs[0]'
# Shows first config
```

**Solution 2: Use fuzzy matching**
```python
from vllm_autopilot.database import ConfigDatabase

db = ConfigDatabase()

# Try exact match
config = db.query(
    gpu_model="RTX 4090",
    model_name="Qwen/Qwen3.5-9B",
    exact_match=True
)

if not config:
    # Try fuzzy match
    config = db.query(
        gpu_model="RTX 4090",
        model_name="Qwen/Qwen3.5-9B",
        exact_match=False
    )
```

**Solution 3: Check GPU model naming**
```python
from vllm_autopilot.hardware import detect_gpus

gpus = detect_gpus()
print(gpus[0].name)  # Use exact name from here

# Example output: "NVIDIA GeForce RTX 4090"
# Not: "RTX 4090" or "4090"
```

---

### Checkpoint Issues

#### Problem: `checkpoint.json` corrupted

**Solution:**
```bash
# Remove corrupted checkpoint
rm vllm_optimization/checkpoint.json

# Restart optimization
vllm-autopilot --model Qwen/Qwen3.5-9B --no-resume
```

---

#### Problem: Can't resume from checkpoint

**Possible Causes:**
1. Checkpoint from different version
2. Model name changed
3. Hardware changed (different GPUs)

**Solution:**
```bash
# Force fresh start
vllm-autopilot --model Qwen/Qwen3.5-9B --no-resume

# Or delete checkpoint directory
rm -rf vllm_optimization/
```

---

### Performance Issues

#### Problem: Agent takes too long to generate experiments

**Possible Causes:**
1. Slow API response from Claude
2. Large history (many iterations)
3. Network latency

**Solution 1: Check API latency**
```python
import time
from anthropic import Anthropic

client = Anthropic()
start = time.time()
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}]
)
print(f"API latency: {time.time() - start:.2f}s")

# Should be <2 seconds
# If >5 seconds, network issue
```

**Solution 2: Prune history**
```python
# Future feature - limit history to last 20 experiments
# Currently loads full history
```

---

#### Problem: Parallel execution is slow

**Possible Causes:**
1. Not actually running in parallel
2. GPU contention
3. ProcessPoolExecutor overhead

**Solution 1: Verify parallel execution**
```bash
# During optimization, open another terminal:
watch -n 1 nvidia-smi

# Should see multiple GPUs active simultaneously
```

**Solution 2: Check ProcessPoolExecutor**
```python
# In orchestrator.py, verify:
max_workers = min(len(configs), num_gpus)
# Should be >1 for multi-GPU

# Check CUDA_VISIBLE_DEVICES is set per worker
```

---

### Error Messages

#### `RuntimeError: CUDA error: out of memory`

See "CUDA out of memory" section above.

---

#### `FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'`

**Solution:**
```bash
# Install NVIDIA drivers
sudo apt-get install nvidia-driver-535

# Or add to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

---

#### `ImportError: cannot import name 'LLM' from 'vllm'`

**Solution:**
```bash
# vLLM version mismatch
pip install vllm==0.6.0  # Or latest stable

# Check version
pip show vllm
```

---

#### `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

**Possible Causes:**
1. Corrupted database file
2. Empty response from API
3. Malformed checkpoint

**Solution:**
```bash
# Check which file is corrupted
# Database:
cat configs/database.json

# Checkpoint:
cat vllm_optimization/checkpoint.json

# If corrupted, reset:
echo '{"configs": []}' > configs/database.json
rm vllm_optimization/checkpoint.json
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Then run optimization
from vllm_autopilot import optimize
config = optimize(...)
```

---

### Monitor GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi dmon -i 0 -o T > gpu_usage.log &

# Monitor temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1
```

---

### Check vLLM Benchmark Directly

```bash
# Test vLLM is working
python -m vllm.benchmarks.benchmark_throughput \
  --model Qwen/Qwen3.5-9B \
  --input-len 1024 \
  --output-len 512 \
  --num-prompts 10 \
  --tensor-parallel-size 1

# Should complete without errors
```

---

### Verbose Output

```bash
# Run with verbose logging
vllm-autopilot --model Qwen/Qwen3.5-9B --log-file debug.log

# Check log file
tail -f debug.log
```

---

## Getting Help

If none of these solutions work:

1. **Check GitHub Issues**: https://github.com/yourusername/vllm-autopilot/issues
2. **Open New Issue** with:
   - Python version (`python --version`)
   - vLLM version (`pip show vllm`)
   - GPU info (`nvidia-smi`)
   - Full error traceback
   - Checkpoint file (if relevant)
3. **Discussions**: https://github.com/yourusername/vllm-autopilot/discussions

---

## Known Issues

### v0.1.0

1. **Safety agent memory estimation can be inaccurate**
   - Affects: Large models with many layers
   - Workaround: Manually adjust safety margin
   - Fix planned: v0.2.0 with improved estimation

2. **Windows support is experimental**
   - Affects: ProcessPoolExecutor on Windows
   - Workaround: Use WSL2 or Linux
   - Fix planned: v0.2.0

3. **Only Claude API supported**
   - Affects: Users without Anthropic API key
   - Workaround: None currently
   - Fix planned: v0.2.0 will add OpenAI GPT support

4. **Database has no verification system**
   - Affects: Trust in community configs
   - Workaround: Manually verify configs
   - Fix planned: v0.2.0 with verification/reputation

5. **No web UI**
   - Affects: Users who prefer GUI
   - Workaround: Use CLI or Python API
   - Fix planned: v0.2.0
