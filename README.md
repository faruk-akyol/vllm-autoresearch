# vLLM Autoresearch

Automatically find optimal vLLM configurations for your GPU using LLM-powered autoresearch.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

vLLM has numerous parameters (`gpu_memory_utilization`, `max_num_seqs`, `tensor_parallel_size`, etc.) that significantly impact throughput. Finding optimal settings manually requires extensive trial and error.

This tool automates the optimization process using an LLM agent to intelligently explore the parameter space, similar to how a researcher would conduct experiments.

## Inspiration

This project is inspired by:
- **[Karpathy's autoresearch](https://github.com/karpathy/autoresearch)** - LLM-guided scientific research methodology
- **HuggingFace's vLLM benchmarking research** - Systematic exploration of vLLM configurations

## Features

- **LLM-Guided Optimization** - Agent forms hypotheses and designs experiments
- **Parallel Execution** - Run multiple experiments simultaneously on multi-GPU systems
- **Safety Agent** - Prevents OOM crashes through memory estimation
- **95th Percentile Target** - Stops when diminishing returns detected
- **Community Database** - Share and reuse optimal configurations
- **Checkpoint/Resume** - Save progress after each iteration

## Installation

```bash
pip install vllm-autopilot
```

Or from source:

```bash
git clone https://github.com/faruk-akyol/vllm-autoresearch
cd vllm-autoresearch
pip install -e .
```

**Requirements:**
- Python 3.10+
- NVIDIA GPU with CUDA
- vLLM 0.6.0+
- Anthropic API key (for Claude agent)

## Quick Start

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run optimization
vllm-autopilot --model Qwen/Qwen3.5-9B --context-length 16384

# Typical runtime: 15-30 minutes
```

## Usage

### Command Line

```bash
# Basic optimization
vllm-autopilot --model meta-llama/Llama-3.1-8B --context-length 8192

# Query database first
vllm-autopilot --query --model Qwen/Qwen3.5-9B --gpu "RTX 4090"

# Custom settings
vllm-autopilot \
  --model Qwen/Qwen3.5-9B \
  --context-length 16384 \
  --max-iterations 25 \
  --target-percentile 0.95
```

### Python API

```python
from vllm_autopilot import optimize
from vllm_autopilot.database import ConfigDatabase

# Query database
db = ConfigDatabase()
config = db.query(
    gpu_model="RTX 4090",
    model_name="Qwen/Qwen3.5-9B",
    context_length=16384,
    num_gpus=1
)

# Run optimization if not found
if not config:
    config = optimize(
        model="Qwen/Qwen3.5-9B",
        context_length=16384,
        max_iterations=20
    )

print(f"Optimal config: {config['config']}")
print(f"Throughput: {config['throughput']} tokens/sec")
```

## How It Works

The optimization follows the autoresearch pattern:

1. **Hardware Detection** - Detect GPUs, VRAM, CUDA version
2. **Database Query** - Check for existing optimal configs
3. **Hypothesis Generation** - LLM agent analyzes history and forms hypothesis
4. **Safety Validation** - Estimate memory usage, reject unsafe configs
5. **Parallel Execution** - Run experiments across available GPUs
6. **Analysis** - Agent learns from results, forms new hypothesis
7. **Convergence** - Stop when 95th percentile target reached

## Parameters Optimized

**Critical Parameters:**
- `gpu_memory_utilization` (0.80 - 0.98) - Fraction of VRAM for KV cache
- `max_num_seqs` (64 - 16384) - Maximum concurrent sequences
- `tensor_parallel_size` (1, 2, 4, 8) - Number of GPUs for model parallelism

**Secondary Parameters:**
- `block_size` (8, 16, 32) - KV cache block size
- `swap_space` (0 - 16 GB) - CPU RAM for overflow

## Example Output

```
🔍 Detecting Hardware...
✓ Found 8× NVIDIA RTX 4090 (24 GB each)

📊 Checking Database...
✗ No existing config found

🚀 Starting Optimization...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration 1/25
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hypothesis: Explore baseline configurations

Running 8 parallel experiments...
[GPU 0] mem=0.90, seqs=256  → 11,234 tok/s ✓
[GPU 1] mem=0.95, seqs=256  → 11,450 tok/s ✓
[GPU 2] mem=0.90, seqs=512  → OOM ✗
...

Best: 12,890 tok/s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration 9/25
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 95th percentile reached!

╔═══════════════════════════════════════════╗
║        OPTIMIZATION COMPLETE              ║
╠═══════════════════════════════════════════╣
║ Optimal Config:                           ║
║   gpu_memory_utilization: 0.94            ║
║   max_num_seqs: 420                       ║
║   tensor_parallel_size: 1                 ║
║   block_size: 16                          ║
║                                           ║
║ Throughput: 14,020 tokens/sec            ║
║ Time: 22 minutes                          ║
╚═══════════════════════════════════════════╝
```

## Project Structure

```
vllm-autoresearch/
├── vllm_autopilot/
│   ├── hardware.py          # GPU detection
│   ├── benchmark_template.py # vLLM benchmark runner
│   ├── agent.py             # LLM hypothesis agent
│   ├── safety.py            # OOM prevention
│   ├── orchestrator.py      # Main optimization loop
│   ├── database.py          # Config storage
│   └── cli.py               # Command-line interface
├── configs/
│   └── database.json        # Community configs
├── docs/
│   ├── API.md               # API documentation
│   ├── ARCHITECTURE.md      # System design
│   ├── CONFIGURATION.md     # Parameter guide
│   ├── TROUBLESHOOTING.md   # Common issues
│   └── FAQ.md               # Questions & answers
├── examples/
│   ├── optimize_qwen.py
│   └── optimize_llama.py
├── program.md               # Research instructions for agent
└── README.md
```

## Architecture

The system consists of four main components:

1. **Orchestrator** - Manages optimization loop, convergence tracking
2. **Hypothesis Agent** - LLM-powered experiment generation (Claude/GPT)
3. **Safety Agent** - Memory estimation and OOM prevention
4. **Parallel Scheduler** - Distributes experiments across GPUs

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [API Reference](docs/API.md) - Python API and CLI documentation
- [Configuration Guide](docs/CONFIGURATION.md) - Parameter explanations
- [Architecture](docs/ARCHITECTURE.md) - System design
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues
- [FAQ](docs/FAQ.md) - Frequently asked questions

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- Share optimal configs to the database
- Report bugs or suggest features
- Improve documentation
- Submit code improvements

## Community Database

The community database stores optimal configurations for different hardware and model combinations. When you find an optimal config, you can contribute it to help others with similar setups.

Each config includes:
- GPU model and count
- Model name and context length
- vLLM parameters
- Measured throughput
- Verification count

## Roadmap

**v0.1.0** (Current)
- Core autoresearch loop
- Parallel experiment execution
- Safety agent
- Community database
- CLI and Python API

**v0.2.0** (Planned)
- Web UI for browsing configs
- OpenAI GPT support
- Improved safety agent
- Database verification system

**v0.3.0** (Planned)
- Multi-objective optimization (throughput + latency)
- Support for other engines (llama.cpp, TensorRT-LLM)
- Ray integration for distributed optimization

**v1.0.0** (Planned)
- Stable API
- Production-ready safety guarantees
- Comprehensive test coverage

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- Built for the [vLLM](https://github.com/vllm-project/vllm) community
- Powered by [Claude](https://anthropic.com) (Anthropic)

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch)
- [HuggingFace LLM Performance Benchmarks](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)

## Support

- **Issues**: [GitHub Issues](https://github.com/faruk-akyol/vllm-autoresearch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/faruk-akyol/vllm-autoresearch/discussions)
