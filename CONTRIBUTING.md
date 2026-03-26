# Contributing to vLLM Autopilot

Thank you for your interest in contributing! 🎉

## How to Contribute

### Reporting Bugs

Found a bug? Please open an issue with:
- Your hardware specs (GPU model, VRAM, CUDA version)
- vLLM version
- Model being optimized
- Error message and traceback
- Steps to reproduce

### Suggesting Features

Have an idea? Open an issue with:
- Clear description of the feature
- Use case / motivation
- Example of how it would work

### Contributing Configs

Share your optimal configs! To add a config to the database:

1. Run optimization on your hardware
2. When prompted, choose "Yes" to save to database
3. Fork the repo
4. Commit your updated `configs/database.json`
5. Submit a PR with:
   - Hardware specs in PR description
   - Throughput achieved
   - Verification (screenshot or log)

### Contributing Code

#### Development Setup

```bash
# Clone repo
git clone https://github.com/yourusername/vllm-autopilot
cd vllm-autopilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black vllm_autopilot/
ruff check vllm_autopilot/
```

#### Code Style

- **Python 3.10+** syntax
- **Black** for formatting (line length 100)
- **Type hints** for all functions
- **Docstrings** for public APIs
- **Tests** for new features

#### Pull Request Process

1. **Fork** the repository
2. **Create branch** from `main`: `git checkout -b feature/your-feature`
3. **Make changes** with clear commits
4. **Add tests** if adding features
5. **Update docs** if changing APIs
6. **Run tests**: `pytest`
7. **Format code**: `black . && ruff check .`
8. **Push** to your fork
9. **Create PR** with clear description

#### PR Checklist

- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
- [ ] Black formatted
- [ ] Ruff checks pass
- [ ] Commit messages are clear

### Improving Documentation

Documentation improvements are always welcome!

- Fix typos
- Add examples
- Clarify confusing sections
- Add tutorials

Just edit the markdown files and submit a PR.

## Development Guidelines

### Architecture

- **`hardware.py`**: Fixed code, GPU detection (never modified by agent)
- **`benchmark_template.py`**: Template modified by agent each iteration
- **`agent.py`**: LLM reasoning and hypothesis generation
- **`safety.py`**: OOM prevention before running benchmarks
- **`orchestrator.py`**: Main optimization loop
- **`database.py`**: Config storage and retrieval

### Adding New Features

When adding features, consider:

1. **Does it fit the autoresearch pattern?** - Stay true to Karpathy's design
2. **Is it hardware-agnostic?** - Should work on different GPUs
3. **Is it model-agnostic?** - Should work on different LLMs
4. **Does it improve speed or accuracy?** - Core goals

### Testing

We need tests for:
- Hardware detection
- Safety agent predictions
- Database operations
- Config parsing
- End-to-end optimization (mocked)

Example test:
```python
def test_safety_agent():
    safety = SafetyAgent(gpu_vram_gb=24)
    config = {"gpu_memory_utilization": 0.95, "max_num_seqs": 512}
    is_safe, estimate, _ = safety.check_config(config, model_size_gb=14, context_length=8192)
    assert isinstance(is_safe, bool)
    assert estimate.total_gb > 0
```

### Adding Support for Other LLM Providers

Want to add OpenAI/Gemini support?

1. Create new agent class: `OpenAIAgent(ResearchAgent)`
2. Implement `generate_experiments()` method
3. Add `--agent-provider` CLI flag
4. Add tests
5. Update docs

## Community

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs/features
- **PRs**: Use Pull Requests for code contributions

## Code of Conduct

Be respectful and constructive. We're here to build cool stuff together! 🚀

## Questions?

Open a GitHub Discussion or comment on an issue. We're happy to help!
