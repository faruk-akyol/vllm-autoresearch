# Changelog

All notable changes to vLLM Autopilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release
- LLM-powered autoresearch for vLLM optimization
- Parallel experiment execution across multiple GPUs
- Safety agent for OOM prevention
- Community configuration database
- Command-line interface
- Python API
- Hardware auto-detection
- 95th percentile convergence target
- Checkpoint/resume support

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

## [0.1.0] - 2024-12-XX

### Added
- Core autoresearch loop based on Karpathy's design
- Claude API integration for hypothesis generation
- GPU detection and hardware profiling
- Safety agent with memory estimation
- Parallel benchmark execution
- Configuration database (JSON-based)
- CLI with query, list, and optimize modes
- Example scripts
- Documentation (README, QUICKSTART, CONTRIBUTING)

---

## Version History

- **0.1.0** - Initial release with core features
- **0.2.0** (Planned) - Web UI, OpenAI support, improved safety agent
- **0.3.0** (Planned) - Multi-objective optimization (throughput + latency)
- **1.0.0** (Planned) - Stable API, production-ready
