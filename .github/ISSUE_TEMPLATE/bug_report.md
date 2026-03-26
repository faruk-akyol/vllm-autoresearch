---
name: Bug report
about: Report a bug or issue
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run command: `vllm-autopilot --model ...`
2. See error at iteration X
3. Error message: ...

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- **GPU Model**: (e.g., RTX 4090, A100 80GB)
- **Number of GPUs**: (e.g., 8)
- **CUDA Version**: (run `nvidia-smi`)
- **vLLM Version**: (run `pip show vllm`)
- **vLLM Autopilot Version**: (run `pip show vllm-autopilot`)
- **Python Version**: (run `python --version`)
- **OS**: (e.g., Ubuntu 22.04, Windows 11)

## Model Being Optimized
- **Model Name**: (e.g., Qwen/Qwen3.5-9B)
- **Context Length**: (e.g., 16384)

## Error Output
```
Paste full error message and traceback here
```

## Logs
If available, attach:
- `vllm_optimization/checkpoint.json`
- Full terminal output
- Any relevant log files

## Additional Context
Any other context about the problem.
