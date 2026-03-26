"""
vLLM Autopilot - Automatic vLLM configuration optimization.

Inspired by Karpathy's autoresearch, uses LLM agents to find optimal
vLLM parameters for maximum throughput on your GPU.
"""

__version__ = "0.1.0"

from vllm_autopilot.orchestrator import optimize
from vllm_autopilot.database import ConfigDatabase

__all__ = ["optimize", "ConfigDatabase"]
