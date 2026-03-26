"""
LLM Agent - Forms hypotheses and designs experiments.

Uses Claude/GPT to analyze benchmark results and generate next configs to test.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import anthropic


class ResearchAgent:
    """LLM-powered research agent for vLLM optimization."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250929"):
        """
        Initialize research agent.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

        # Load research program
        program_path = Path(__file__).parent.parent / "program.md"
        self.program = program_path.read_text()

    def generate_experiments(
        self,
        iteration: int,
        history: List[Dict[str, Any]],
        num_experiments: int,
        hardware_info: Dict[str, Any],
        model_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate next batch of experiments based on history.

        Args:
            iteration: Current iteration number
            history: List of previous experiment results
            num_experiments: Number of parallel experiments to generate
            hardware_info: GPU/hardware information
            model_info: Model name, size, context length

        Returns:
            Dict with:
                - hypothesis: str (agent's reasoning)
                - experiments: List[Dict] (configs to test)
                - prediction: str (expected outcomes)
        """
        # Build prompt for agent
        prompt = self._build_prompt(iteration, history, num_experiments, hardware_info, model_info)

        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        try:
            # Extract JSON from response
            content = response.content[0].text

            # Find JSON block (between ```json and ```)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                # Try to parse entire content as JSON
                json_str = content.strip()

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            # Fallback: return simple default if parsing fails
            print(f"Warning: Failed to parse agent response: {e}")
            return self._generate_fallback_experiments(num_experiments, history)

    def _build_prompt(
        self,
        iteration: int,
        history: List[Dict[str, Any]],
        num_experiments: int,
        hardware_info: Dict[str, Any],
        model_info: Dict[str, Any],
    ) -> str:
        """Build prompt for agent."""
        prompt = f"""You are a research agent optimizing vLLM throughput.

## Research Program

{self.program}

## Current Status

**Iteration:** {iteration}
**Hardware:** {hardware_info['num_gpus']}× {hardware_info['gpu_model']} ({hardware_info['vram_per_gpu']} GB each)
**Model:** {model_info['model_name']} (~{model_info['size_gb']:.1f} GB)
**Context Length:** {model_info['context_length']} tokens

## Experiment History

"""

        # Add previous results
        if history:
            for exp in history[-10:]:  # Last 10 experiments
                prompt += f"\n**Iteration {exp['iteration']}:**\n"
                if 'hypothesis' in exp:
                    prompt += f"Hypothesis: {exp['hypothesis']}\n"
                prompt += f"Config: {json.dumps(exp['config'], indent=2)}\n"
                prompt += f"Result: {exp['result']['throughput']:.1f} tok/s ({exp['result']['status']})\n"
        else:
            prompt += "\nNo previous experiments yet. This is iteration 1.\n"

        prompt += f"""

## Your Task

Generate {num_experiments} experiment configs to test in parallel.

**Requirements:**
1. Analyze previous results (if any)
2. Form a scientific hypothesis about what to test next
3. Design {num_experiments} different configs to test your hypothesis
4. Predict expected outcomes
5. Return results as JSON

**JSON Format:**

```json
{{
  "hypothesis": "Your reasoning here - why these configs?",
  "experiments": [
    {{
      "gpu_memory_utilization": 0.90,
      "max_num_seqs": 256,
      "tensor_parallel_size": 1,
      "block_size": 16,
      "swap_space": 4
    }},
    ... ({num_experiments} total)
  ],
  "prediction": "Which config do you expect to perform best and why?"
}}
```

**Strategy Tips:**
- Start with vLLM defaults if iteration 1
- Test parameter ranges systematically
- Learn from OOM failures (reduce params)
- Use parallel experiments to explore grid
- Aim for 95th percentile, not perfection

Now generate the {num_experiments} experiments:
"""

        return prompt

    def _generate_fallback_experiments(
        self, num_experiments: int, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate simple fallback experiments if agent fails."""
        # Default baseline configs
        base_configs = [
            {"gpu_memory_utilization": 0.85, "max_num_seqs": 128, "tensor_parallel_size": 1, "block_size": 16, "swap_space": 4},
            {"gpu_memory_utilization": 0.90, "max_num_seqs": 256, "tensor_parallel_size": 1, "block_size": 16, "swap_space": 4},
            {"gpu_memory_utilization": 0.95, "max_num_seqs": 512, "tensor_parallel_size": 1, "block_size": 16, "swap_space": 4},
        ]

        # Repeat configs to fill num_experiments
        experiments = []
        for i in range(num_experiments):
            experiments.append(base_configs[i % len(base_configs)])

        return {
            "hypothesis": "Fallback: Testing baseline configs due to agent parsing error",
            "experiments": experiments,
            "prediction": "Moderate throughput expected",
        }

    def analyze_results(
        self, iteration: int, experiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze experiment results and provide insights.

        Args:
            iteration: Current iteration
            experiments: List of experiment results

        Returns:
            Dict with analysis and recommendations
        """
        prompt = f"""Analyze these vLLM benchmark results from iteration {iteration}:

{json.dumps(experiments, indent=2)}

Provide:
1. **Best config**: Which performed best?
2. **Patterns**: What trends do you see?
3. **Insights**: What did we learn?
4. **Convergence**: Are we converging on optimal?

Return as JSON:
```json
{{
  "best_config": {{"gpu_memory_utilization": ..., "max_num_seqs": ...}},
  "best_throughput": 1234.5,
  "patterns": "What patterns emerged",
  "insights": "Key learnings",
  "converged": true/false,
  "recommendation": "Continue or stop"
}}
```
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            content = response.content[0].text
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()

            return json.loads(json_str)
        except:
            # Fallback analysis
            best_exp = max(experiments, key=lambda x: x["result"].get("throughput", 0))
            return {
                "best_config": best_exp["config"],
                "best_throughput": best_exp["result"]["throughput"],
                "patterns": "Unable to analyze - parsing error",
                "insights": "See raw results",
                "converged": False,
                "recommendation": "continue",
            }


if __name__ == "__main__":
    # Test agent
    agent = ResearchAgent()

    hardware_info = {
        "num_gpus": 8,
        "gpu_model": "A100-80GB",
        "vram_per_gpu": 80,
    }

    model_info = {
        "model_name": "Qwen/Qwen3.5-9B",
        "size_gb": 18.0,
        "context_length": 16384,
    }

    print("Generating experiments for iteration 1...")
    result = agent.generate_experiments(
        iteration=1,
        history=[],
        num_experiments=8,
        hardware_info=hardware_info,
        model_info=model_info,
    )

    print("\nAgent Output:")
    print(json.dumps(result, indent=2))
