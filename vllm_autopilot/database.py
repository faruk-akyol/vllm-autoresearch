"""
Configuration Database - Store and retrieve optimal configs.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


class ConfigDatabase:
    """Simple JSON-based configuration database."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database.

        Args:
            db_path: Path to database file (default: configs/database.json)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "configs" / "database.json"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or create database
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = {"version": "1.0.0", "configs": []}
            self._save()

    def query(
        self,
        gpu_model: str,
        model_name: str,
        context_length: int,
        num_gpus: int = 1,
    ) -> Optional[Dict]:
        """
        Query for existing config.

        Args:
            gpu_model: GPU model name (e.g., "A100-80GB")
            model_name: LLM model name
            context_length: Context length in tokens
            num_gpus: Number of GPUs

        Returns:
            Config dict if found, None otherwise
        """
        # Try exact match first
        for config in self.data["configs"]:
            if (
                config["gpu_model"] == gpu_model
                and config["model_name"] == model_name
                and config["context_length"] == context_length
                and config["num_gpus"] == num_gpus
            ):
                return config

        # Try similar match (same GPU/model, different context)
        similar = []
        for config in self.data["configs"]:
            if config["gpu_model"] == gpu_model and config["model_name"] == model_name:
                similar.append(config)

        if similar:
            # Return closest context length match
            similar.sort(key=lambda x: abs(x["context_length"] - context_length))
            return similar[0]

        return None

    def save_config(
        self,
        gpu_model: str,
        num_gpus: int,
        vram_per_gpu: int,
        model_name: str,
        model_size_gb: float,
        context_length: int,
        config: Dict,
        throughput: float,
    ) -> None:
        """
        Save optimal configuration.

        Args:
            gpu_model: GPU model name
            num_gpus: Number of GPUs
            vram_per_gpu: VRAM per GPU in GB
            model_name: LLM model name
            model_size_gb: Model size in GB
            context_length: Context length
            config: Optimal vLLM configuration
            throughput: Achieved throughput
        """
        # Check if config already exists
        existing_idx = None
        for idx, entry in enumerate(self.data["configs"]):
            if (
                entry["gpu_model"] == gpu_model
                and entry["model_name"] == model_name
                and entry["context_length"] == context_length
                and entry["num_gpus"] == num_gpus
            ):
                existing_idx = idx
                break

        new_entry = {
            "gpu_model": gpu_model,
            "num_gpus": num_gpus,
            "vram_per_gpu_gb": vram_per_gpu,
            "model_name": model_name,
            "model_size_gb": model_size_gb,
            "context_length": context_length,
            "config": config,
            "throughput_tokens_per_sec": throughput,
            "verified_count": 1,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if existing_idx is not None:
            # Update existing
            old_entry = self.data["configs"][existing_idx]
            new_entry["verified_count"] = old_entry.get("verified_count", 0) + 1

            # Keep better throughput
            if throughput > old_entry.get("throughput_tokens_per_sec", 0):
                self.data["configs"][existing_idx] = new_entry
                print(f"✓ Updated config with better throughput: {throughput:.1f} tok/s")
            else:
                # Just increment verification count
                self.data["configs"][existing_idx]["verified_count"] += 1
                print(f"✓ Incremented verification count")
        else:
            # Add new
            self.data["configs"].append(new_entry)
            print(f"✓ Added new config to database")

        self._save()

    def _save(self) -> None:
        """Save database to disk."""
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def list_configs(self, gpu_model: Optional[str] = None) -> List[Dict]:
        """
        List all configs, optionally filtered by GPU model.

        Args:
            gpu_model: Filter by GPU model (optional)

        Returns:
            List of config dicts
        """
        configs = self.data["configs"]

        if gpu_model:
            configs = [c for c in configs if c["gpu_model"] == gpu_model]

        return configs

    def print_config(self, config: Dict) -> None:
        """Print config in readable format."""
        print("\n" + "=" * 60)
        print("CONFIGURATION FROM DATABASE")
        print("=" * 60)

        print(f"\nHardware:")
        print(f"  {config['num_gpus']}× {config['gpu_model']} ({config['vram_per_gpu_gb']}GB each)")

        print(f"\nModel:")
        print(f"  {config['model_name']} (~{config['model_size_gb']:.1f} GB)")
        print(f"  Context: {config['context_length']} tokens")

        print(f"\nOptimal Configuration:")
        for key, value in config["config"].items():
            print(f"  {key}: {value}")

        print(f"\nPerformance:")
        print(f"  Throughput: {config['throughput_tokens_per_sec']:.1f} tokens/sec")

        print(f"\nMetadata:")
        print(f"  Verified by: {config['verified_count']} user(s)")
        print(f"  Last updated: {config['last_updated']}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test database
    db = ConfigDatabase()

    # Save test config
    db.save_config(
        gpu_model="A100-80GB",
        num_gpus=8,
        vram_per_gpu=80,
        model_name="Qwen/Qwen3.5-9B",
        model_size_gb=18.0,
        context_length=16384,
        config={
            "gpu_memory_utilization": 0.95,
            "max_num_seqs": 16384,
            "tensor_parallel_size": 8,
            "block_size": 16,
            "swap_space": 4,
        },
        throughput=27500.0,
    )

    # Query
    result = db.query(
        gpu_model="A100-80GB",
        model_name="Qwen/Qwen3.5-9B",
        context_length=16384,
        num_gpus=8,
    )

    if result:
        db.print_config(result)
    else:
        print("No config found")
