"""
Checkpoint management for model saving and loading.
Supports JSON format for human-readable weights.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict


class CheckpointManager:
    """Manages model checkpointing with multiple formats."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _convert_to_serializable(self, state_dict: Dict[str, List]) -> Dict[str, List]:
        """Convert Value objects or floats to raw data."""
        serializable_state = {}
        for key, matrix in state_dict.items():
            serializable_row = []
            for row in matrix:
                serializable_val = []
                for v in row:
                    if hasattr(v, "data"):
                        serializable_val.append(v.data)
                    else:
                        serializable_val.append(float(v))
                serializable_row.append(serializable_val)
            serializable_state[key] = serializable_row
        return serializable_state

    def save_json(
        self,
        state_dict: Dict[str, List],
        config: Any,
        step: int,
        loss: float,
        filename: Optional[str] = None,
    ):
        """Save checkpoint in JSON format (human-readable, larger)."""
        if filename is None:
            filename = f"checkpoint_step_{step}.json"

        serializable_state = self._convert_to_serializable(state_dict)

        checkpoint = {
            "step": step,
            "loss": loss,
            "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
            "state_dict": serializable_state,
        }

        path = self.checkpoint_dir / filename
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        return path

    def save_pickle(
        self,
        state_dict: Dict[str, List],
        config: Any,
        step: int,
        loss: float,
        filename: Optional[str] = None,
    ):
        """Save checkpoint in pickle format (compact, faster)."""
        if filename is None:
            filename = f"checkpoint_step_{step}.pkl"

        serializable_state = self._convert_to_serializable(state_dict)

        checkpoint = {
            "step": step,
            "loss": loss,
            "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
            "state_dict": serializable_state,
        }

        path = self.checkpoint_dir / filename
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        return path

    def load_json(self, filename: str) -> Dict:
        """Load checkpoint from JSON file."""
        path = self.checkpoint_dir / filename
        with open(path, "r") as f:
            return json.load(f)

    def load_pickle(self, filename: str) -> Dict:
        """Load checkpoint from pickle file."""
        path = self.checkpoint_dir / filename
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_best(
        self, state_dict: Dict[str, List], config: Any, step: int, loss: float, format: str = "pkl"
    ):
        """Save as best model (overwrites previous best)."""
        filename = f"best_model.{format}"
        if format == "json":
            return self.save_json(state_dict, config, step, loss, filename)
        else:
            return self.save_pickle(state_dict, config, step, loss, filename)

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        checkpoints = []
        for ext in ["*.json", "*.pkl"]:
            checkpoints.extend([f.name for f in self.checkpoint_dir.glob(ext)])
        return sorted(checkpoints)

    def get_latest(self) -> Optional[str]:
        """Get the most recent checkpoint filename."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Sort by step number
        def extract_step(name):
            try:
                return int(name.split("_")[-1].split(".")[0])
            except Exception:
                return 0

        return max(checkpoints, key=extract_step)
