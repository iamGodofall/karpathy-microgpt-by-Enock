"""
Logging and metrics tracking for training.
Supports console output and file logging with structured metrics.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class Metrics:
    """Training metrics at a specific step."""

    step: int
    loss: float
    perplexity: float
    learning_rate: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TrainingLogger:
    """Logs training metrics to console and file."""

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        if experiment_name is None:
            experiment_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"
        self.metrics_history: List[Metrics] = []
        self.start_time = time.time()

        # Write header
        with open(self.log_file, "w") as f:
            f.write(json.dumps({"event": "start", "timestamp": datetime.now().isoformat()}) + "\n")

    def log(self, metrics: Metrics):
        """Log metrics to file and console."""
        self.metrics_history.append(metrics)

        # Console output
        elapsed = time.time() - self.start_time
        print(
            f"step {metrics.step:5d} | loss {metrics.loss:.4f} | "
            f"ppl {metrics.perplexity:.2f} | lr {metrics.learning_rate:.6f} | "
            f"time {elapsed:.1f}s",
            end="\r",
        )

        # File output (JSON lines)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

    def log_event(self, event: str, data: Optional[Dict] = None):
        """Log a non-metric event (e.g., save, eval)."""
        entry = {"event": event, "timestamp": datetime.now().isoformat()}
        if data:
            entry.update(data)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_summary(self) -> Dict:
        """Get summary statistics of training."""
        if not self.metrics_history:
            return {}

        losses = [m.loss for m in self.metrics_history]
        perplexities = [m.perplexity for m in self.metrics_history]

        return {
            "total_steps": len(self.metrics_history),
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "avg_loss": sum(losses) / len(losses),
            "final_perplexity": perplexities[-1],
            "best_perplexity": min(perplexities),
            "total_time": time.time() - self.start_time,
        }

    def save_summary(self):
        """Save training summary to file."""
        summary = self.get_summary()
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nTraining summary saved to {summary_file}")
        return summary


class ProgressTracker:
    """Track training progress with progress bar."""

    def __init__(self, total_steps: int, bar_length: int = 40):
        self.total_steps = total_steps
        self.bar_length = bar_length
        self.start_time = time.time()

    def update(self, step: int, loss: float):
        """Update and display progress bar."""
        progress = step / self.total_steps
        filled = int(self.bar_length * progress)
        bar = "█" * filled + "░" * (self.bar_length - filled)

        elapsed = time.time() - self.start_time
        eta = elapsed / progress - elapsed if progress > 0 else 0

        print(
            f"\r[{bar}] {progress*100:.1f}% | Step {step}/{self.total_steps} | "
            f"Loss: {loss:.4f} | ETA: {eta:.1f}s",
            end="",
        )

    def finish(self):
        """Mark progress as complete."""
        print()  # New line after progress bar
