"""
Experiment tracking for microgpt.
Track experiments, compare results, manage runs.
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class Experiment:
    """Single experiment record."""

    id: str
    name: str
    config: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed
    metrics: Dict[str, List[float]] = None
    artifacts: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.artifacts is None:
            self.artifacts = []
        if self.tags is None:
            self.tags = []


class ExperimentTracker:
    """Track and manage experiments."""

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_experiment: Optional[Experiment] = None
        self.experiments: List[Experiment] = []
        self._load_experiments()

    def _load_experiments(self):
        """Load existing experiments."""
        for exp_file in self.experiment_dir.glob("*.json"):
            with open(exp_file) as f:
                data = json.load(f)
                exp = Experiment(**data)
                self.experiments.append(exp)

    def start_experiment(
        self, name: str, config: Dict[str, Any], tags: List[str] = None
    ) -> Experiment:
        """Start a new experiment."""
        exp_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:8]

        exp = Experiment(
            id=exp_id, name=name, config=config, start_time=time.time(), tags=tags or []
        )

        self.current_experiment = exp
        self.experiments.append(exp)
        self._save_experiment(exp)

        print(f"Started experiment: {name} (ID: {exp_id})")
        return exp

    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric for current experiment."""
        if self.current_experiment is None:
            raise RuntimeError("No active experiment")

        if name not in self.current_experiment.metrics:
            self.current_experiment.metrics[name] = []

        self.current_experiment.metrics[name].append(
            {"value": value, "step": step, "timestamp": time.time()}
        )

        self._save_experiment(self.current_experiment)

    def log_artifact(self, path: str):
        """Log an artifact (file path)."""
        if self.current_experiment is None:
            raise RuntimeError("No active experiment")

        self.current_experiment.artifacts.append(path)
        self._save_experiment(self.current_experiment)

    def end_experiment(self, status: str = "completed"):
        """End current experiment."""
        if self.current_experiment is None:
            return

        self.current_experiment.end_time = time.time()
        self.current_experiment.status = status
        self._save_experiment(self.current_experiment)

        duration = self.current_experiment.end_time - self.current_experiment.start_time
        print(f"Experiment {self.current_experiment.id} {status} in {duration:.1f}s")
        self.current_experiment = None

    def _save_experiment(self, exp: Experiment):
        """Save experiment to disk."""
        path = self.experiment_dir / f"{exp.id}.json"
        with open(path, "w") as f:
            json.dump(asdict(exp), f, indent=2, default=str)

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        for exp in self.experiments:
            if exp.id == exp_id:
                return exp
        return None

    def list_experiments(self, status: str = None, tag: str = None) -> List[Experiment]:
        """List experiments with optional filtering."""
        result = self.experiments

        if status:
            result = [e for e in result if e.status == status]

        if tag:
            result = [e for e in result if tag in e.tags]

        return sorted(result, key=lambda e: e.start_time, reverse=True)

    def compare_experiments(self, exp_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiments = [self.get_experiment(eid) for eid in exp_ids]
        experiments = [e for e in experiments if e is not None]

        if not experiments:
            return {}

        # Compare configs
        config_keys = set()
        for e in experiments:
            config_keys.update(e.config.keys())

        config_diff = {}
        for key in config_keys:
            values = [e.config.get(key) for e in experiments]
            if len(set(str(v) for v in values)) > 1:
                config_diff[key] = {e.id: e.config.get(key) for e in experiments}

        # Compare metrics
        metric_comparison = {}
        all_metrics = set()
        for e in experiments:
            all_metrics.update(e.metrics.keys())

        for metric in all_metrics:
            metric_comparison[metric] = {
                e.id: {
                    "final": e.metrics[metric][-1]["value"] if e.metrics.get(metric) else None,
                    "best": min((m["value"] for m in e.metrics.get(metric, [])), default=None),
                    "history": [m["value"] for m in e.metrics.get(metric, [])],
                }
                for e in experiments
            }

        return {
            "config_differences": config_diff,
            "metric_comparison": metric_comparison,
            "experiments": [{"id": e.id, "name": e.name, "status": e.status} for e in experiments],
        }

    def get_best_experiment(self, metric: str = "loss", mode: str = "min") -> Optional[Experiment]:
        """Get best experiment by metric."""
        completed = [e for e in self.experiments if e.status == "completed"]

        if not completed:
            return None

        if mode == "min":
            return min(
                completed, key=lambda e: e.metrics.get(metric, [{}])[-1].get("value", float("inf"))
            )
        else:
            return max(
                completed, key=lambda e: e.metrics.get(metric, [{}])[-1].get("value", float("-inf"))
            )

    def generate_report(self, exp_id: str = None) -> str:
        """Generate markdown report for experiment(s)."""
        if exp_id:
            exp = self.get_experiment(exp_id)
            experiments = [exp] if exp else []
        else:
            experiments = self.experiments[:10]  # Last 10

        lines = ["# Experiment Report\n"]

        for exp in experiments:
            lines.append(f"## {exp.name} (ID: {exp.id})")
            lines.append(f"- **Status**: {exp.status}")
            lines.append(f"- **Started**: {datetime.fromtimestamp(exp.start_time)}")
            if exp.end_time:
                duration = exp.end_time - exp.start_time
                lines.append(f"- **Duration**: {duration:.1f}s")
            lines.append(f"- **Tags**: {', '.join(exp.tags) if exp.tags else 'None'}")

            lines.append("\n### Configuration")
            for key, value in exp.config.items():
                lines.append(f"- {key}: {value}")

            lines.append("\n### Metrics")
            for name, values in exp.metrics.items():
                if values:
                    final = values[-1]["value"]
                    best = min(v["value"] for v in values)
                    lines.append(f"- {name}: {final:.4f} (best: {best:.4f})")

            lines.append("\n---\n")

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    tracker = ExperimentTracker()

    # Start experiment
    config = {"model.n_embd": 64, "model.n_layer": 2, "training.lr": 0.01, "training.batch_size": 4}

    exp = tracker.start_experiment("test_run", config, tags=["test", "small"])

    # Simulate training
    for step in range(10):
        loss = 2.0 / (step + 1)  # Decreasing loss
        tracker.log_metric("loss", loss, step=step)
        tracker.log_metric("accuracy", 0.5 + step * 0.05, step=step)
        time.sleep(0.1)

    # Log artifact
    tracker.log_artifact("model_checkpoint.pkl")

    # End experiment
    tracker.end_experiment()

    # Generate report
    report = tracker.generate_report(exp.id)
    print(report)

    # List experiments
    print("\nAll experiments:")
    for e in tracker.list_experiments():
        print(f"  {e.id}: {e.name} ({e.status})")
