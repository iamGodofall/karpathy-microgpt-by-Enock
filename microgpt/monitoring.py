"""
Monitoring and observability for microgpt ecosystem.
Real-time metrics, alerting, and health checks.
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path


@dataclass
class Metric:
    """Single metric data point."""

    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, deque] = {}
        self.max_history = max_history
        self._lock = threading.Lock()

    def record(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.max_history)

            metric = Metric(name=name, value=value, timestamp=time.time(), tags=tags or {})
            self.metrics[name].append(metric)

    def get_metric(self, name: str, last_n: int = None) -> List[Metric]:
        """Get metric history."""
        with self._lock:
            metrics = self.metrics.get(name, deque())
            if last_n:
                return list(metrics)[-last_n:]
            return list(metrics)

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = [m.value for m in self.get_metric(name)]
        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last": values[-1],
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}

    def export_json(self, path: str):
        """Export metrics to JSON."""
        data = {
            name: [{"value": m.value, "timestamp": m.timestamp, "tags": m.tags} for m in metrics]
            for name, metrics in self.metrics.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class HealthChecker:
    """Health check system."""

    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.status: Dict[str, bool] = {}
        self.last_check: Dict[str, float] = {}

    def register(self, name: str, check_func: Callable[[], bool]):
        """Register a health check."""
        self.checks[name] = check_func

    def check(self, name: str) -> bool:
        """Run a specific health check."""
        if name not in self.checks:
            return False

        try:
            result = self.checks[name]()
            self.status[name] = result
            self.last_check[name] = time.time()
            return result
        except Exception as e:
            self.status[name] = False
            self.last_check[name] = time.time()
            return False

    def check_all(self) -> Dict[str, bool]:
        """Run all health checks."""
        return {name: self.check(name) for name in self.checks.keys()}

    def is_healthy(self) -> bool:
        """Check if all systems are healthy."""
        return all(self.check_all().values())


class AlertManager:
    """Alert management system."""

    def __init__(self):
        self.alerts: List[Dict] = []
        self.thresholds: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def set_threshold(self, metric: str, min_val: float = None, max_val: float = None):
        """Set alert threshold for a metric."""
        self.thresholds[metric] = {"min": min_val, "max": max_val}

    def check_metric(self, metric: str, value: float) -> Optional[Dict]:
        """Check if metric triggers alert."""
        if metric not in self.thresholds:
            return None

        threshold = self.thresholds[metric]
        if threshold["min"] is not None and value < threshold["min"]:
            return {
                "metric": metric,
                "value": value,
                "threshold": threshold["min"],
                "type": "below_minimum",
                "severity": "warning",
                "timestamp": time.time(),
            }

        if threshold["max"] is not None and value > threshold["max"]:
            return {
                "metric": metric,
                "value": value,
                "threshold": threshold["max"],
                "type": "above_maximum",
                "severity": "critical",
                "timestamp": time.time(),
            }

        return None

    def add_alert(self, alert: Dict):
        """Add an alert."""
        with self._lock:
            self.alerts.append(alert)

    def get_alerts(self, severity: str = None) -> List[Dict]:
        """Get alerts, optionally filtered by severity."""
        with self._lock:
            if severity:
                return [a for a in self.alerts if a.get("severity") == severity]
            return list(self.alerts)

    def clear_alerts(self):
        """Clear all alerts."""
        with self._lock:
            self.alerts = []


class ModelMonitor:
    """Monitor model performance and health."""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.health = HealthChecker()
        self.alerts = AlertManager()
        self._running = False
        self._monitor_thread = None

    def start(self, interval: float = 60.0):
        """Start monitoring."""
        self._running = True

        def monitor_loop():
            while self._running:
                # Run health checks
                self.health.check_all()

                # Check metrics against thresholds
                for name, stats in self.metrics.all_stats().items():
                    if "last" in stats:
                        alert = self.alerts.check_metric(name, stats["last"])
                        if alert:
                            self.alerts.add_alert(alert)

                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        print(f"Monitoring started (interval: {interval}s)")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("Monitoring stopped")

    def record_training_step(self, step: int, loss: float, lr: float, grad_norm: float = None):
        """Record training metrics."""
        self.metrics.record("train_loss", loss, {"step": str(step)})
        self.metrics.record("learning_rate", lr, {"step": str(step)})
        if grad_norm:
            self.metrics.record("grad_norm", grad_norm, {"step": str(step)})

    def record_generation(self, tokens_generated: int, time_taken: float):
        """Record generation metrics."""
        tokens_per_sec = tokens_generated / time_taken if time_taken > 0 else 0
        self.metrics.record("tokens_per_sec", tokens_per_sec)
        self.metrics.record("generation_time", time_taken)

    def record_inference(self, input_tokens: int, output_tokens: int, time_taken: float):
        """Record inference metrics."""
        self.metrics.record("inference_time", time_taken)
        self.metrics.record("input_tokens", input_tokens)
        self.metrics.record("output_tokens", output_tokens)

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard."""
        return {
            "health": self.health.status,
            "metrics": self.metrics.all_stats(),
            "alerts": self.alerts.get_alerts(),
            "timestamp": time.time(),
        }

    def save_report(self, path: str = "monitoring_report.json"):
        """Save monitoring report."""
        data = self.get_dashboard_data()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Monitoring report saved to {path}")


# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = ModelMonitor()

    # Set up health checks
    def check_memory():
        import psutil

        mem = psutil.virtual_memory()
        return mem.percent < 90

    def check_disk():
        import shutil

        stat = shutil.disk_usage(".")
        return stat.free > 1e9  # 1GB

    monitor.health.register("memory", check_memory)
    monitor.health.register("disk", check_disk)

    # Set alert thresholds
    monitor.alerts.set_threshold("train_loss", max_val=5.0)
    monitor.alerts.set_threshold("tokens_per_sec", min_val=1.0)

    # Start monitoring
    monitor.start(interval=5)

    # Simulate some activity
    import random

    for i in range(10):
        monitor.record_training_step(i, random.random() * 3, 0.01)
        time.sleep(0.5)

    # Get report
    monitor.save_report()

    # Stop
    monitor.stop()

    print("\nHealth status:", monitor.health.is_healthy())
    print("Metrics:", monitor.metrics.all_stats())
