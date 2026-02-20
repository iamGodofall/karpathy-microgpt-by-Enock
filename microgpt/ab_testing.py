"""
A/B testing framework for microgpt.
Compare model variants in production.
"""

import json
import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ABTestResult:
    """Result of an A/B test."""

    test_id: str
    model_a: str
    model_b: str
    traffic_split: float  # 0.5 = 50/50
    total_requests: int
    results: List[Dict[str, Any]]
    winner: Optional[str]
    confidence: float


class ABTestManager:
    """Manage A/B tests between model variants."""

    def __init__(self, storage_dir: str = "ab_tests"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.active_tests: Dict[str, ABTestResult] = {}
        self._load_tests()

    def _load_tests(self):
        """Load existing tests."""
        for test_file in self.storage_dir.glob("*.json"):
            with open(test_file) as f:
                data = json.load(f)
                self.active_tests[data["test_id"]] = ABTestResult(**data)

    def create_test(
        self, model_a: str, model_b: str, traffic_split: float = 0.5, test_id: str = None
    ) -> str:
        """Create a new A/B test."""
        if test_id is None:
            test_id = f"ab_{int(time.time())}"

        test = ABTestResult(
            test_id=test_id,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            total_requests=0,
            results=[],
            winner=None,
            confidence=0.0,
        )

        self.active_tests[test_id] = test
        self._save_test(test)

        print(f"Created A/B test: {test_id}")
        print(f"  Model A: {model_a} ({traffic_split * 100:.0f}%)")
        print(f"  Model B: {model_b} ({(1 - traffic_split) * 100:.0f}%)")

        return test_id

    def route_request(self, test_id: str, user_id: str = None) -> str:
        """Route request to model A or B based on traffic split."""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        # Deterministic routing based on user_id if provided
        if user_id:
            random.seed(hash(user_id) % (2**32))

        # Route
        if random.random() < test.traffic_split:
            return test.model_a
        else:
            return test.model_b

    def record_result(self, test_id: str, model_used: str, metrics: Dict[str, float]):
        """Record result from a test request."""
        if test_id not in self.active_tests:
            return

        test = self.active_tests[test_id]
        test.total_requests += 1

        result = {"timestamp": time.time(), "model": model_used, "metrics": metrics}
        test.results.append(result)

        self._save_test(test)

    def _save_test(self, test: ABTestResult):
        """Save test to disk."""
        path = self.storage_dir / f"{test.test_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(test), f, indent=2, default=str)

    def analyze_test(self, test_id: str, metric: str = "loss") -> Dict[str, Any]:
        """Analyze A/B test results."""
        if test_id not in self.active_tests:
            return {}

        test = self.active_tests[test_id]

        # Aggregate metrics by model
        model_a_metrics = [
            r["metrics"].get(metric, 0) for r in test.results if r["model"] == test.model_a
        ]
        model_b_metrics = [
            r["metrics"].get(metric, 0) for r in test.results if r["model"] == test.model_b
        ]

        if not model_a_metrics or not model_b_metrics:
            return {"error": "Insufficient data"}

        # Calculate statistics
        stats = {
            "model_a": {
                "mean": sum(model_a_metrics) / len(model_a_metrics),
                "count": len(model_a_metrics),
            },
            "model_b": {
                "mean": sum(model_b_metrics) / len(model_b_metrics),
                "count": len(model_b_metrics),
            },
            "total_requests": test.total_requests,
        }

        # Determine winner (lower is better for loss)
        if stats["model_a"]["mean"] < stats["model_b"]["mean"]:
            stats["winner"] = test.model_a
            stats["improvement"] = (stats["model_b"]["mean"] - stats["model_a"]["mean"]) / stats[
                "model_b"
            ]["mean"]
        else:
            stats["winner"] = test.model_b
            stats["improvement"] = (stats["model_a"]["mean"] - stats["model_b"]["mean"]) / stats[
                "model_a"
            ]["mean"]

        return stats

    def conclude_test(self, test_id: str, metric: str = "loss") -> Optional[str]:
        """Conclude test and declare winner."""
        stats = self.analyze_test(test_id, metric)

        if "error" in stats:
            print(f"Cannot conclude test: {stats['error']}")
            return None

        test = self.active_tests[test_id]
        test.winner = stats["winner"]
        test.confidence = (
            min(stats["model_a"]["count"], stats["model_b"]["count"]) / 100
        )  # Simplified

        self._save_test(test)

        print(f"Test {test_id} concluded")
        print(f"Winner: {test.winner}")
        print(f"Improvement: {stats['improvement']*100:.2f}%")

        return test.winner

    def list_tests(self) -> List[Dict[str, Any]]:
        """List all tests."""
        return [
            {
                "test_id": t.test_id,
                "model_a": t.model_a,
                "model_b": t.model_b,
                "total_requests": t.total_requests,
                "status": "concluded" if t.winner else "running",
            }
            for t in self.active_tests.values()
        ]


class MultiArmedBandit:
    """Multi-armed bandit for dynamic A/B testing."""

    def __init__(self, models: List[str], epsilon: float = 0.1):
        self.models = models
        self.epsilon = epsilon
        self.counts = {m: 0 for m in models}
        self.values = {m: 0.0 for m in models}

    def select(self) -> str:
        """Select model using epsilon-greedy."""
        if random.random() < self.epsilon:
            # Explore
            return random.choice(self.models)
        else:
            # Exploit
            return max(self.models, key=lambda m: self.values[m])

    def update(self, model: str, reward: float):
        """Update model value estimate."""
        self.counts[model] += 1
        n = self.counts[model]
        value = self.values[model]
        self.values[model] = ((n - 1) / n) * value + (1 / n) * reward

    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            "counts": self.counts,
            "values": self.values,
            "best_model": max(self.models, key=lambda m: self.values[m]),
        }


# Example usage
if __name__ == "__main__":
    # A/B Testing
    ab = ABTestManager()

    # Create test
    test_id = ab.create_test("model_v1", "model_v2", traffic_split=0.5)

    # Simulate requests
    for i in range(100):
        model = ab.route_request(test_id, user_id=f"user_{i}")

        # Simulate metric
        metrics = {"loss": random.random()}
        ab.record_result(test_id, model, metrics)

    # Analyze
    stats = ab.analyze_test(test_id)
    print("\nA/B Test Results:")
    print(f"  Model A (v1): {stats['model_a']['mean']:.4f}")
    print(f"  Model B (v2): {stats['model_b']['mean']:.4f}")
    print(f"  Winner: {stats['winner']}")

    # Conclude
    winner = ab.conclude_test(test_id)

    # Multi-armed bandit
    print("\nMulti-Armed Bandit:")
    bandit = MultiArmedBandit(["model_a", "model_b", "model_c"])

    for _ in range(100):
        model = bandit.select()
        # Simulate reward (higher is better)
        reward = random.random()
        bandit.update(model, reward)

    print(f"Stats: {bandit.get_stats()}")
