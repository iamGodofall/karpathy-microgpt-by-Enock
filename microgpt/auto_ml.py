"""
AutoML for microgpt.
Automated model selection, architecture search, and optimization.
"""

import random
import time
import math
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict


@dataclass
class AutoMLConfig:
    """Configuration for AutoML."""

    max_trials: int = 10
    timeout_seconds: int = 3600
    metric: str = "loss"
    mode: str = "min"  # min or max
    search_strategy: str = "random"  # random, bayesian, evolutionary


class NeuralArchitectureSearch:
    """Neural Architecture Search for microgpt."""

    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.search_space = {
            "n_layer": [1, 2, 4, 8],
            "n_embd": [32, 64, 128, 256],
            "n_head": [2, 4, 8],
            "block_size": [16, 32, 64, 128],
        }
        self.results: List[Dict] = []

    def search(self, train_data: Any, eval_func: Callable) -> Dict[str, Any]:
        """Search for optimal architecture."""
        print(f"Starting Neural Architecture Search ({self.config.search_strategy})...")

        best_config = None
        best_score = float("inf") if self.config.mode == "min" else float("-inf")

        for trial in range(self.config.max_trials):
            # Sample architecture
            arch = {k: random.choice(v) for k, v in self.search_space.items()}

            # Ensure n_embd divisible by n_head
            arch["n_head"] = min(arch["n_head"], arch["n_embd"] // 16)

            print(f"\nTrial {trial+1}/{self.config.max_trials}: {arch}")

            # Train and evaluate
            start = time.time()
            score = self._evaluate_architecture(arch, train_data, eval_func)
            duration = time.time() - start

            result = {"trial": trial, "architecture": arch, "score": score, "duration": duration}
            self.results.append(result)

            # Update best
            is_better = (self.config.mode == "min" and score < best_score) or (
                self.config.mode == "max" and score > best_score
            )

            if is_better:
                best_score = score
                best_config = arch
                print(f"  New best: {score:.4f}")
            else:
                print(f"  Score: {score:.4f}")

            # Check timeout
            if (
                time.time() - self.results[0].get("start_time", time.time())
                > self.config.timeout_seconds
            ):
                print("Timeout reached")
                break

        return {
            "best_config": best_config,
            "best_score": best_score,
            "all_results": self.results,
            "total_trials": len(self.results),
        }

    def _evaluate_architecture(self, arch: Dict, train_data: Any, eval_func: Callable) -> float:
        """Train and evaluate an architecture."""
        from model import GPT, GPTConfig
        from trainer import Trainer

        config = GPTConfig(
            vocab_size=100,
            n_embd=arch["n_embd"],
            n_layer=arch["n_layer"],
            n_head=arch["n_head"],
            block_size=arch["block_size"],
        )

        model = GPT(config)
        trainer = Trainer(model, num_steps=50)  # Quick training

        # Quick training
        for _ in range(10):
            batch = random.sample(train_data, min(4, len(train_data)))
            for item in batch:
                if isinstance(item, tuple):
                    tokens, targets = item
                else:
                    tokens = item
                    targets = item
                trainer.train_step(tokens, targets)

        return eval_func(model)


class AutoFeatureEngineering:
    """Automated feature engineering."""

    def __init__(self):
        self.transformations = [
            "normalize",
            "center",
            "log",
            "sqrt",
            "square",
        ]

    def find_best_transforms(
        self, features: List[List[float]], targets: List[float], model_builder: Callable
    ) -> List[str]:
        """Find best feature transformations."""
        from sklearn.feature_selection import mutual_info_regression

        best_transforms = []
        best_score = float("inf")

        # Try different combinations
        for transform in self.transformations:
            transformed = self._apply_transform(features, transform)
            score = self._evaluate_features(transformed, targets, model_builder)

            if score < best_score:
                best_score = score
                best_transforms = [transform]

        return best_transforms

    def _apply_transform(self, features: List[List[float]], transform: str) -> List[List[float]]:
        """Apply transformation to features."""
        if transform == "normalize":
            return [[x / (sum(f) + 1e-8) for x in f] for f in features]
        elif transform == "center":
            return [[x - sum(f) / len(f) for x in f] for f in features]
        elif transform == "log":
            return [[math.log(x + 1) for x in f] for f in features]
        elif transform == "sqrt":
            return [[x**0.5 for x in f] for f in features]
        elif transform == "square":
            return [[x**2 for x in f] for f in features]
        return features

    def _evaluate_features(
        self, features: List[List[float]], targets: List[float], model_builder: Callable
    ) -> float:
        """Evaluate feature quality."""
        # Simple evaluation - lower is better
        return random.random()  # Placeholder


class AutoMLPipeline:
    """Complete AutoML pipeline."""

    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.nas = NeuralArchitectureSearch(config)
        self.fe = AutoFeatureEngineering()

    def run(self, train_data: Any, eval_func: Callable) -> Dict[str, Any]:
        """Run complete AutoML pipeline."""
        print("=" * 70)
        print("AutoML Pipeline")
        print("=" * 70)

        # Architecture search
        print("\n1. Neural Architecture Search")
        arch_result = self.nas.search(train_data, eval_func)

        # Feature engineering (if applicable)
        print("\n2. Feature Engineering")
        # fe_result = self.fe.find_best_transforms(...)

        return {
            "architecture": arch_result,
            "best_config": arch_result["best_config"],
            "best_score": arch_result["best_score"],
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    train_data = [
        ([1, 2, 3, 4], [2, 3, 4, 5]),
        ([2, 3, 4, 5], [3, 4, 5, 6]),
    ] * 10

    def eval_func(model):
        return random.random()  # Placeholder

    # Run AutoML
    config = AutoMLConfig(max_trials=3)
    automl = AutoMLPipeline(config)
    result = automl.run(train_data, eval_func)

    print(f"\nBest architecture: {result['best_config']}")
    print(f"Best score: {result['best_score']:.4f}")
