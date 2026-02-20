"""
Benchmarking and profiling utilities for microgpt.
Measure training speed, memory usage, and generation quality.
"""

import time
import random
from typing import Dict, List, Optional

from dataclasses import dataclass
from .model import GPT
from .data import CharTokenizer


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    total_time: float
    samples_per_second: float
    memory_estimate_mb: float
    perplexity: Optional[float] = None
    additional_metrics: Dict = None


class SpeedBenchmark:
    """Benchmark training and inference speed."""

    def __init__(self, model: GPT):
        self.model = model

    def benchmark_training(
        self, num_steps: int = 100, sequence_length: int = 16
    ) -> BenchmarkResult:
        """Benchmark training speed."""
        print(f"Benchmarking training for {num_steps} steps...")

        # Generate dummy data
        tokenizer = CharTokenizer()
        tokenizer.fit(["benchmark test data"])

        tokens = [tokenizer.bos_token] * sequence_length

        start_time = time.time()

        for step in range(num_steps):
            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]

            # Forward pass
            for pos in range(sequence_length - 1):
                _ = self.model.forward(tokens[pos], pos, keys, values)

            # Backward pass (simplified)

            # Would compute loss and backprop

        total_time = time.time() - start_time

        # Estimate memory (rough approximation)
        param_count = self.model.num_params()
        memory_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float

        return BenchmarkResult(
            name="training",
            total_time=total_time,
            samples_per_second=num_steps / total_time,
            memory_estimate_mb=memory_mb,
        )

    def benchmark_inference(self, num_samples: int = 100, max_length: int = 50) -> BenchmarkResult:
        """Benchmark generation speed."""
        print(f"Benchmarking inference for {num_samples} samples...")

        self.model.set_training(False)
        tokenizer = CharTokenizer()
        tokenizer.fit(["benchmark"])

        start_time = time.time()

        for _ in range(num_samples):
            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]

            token_id = tokenizer.bos_token
            for pos in range(max_length):
                logits = self.model.forward(token_id, pos, keys, values)
                probs = [logit.data for logit in logits]

                token_id = random.choices(range(len(probs)), weights=probs)[0]

                if token_id == tokenizer.bos_token:
                    break

        total_time = time.time() - start_time

        return BenchmarkResult(
            name="inference",
            total_time=total_time,
            samples_per_second=num_samples / total_time,
            memory_estimate_mb=0,
        )


class MemoryProfiler:
    """Profile memory usage during training."""

    def __init__(self, model: GPT):
        self.model = model
        self.peak_memory = 0
        self.memory_timeline: List[tuple] = []

    def estimate_memory(self) -> Dict[str, float]:
        """Estimate memory usage by component."""
        param_memory = self.model.num_params() * 4  # 4 bytes per float

        # Activations (rough estimate)
        activation_memory = (
            self.model.n_layer * self.model.block_size * self.model.n_embd * 4  # bytes per float
        )

        # Gradients (same size as parameters)

        gradient_memory = param_memory

        # Optimizer states (Adam: 2x parameters for m and v)
        optimizer_memory = 2 * param_memory

        total = param_memory + activation_memory + gradient_memory + optimizer_memory

        return {
            "parameters_mb": param_memory / (1024 * 1024),
            "activations_mb": activation_memory / (1024 * 1024),
            "gradients_mb": gradient_memory / (1024 * 1024),
            "optimizer_mb": optimizer_memory / (1024 * 1024),
            "total_mb": total / (1024 * 1024),
        }

    def print_memory_report(self):
        """Print detailed memory usage report."""
        memory = self.estimate_memory()

        print("\n" + "=" * 50)
        print("Memory Usage Report")
        print("=" * 50)
        print(f"Parameters:    {memory['parameters_mb']:8.2f} MB")
        print(f"Activations:   {memory['activations_mb']:8.2f} MB")
        print(f"Gradients:     {memory['gradients_mb']:8.2f} MB")
        print(f"Optimizer:     {memory['optimizer_mb']:8.2f} MB")
        print("-" * 50)
        print(f"Total:         {memory['total_mb']:8.2f} MB")
        print("=" * 50)


class QualityMetrics:
    """Evaluate generation quality metrics."""

    def __init__(self, model: GPT, tokenizer: CharTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def calculate_perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity on test set."""
        total_loss = 0.0
        total_tokens = 0

        self.model.set_training(False)

        for text in texts:
            tokens = (
                [self.tokenizer.bos_token]
                + [
                    self.tokenizer.char_to_idx.get(ch, self.tokenizer.bos_token)
                    for ch in text
                    if ch in self.tokenizer.char_to_idx
                ]
                + [self.tokenizer.bos_token]
            )

            if len(tokens) < 2:
                continue

            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]

            for pos in range(len(tokens) - 1):
                _ = self.model.forward(tokens[pos], pos, keys, values)

                # Simplified loss calculation
                # Would compute actual cross-entropy here
                total_loss += 2.0  # Placeholder
                total_tokens += 1

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")

        perplexity = 2.71828**avg_loss  # e^loss

        return perplexity

    def diversity_metrics(self, generated_texts: List[str]) -> Dict:
        """Measure diversity of generated text."""
        all_tokens = []
        for text in generated_texts:
            all_tokens.extend(list(text))

        # Vocabulary usage
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)

        # Repetition rate
        repeats = sum(1 for i in range(1, len(all_tokens)) if all_tokens[i] == all_tokens[i - 1])

        return {
            "vocab_usage_ratio": unique_tokens / total_tokens if total_tokens > 0 else 0,
            "repetition_rate": repeats / total_tokens if total_tokens > 0 else 0,
            "unique_tokens": unique_tokens,
            "total_tokens": total_tokens,
        }

    def benchmark_quality(self, num_samples: int = 100) -> Dict:
        """Run quality benchmarks."""
        print(f"Generating {num_samples} samples for quality evaluation...")

        self.model.set_training(False)

        generated = []
        for _ in range(num_samples):
            tokens = self.model.generate(self.tokenizer.bos_token, max_length=50, temperature=0.8)
            text = self.tokenizer.decode(tokens)
            generated.append(text)

        diversity = self.diversity_metrics(generated)

        return {
            "num_samples": num_samples,
            "avg_length": sum(len(t) for t in generated) / len(generated),
            **diversity,
        }


class ComparativeBenchmark:
    """Compare different model configurations."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def compare_configs(self, configs: List[Dict], num_steps: int = 50):
        """Benchmark multiple configurations."""
        for config in configs:
            print(f"\nBenchmarking: {config['name']}")

            model = GPT(
                vocab_size=config.get("vocab_size", 27),
                block_size=config.get("block_size", 16),
                n_layer=config.get("n_layer", 1),
                n_embd=config.get("n_embd", 16),
                n_head=config.get("n_head", 4),
            )

            benchmark = SpeedBenchmark(model)
            result = benchmark.benchmark_training(num_steps)
            result.name = config["name"]
            self.results.append(result)

        self._print_comparison()

    def _print_comparison(self):
        """Print comparison table."""
        print("\n" + "=" * 70)
        print(f"{'Configuration':<30} {'Time (s)':<12} {'Samples/s':<12} {'Memory (MB)':<12}")
        print("=" * 70)

        for result in self.results:
            print(
                f"{result.name:<30} {result.total_time:<12.2f} "
                f"{result.samples_per_second:<12.2f} "
                f"{result.memory_estimate_mb:<12.2f}"
            )

        print("=" * 70)


def run_full_benchmark(model: GPT, tokenizer: CharTokenizer):
    """Run comprehensive benchmark suite."""
    print("=" * 70)
    print("MICROGPT BENCHMARK SUITE")
    print("=" * 70)

    # Speed benchmarks
    speed = SpeedBenchmark(model)

    print("\n--- Training Speed ---")
    train_result = speed.benchmark_training(num_steps=100)
    print(f"Time: {train_result.total_time:.2f}s")
    print(f"Throughput: {train_result.samples_per_second:.2f} steps/s")

    print("\n--- Inference Speed ---")
    infer_result = speed.benchmark_inference(num_samples=100)
    print(f"Time: {infer_result.total_time:.2f}s")
    print(f"Throughput: {infer_result.samples_per_second:.2f} samples/s")

    # Memory profiling
    print("\n--- Memory Usage ---")
    profiler = MemoryProfiler(model)
    profiler.print_memory_report()

    # Quality metrics
    print("\n--- Quality Metrics ---")
    quality = QualityMetrics(model, tokenizer)
    metrics = quality.benchmark_quality(num_samples=50)
    print(f"Average length: {metrics['avg_length']:.1f}")
    print(f"Vocab usage: {metrics['vocab_usage_ratio']:.2%}")
    print(f"Repetition rate: {metrics['repetition_rate']:.2%}")

    print("\n" + "=" * 70)

    print("BENCHMARK COMPLETE")
    print("=" * 70)
