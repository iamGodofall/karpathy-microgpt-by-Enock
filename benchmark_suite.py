"""
Comprehensive benchmarking suite for microgpt ecosystem.
Measures performance, memory usage, and quality metrics.
"""

import time
import random
import sys
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, asdict
import json


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    metric: str
    value: float
    unit: str
    details: Dict[str, Any] = None


class BenchmarkSuite:
    """Run comprehensive benchmarks."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark(self, name: str, metric: str, unit: str, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run a benchmark and record result."""
        print(f"  Benchmarking {name}...", end=" ")
        
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            
            benchmark = BenchmarkResult(
                name=name,
                metric=metric,
                value=result if isinstance(result, (int, float)) else duration,
                unit=unit,
                details={"duration": duration} if not isinstance(result, dict) else result
            )
            print(f"✓ {benchmark.value:.4f} {unit}")
            self.results.append(benchmark)
            return benchmark
        except Exception as e:
            print(f"✗ Error: {e}")
            benchmark = BenchmarkResult(
                name=name,
                metric=metric,
                value=0,
                unit=unit,
                details={"error": str(e)}
            )
            self.results.append(benchmark)
            return benchmark
    
    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("=" * 70)
        print("microgpt Ecosystem Benchmark Suite")
        print("=" * 70)
        
        self._benchmark_core()
        self._benchmark_model()
        self._benchmark_training()
        self._benchmark_generation()
        self._benchmark_integrations()
        
        return self._generate_report()
    
    def _benchmark_core(self):
        """Benchmark core operations."""
        print("\n--- Core Operations ---")
        
        from microgpt import Value, softmax, rmsnorm, linear
        
        # Benchmark Value operations
        def bench_value_ops():
            a = Value(2.0)
            b = Value(3.0)
            for _ in range(1000):
                c = a * b + a
                c.backward()
            return 1000
        
        self.benchmark("Value ops (1000x)", "throughput", "ops/sec", bench_value_ops)
        
        # Benchmark softmax
        def bench_softmax():
            logits = [Value(random.random()) for _ in range(100)]
            for _ in range(100):
                softmax(logits)
            return 100
        
        self.benchmark("Softmax (100x, 100-dim)", "throughput", "ops/sec", bench_softmax)
        
        # Benchmark rmsnorm
        def bench_rmsnorm():
            x = [Value(random.random()) for _ in range(512)]
            for _ in range(100):
                rmsnorm(x)
            return 100
        
        self.benchmark("RMSNorm (100x, 512-dim)", "throughput", "ops/sec", bench_rmsnorm)
    
    def _benchmark_model(self):
        """Benchmark model operations."""
        print("\n--- Model Operations ---")
        
        from model import GPT, GPTConfig
        
        config = GPTConfig(vocab_size=100, n_embd=64, n_layer=2, n_head=4, block_size=16)
        model = GPT(config)
        
        # Forward pass
        def bench_forward():
            tokens = [random.randint(0, 99) for _ in range(10)]
            for _ in range(10):
                model.forward(tokens)
            return 10
        
        self.benchmark("Forward pass (10x, 10 tokens)", "throughput", "passes/sec", bench_forward)
        
        # Parameter count
        params = len(model.parameters())
        self.benchmark("Parameter count", "size", "parameters", lambda: params)
    
    def _benchmark_training(self):
        """Benchmark training speed."""
        print("\n--- Training ---")
        
        from model import GPT, GPTConfig
        from data import CharTokenizer
        
        config = GPTConfig(vocab_size=128, n_embd=32, n_layer=1, block_size=8)
        model = GPT(config)
        tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
        
        def bench_train_step():
            tokens = tokenizer.encode("hello")
            targets = tokenizer.encode("world")
            for _ in range(10):
                model.train_step(tokens, targets)
            return 10
        
        self.benchmark("Training step (10x)", "throughput", "steps/sec", bench_train_step)
    
    def _benchmark_generation(self):
        """Benchmark generation speed."""
        print("\n--- Generation ---")
        
        from model import GPT, GPTConfig
        from data import CharTokenizer
        
        config = GPTConfig(vocab_size=128, n_embd=32, n_layer=1)
        model = GPT(config)
        tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
        
        def bench_generate():
            result = model.generate("h", tokenizer, max_length=20)
            return len(result)
        
        self.benchmark("Generation (20 tokens)", "speed", "tokens/sec", bench_generate)
    
    def _benchmark_integrations(self):
        """Benchmark enhanced integrations."""
        print("\n--- Enhanced Integrations ---")
        
        # HRM benchmark
        from hrm_enhanced import EnhancedHierarchicalReasoningModel, EnhancedHRMConfig
        
        config = EnhancedHRMConfig(
            vocab_size=100,
            hidden_size=32,
            H_layers=1,
            L_layers=1,
            H_cycles=2,
            L_cycles=2
        )
        hrm = EnhancedHierarchicalReasoningModel(config)
        
        def bench_hrm():
            tokens = [1, 2, 3, 4, 5]
            result = hrm.forward(tokens, training=False)
            return result["steps"]
        
        self.benchmark("HRM forward", "latency", "steps", bench_hrm)
        
        # OpenClaw benchmark
        from openclaw_enhanced import EnhancedOpenClaw
        
        oc = EnhancedOpenClaw(storage_dir=".bench_microgpt")
        
        def bench_session():
            session = oc.create_session()
            for i in range(10):
                session.add_message("user", f"message {i}")
            return len(session.messages)
        
        self.benchmark("Session operations (10 msgs)", "throughput", "ops/sec", bench_session)
        
        # Unified benchmark
        from unified_integration import UnifiedAI, UnifiedConfig
        
        u_config = UnifiedConfig(vocab_size=100, hidden_size=32, hrm_H_layers=1, hrm_L_layers=1)
        ai = UnifiedAI(u_config)
        
        def bench_unified():
            result = ai.chat("test", use_reasoning=False)
            return 1
        
        self.benchmark("Unified chat", "latency", "requests/sec", bench_unified)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate benchmark report."""
        print("\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        
        # Group by category
        categories = {}
        for r in self.results:
            cat = r.name.split()[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        for cat, results in categories.items():
            print(f"\n{cat}:")
            for r in results:
                print(f"  {r.name}: {r.value:.4f} {r.unit}")
        
        # Save report
        report_path = "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                "benchmarks": [asdict(r) for r in self.results],
                "summary": {
                    "total": len(self.results),
                    "categories": list(categories.keys())
                }
            }, f, indent=2)
        
        print(f"\nReport saved to {report_path}")
        print("=" * 70)
        
        return {
            "total": len(self.results),
            "categories": categories,
            "report_path": report_path,
        }


def run_benchmarks():
    """Run all benchmarks."""
    suite = BenchmarkSuite()
    return suite.run_all()


if __name__ == "__main__":
    run_benchmarks()
