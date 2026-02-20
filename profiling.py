"""
Performance profiling and analysis tools.
Memory tracking, speed benchmarking, and optimization recommendations.
"""

import time
import math
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ProfileResult:
    """Result from profiling a function."""
    name: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.calls, 1)
    
    def __str__(self) -> str:
        return (f"{self.name:30} | "
                f"Calls: {self.calls:5} | "
                f"Avg: {self.avg_time*1000:8.3f}ms | "
                f"Min: {self.min_time*1000:8.3f}ms | "
                f"Max: {self.max_time*1000:8.3f}ms")


class Profiler:
    """
    Simple profiler for tracking performance.
    """
    
    def __init__(self):
        self.results: Dict[str, ProfileResult] = {}
        self.active: Dict[str, float] = {}
    
    def start(self, name: str):
        """Start timing a section."""
        self.active[name] = time.time()
    
    def end(self, name: str):
        """End timing a section."""
        if name not in self.active:
            return
        
        elapsed = time.time() - self.active[name]
        del self.active[name]
        
        if name not in self.results:
            self.results[name] = ProfileResult(name=name)
        
        result = self.results[name]
        result.calls += 1
        result.total_time += elapsed
        result.min_time = min(result.min_time, elapsed)
        result.max_time = max(result.max_time, elapsed)
    
    def profile(self, name: str):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                self.start(name)
                result = func(*args, **kwargs)
                self.end(name)
                return result
            return wrapper
        return decorator
    
    def report(self):
        """Print profiling report."""
        print("=" * 80)
        print("PERFORMANCE PROFILE")
        print("=" * 80)
        
        # Sort by total time
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: -x.total_time
        )
        
        for result in sorted_results:
            print(result)
        
        print("=" * 80)
        
        # Summary
        total_time = sum(r.total_time for r in self.results.values())
        print(f"Total profiled time: {total_time:.3f}s")
        print(f"Functions profiled: {len(self.results)}")
    
    def reset(self):
        """Reset all profiling data."""
        self.results.clear()
        self.active.clear()


class MemoryTracker:
    """
    Track memory usage during training/inference.
    """
    
    def __init__(self):
        self.samples: List[Dict] = []
        self.peak_memory = 0.0
    
    def sample(self, label: str = ""):
        """Record memory usage."""
        # Simplified - would use psutil in practice
        import sys
        
        # Estimate from object counts
        import gc
        gc.collect()
        
        # Rough estimate
        memory_mb = len(gc.get_objects()) * 0.001  # Very rough
        
        self.samples.append({
            'time': time.time(),
            'label': label,
            'memory_mb': memory_mb
        })
        
        self.peak_memory = max(self.peak_memory, memory_mb)
    
    def report(self):
        """Print memory report."""
        print("=" * 60)
        print("MEMORY USAGE")
        print("=" * 60)
        
        if not self.samples:
            print("No samples recorded")
            return
        
        print(f"Peak memory: {self.peak_memory:.2f} MB")
        print(f"Samples: {len(self.samples)}")
        
        # Show trend
        print("\nMemory trend:")
        for i, sample in enumerate(self.samples[::max(1, len(self.samples)//10)]):
            print(f"  {sample['label'][:30]:30} | {sample['memory_mb']:8.2f} MB")
        
        print("=" * 60)


class ModelAnalyzer:
    """
    Analyze model architecture and efficiency.
    """
    
    def __init__(self, model):
        self.model = model
    
    def analyze(self) -> Dict:
        """Full model analysis."""
        analysis = {
            'parameters': self._count_parameters(),
            'memory': self._estimate_memory(),
            'flops': self._estimate_flops(),
            'bottlenecks': self._find_bottlenecks(),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _count_parameters(self) -> Dict:
        """Count parameters by layer."""
        counts = {}
        total = 0
        
        for name, matrix in self.model.state_dict.items():
            count = len(matrix) * len(matrix[0]) if matrix else 0
            counts[name] = count
            total += count
        
        return {
            'total': total,
            'by_layer': counts,
            'trainable': total  # All trainable in base model
        }
    
    def _estimate_memory(self) -> Dict:
        """Estimate memory usage."""
        params = self._count_parameters()['total']
        
        # Parameter memory (4 bytes per float)
        param_memory = params * 4 / (1024 ** 2)  # MB
        
        # Activation memory (rough estimate)
        batch_size = 1
        seq_len = self.model.block_size
        hidden_dim = self.model.n_embd
        n_layers = self.model.n_layer
        
        activation_memory = (
            batch_size * seq_len * hidden_dim * n_layers * 4 / (1024 ** 2)
        )
        
        # KV cache
        kv_memory = (
            2 * batch_size * seq_len * hidden_dim * n_layers * 4 / (1024 ** 2)
        )
        
        return {
            'parameters_mb': param_memory,
            'activations_mb': activation_memory,
            'kv_cache_mb': kv_memory,
            'total_mb': param_memory + activation_memory + kv_memory,
            'recommendation': 'Use quantization' if param_memory > 1000 else None
        }
    
    def _estimate_flops(self) -> Dict:
        """Estimate FLOPs per forward pass."""
        n_layers = self.model.n_layer
        seq_len = self.model.block_size
        hidden_dim = self.model.n_embd
        vocab_size = self.model.vocab_size
        
        # Attention FLOPs
        attn_flops = 2 * seq_len * seq_len * hidden_dim
        
        # MLP FLOPs
        mlp_flops = 8 * seq_len * hidden_dim * hidden_dim  # 4x expansion
        
        # Total per layer
        layer_flops = attn_flops + mlp_flops
        
        # All layers
        total_flops = n_layers * layer_flops
        
        # Output projection
        output_flops = 2 * seq_len * hidden_dim * vocab_size
        
        return {
            'per_layer': layer_flops,
            'total_forward': total_flops + output_flops,
            'per_token': (total_flops + output_flops) / seq_len
        }
    
    def _find_bottlenecks(self) -> List[Dict]:
        """Identify potential bottlenecks."""
        bottlenecks = []
        
        # Check attention complexity
        if self.model.block_size > 2048:
            bottlenecks.append({
                'type': 'attention_quadratic',
                'severity': 'high',
                'description': f'Attention is O(n²) with n={self.model.block_size}',
                'fix': 'Use Flash Attention or linear attention'
            })
        
        # Check model size
        total_params = self._count_parameters()['total']
        if total_params > 1e9:
            bottlenecks.append({
                'type': 'model_size',
                'severity': 'medium',
                'description': f'Large model: {total_params/1e9:.1f}B params',
                'fix': 'Use quantization or LoRA for fine-tuning'
            })
        
        # Check memory
        memory = self._estimate_memory()['total_mb']
        if memory > 10000:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high',
                'description': f'High memory: {memory/1024:.1f} GB',
                'fix': 'Use gradient checkpointing or smaller batch size'
            })
        
        return bottlenecks
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate optimization recommendations."""
        recs = []
        
        # Based on bottlenecks
        for b in analysis['bottlenecks']:
            recs.append(f"[{b['severity'].upper()}] {b['fix']}")
        
        # Based on model size
        params = analysis['parameters']['total']
        if params > 100_000_000:
            recs.append("[MEDIUM] Consider using LoRA for efficient fine-tuning")
        
        # Based on sequence length
        if self.model.block_size > 1024:
            recs.append("[MEDIUM] Use ALiBi or RoPE for better long-range modeling")
        
        # General
        recs.append("[LOW] Enable gradient checkpointing for training")
        recs.append("[LOW] Use mixed precision training")
        
        return recs
    
    def print_report(self):
        """Print full analysis report."""
        analysis = self.analyze()
        
        print("=" * 70)
        print("MODEL ANALYSIS REPORT")
        print("=" * 70)
        
        # Parameters
        print("\n📊 PARAMETERS")
        print(f"  Total: {analysis['parameters']['total']:,}")
        print(f"  Trainable: {analysis['parameters']['trainable']:,}")
        
        # Memory
        print("\n💾 MEMORY ESTIMATE")
        for k, v in analysis['memory'].items():
            if v is not None:
                print(f"  {k}: {v:.2f} MB" if isinstance(v, float) else f"  {k}: {v}")
        
        # FLOPs
        print("\n⚡ FLOPs")
        print(f"  Per forward pass: {analysis['flops']['total_forward']:,.0f}")
        print(f"  Per token: {analysis['flops']['per_token']:,.0f}")
        
        # Bottlenecks
        if analysis['bottlenecks']:
            print("\n⚠️  BOTTLENECKS")
            for b in analysis['bottlenecks']:
                icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[b['severity']]
                print(f"  {icon} {b['type']}: {b['description']}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
        
        print("=" * 70)


class SpeedBenchmark:
    """
    Benchmark inference speed.
    """
    
    def __init__(self, model):
        self.model = model
    
    def benchmark_generation(self, 
                            prompt_length: int = 10,
                            generation_length: int = 100,
                            num_runs: int = 10) -> Dict:
        """Benchmark text generation speed."""
        import random
        
        times = []
        
        for _ in range(num_runs):
            # Random prompt
            prompt = [random.randint(0, self.model.vocab_size - 1) 
                     for _ in range(prompt_length)]
            
            start = time.time()
            
            # Generate
            self.model.generate(
                prompt[-1],
                max_length=generation_length,
                temperature=1.0
            )
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'tokens_per_sec': generation_length / (sum(times) / len(times)),
            'prompt_length': prompt_length,
            'generation_length': generation_length
        }
    
    def benchmark_training_step(self, 
                                seq_length: int = 128,
                                num_runs: int = 10) -> Dict:
        """Benchmark training step speed."""
        import random
        
        times = []
        
        for _ in range(num_runs):
            # Random sequence
            tokens = [random.randint(0, self.model.vocab_size - 1) 
                     for _ in range(seq_length)]
            
            start = time.time()
            
            # Forward + backward (simplified)
            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]
            
            for i, tok in enumerate(tokens[:-1]):
                logits = self.model.forward(tok, i, keys, values)
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'tokens_per_sec': seq_length / (sum(times) / len(times))
        }
    
    def print_report(self):
        """Print benchmark report."""
        print("=" * 60)
        print("SPEED BENCHMARK")
        print("=" * 60)
        
        # Generation
        print("\n📝 Generation")
        gen_results = self.benchmark_generation()
        print(f"  Avg time: {gen_results['avg_time']*1000:.1f} ms")
        print(f"  Tokens/sec: {gen_results['tokens_per_sec']:.1f}")
        
        # Training
        print("\n🎓 Training Step")
        train_results = self.benchmark_training_step()
        print(f"  Avg time: {train_results['avg_time']*1000:.1f} ms")
        print(f"  Tokens/sec: {train_results['tokens_per_sec']:.1f}")
        
        print("=" * 60)


def profile_model(model, operation: str = "all"):
    """
    Convenience function for profiling.
    """
    if operation == "analyze":
        analyzer = ModelAnalyzer(model)
        analyzer.print_report()
    elif operation == "speed":
        benchmark = SpeedBenchmark(model)
        benchmark.print_report()
    elif operation == "all":
        analyzer = ModelAnalyzer(model)
        analyzer.print_report()
        print()
        benchmark = SpeedBenchmark(model)
        benchmark.print_report()
