"""
Performance profiler for microgpt ecosystem.
Profiles memory usage, computation time, and bottlenecks.
"""

import time
import sys
import random
import tracemalloc
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import json



@dataclass
class ProfileResult:
    """Result of profiling a function."""
    name: str
    calls: int = 0
    total_time: float = 0.0
    max_memory: int = 0
    avg_time: float = 0.0
    
    def update(self, duration: float, memory: int):
        """Update statistics."""
        self.calls += 1
        self.total_time += duration
        self.max_memory = max(self.max_memory, memory)
        self.avg_time = self.total_time / self.calls


class PerformanceProfiler:
    """Profile performance of microgpt components."""
    
    def __init__(self):
        self.profiles: Dict[str, ProfileResult] = {}
        self.active: bool = False
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block."""
        if not self.active:
            yield
            return
        
        # Start profiling
        tracemalloc.start()
        start_time = time.perf_counter()
        start_mem = tracemalloc.get_traced_memory()[0]
        
        try:
            yield
        finally:
            # End profiling
            end_time = time.perf_counter()
            end_mem = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            duration = end_time - start_time
            memory = end_mem - start_mem
            
            if name not in self.profiles:
                self.profiles[name] = ProfileResult(name)
            self.profiles[name].update(duration, memory)
    
    def start(self):
        """Start profiling."""
        self.active = True
        print("Profiling started...")
    
    def stop(self):
        """Stop profiling and show results."""
        self.active = False
        self._print_report()
    
    def _print_report(self):
        """Print profiling report."""
        print("\n" + "=" * 70)
        print("Performance Profile Report")
        print("=" * 70)
        print(f"{'Function':<40} {'Calls':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'Max Mem(KB)':<12}")
        print("-" * 70)
        
        for name, result in sorted(self.profiles.items(), key=lambda x: x[1].total_time, reverse=True):
            print(f"{name:<40} {result.calls:<8} {result.total_time:<10.4f} "
                  f"{result.avg_time*1000:<10.2f} {result.max_memory/1024:<12.2f}")
        
        print("=" * 70)
    
    def save_report(self, path: str = "profile_report.json"):
        """Save report to file."""
        data = {
            name: {
                "calls": r.calls,
                "total_time": r.total_time,
                "avg_time": r.avg_time,
                "max_memory_bytes": r.max_memory,
            }
            for name, r in self.profiles.items()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Profile report saved to {path}")


class ModelProfiler:
    """Profile model-specific metrics."""
    
    def __init__(self, model):
        self.model = model
        self.layer_times: Dict[str, List[float]] = {}
        self.activation_stats: Dict[str, Dict] = {}
    
    def profile_forward(self, tokens: List[int]) -> Dict[str, Any]:
        """Profile a forward pass."""
        results = {}
        
        # Time the full forward pass
        start = time.perf_counter()
        
        # This would hook into the model's forward pass
        # For now, just measure total time
        if hasattr(self.model, 'forward'):
            _ = self.model.forward(tokens)
        
        duration = time.perf_counter() - start
        results['total_time'] = duration
        
        # Estimate FLOPs
        if hasattr(self.model, 'config'):
            config = self.model.config
            seq_len = len(tokens)
            hidden = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
            layers = config.n_layer if hasattr(config, 'n_layer') else config.H_layers + config.L_layers
            
            # Approximate FLOPs for transformer
            flops = 2 * seq_len * hidden * hidden * layers
            results['estimated_flops'] = flops
            results['gflops_per_sec'] = flops / duration / 1e9 if duration > 0 else 0
        
        return results
    
    def memory_profile(self) -> Dict[str, int]:
        """Profile memory usage."""
        tracemalloc.start()
        
        # Get baseline
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Run forward pass
        if hasattr(self.model, 'forward'):
            tokens = [0] * 10  # Dummy tokens
            _ = self.model.forward(tokens)
        
        # Get peak
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'baseline_bytes': baseline,
            'peak_bytes': peak,
            'used_bytes': peak - baseline,
        }


def profile_training_loop(model, data, num_steps: int = 10) -> Dict[str, Any]:
    """Profile a training loop."""
    profiler = PerformanceProfiler()
    profiler.start()
    
    times = []
    losses = []
    
    for step in range(num_steps):
        with profiler.profile(f"step_{step}"):
            start = time.perf_counter()
            
            # Sample data
            tokens, targets = random.choice(data) if isinstance(data, list) else (data, data)
            
            # Train step
            if hasattr(model, 'train_step'):
                loss, info = model.train_step(tokens, targets)
                losses.append(loss.data if hasattr(loss, 'data') else loss)
            
            duration = time.perf_counter() - start
            times.append(duration)
    
    profiler.stop()
    
    return {
        'avg_step_time': sum(times) / len(times),
        'total_time': sum(times),
        'final_loss': losses[-1] if losses else None,
        'tokens_per_sec': len(tokens) / (sum(times) / len(times)) if times else 0,
    }


# Example usage
if __name__ == "__main__":
    print("Performance Profiler Demo")
    print("=" * 70)
    
    # Profile basic operations
    profiler = PerformanceProfiler()
    profiler.start()
    
    from microgpt import Value, softmax, rmsnorm
    
    # Profile Value operations
    with profiler.profile("value_ops"):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
    
    # Profile softmax
    with profiler.profile("softmax"):
        logits = [Value(random.random()) for _ in range(100)]
        probs = softmax(logits)
    
    # Profile rmsnorm
    with profiler.profile("rmsnorm"):
        x = [Value(random.random()) for _ in range(512)]
        y = rmsnorm(x)
    
    profiler.stop()
    profiler.save_report()
    
    # Profile model
    print("\n" + "=" * 70)
    print("Model Profiling")
    print("=" * 70)
    
    from model import GPT, GPTConfig
    
    config = GPTConfig(vocab_size=100, n_embd=32, n_layer=2)
    model = GPT(config)
    
    model_profiler = ModelProfiler(model)
    
    tokens = [random.randint(0, 99) for _ in range(20)]
    forward_stats = model_profiler.profile_forward(tokens)
    print(f"Forward pass: {forward_stats['total_time']*1000:.2f} ms")
    if 'gflops_per_sec' in forward_stats:
        print(f"Performance: {forward_stats['gflops_per_sec']:.4f} GFLOPs/s")
    
    mem_stats = model_profiler.memory_profile()
    print(f"Memory used: {mem_stats['used_bytes']/1024:.2f} KB")
