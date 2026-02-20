"""
Model comparison utilities for microgpt.
Compare different models, architectures, and configurations.
"""

import json
import time
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelComparison:
    """Comparison result between models."""
    model_a_name: str
    model_b_name: str
    metrics: Dict[str, Dict[str, float]]
    winner: str
    differences: Dict[str, Any]


class ModelComparator:
    """Compare multiple models on various metrics."""
    
    def __init__(self):
        self.results: List[ModelComparison] = []
    
    def compare(self, model_a: Any, model_b: Any, 
                name_a: str, name_b: str,
                eval_funcs: Dict[str, Callable]) -> ModelComparison:
        """Compare two models on multiple metrics."""
        print(f"Comparing {name_a} vs {name_b}...")
        
        metrics = {}
        for metric_name, eval_func in eval_funcs.items():
            score_a = eval_func(model_a)
            score_b = eval_func(model_b)
            
            metrics[metric_name] = {
                name_a: score_a,
                name_b: score_b,
                'difference': score_a - score_b,
                'relative': (score_a - score_b) / max(abs(score_b), 1e-8)
            }
        
        # Determine winner
        wins_a = sum(1 for m in metrics.values() if m['difference'] < 0)  # Lower is better
        wins_b = sum(1 for m in metrics.values() if m['difference'] > 0)
        
        winner = name_a if wins_a > wins_b else name_b if wins_b > wins_a else "tie"
        
        comparison = ModelComparison(
            model_a_name=name_a,
            model_b_name=name_b,
            metrics=metrics,
            winner=winner,
            differences={k: v['difference'] for k, v in metrics.items()}
        )
        
        self.results.append(comparison)
        return comparison
    
    def compare_configs(self, configs: List[Any], names: List[str],
                       train_func: Callable, eval_func: Callable) -> Dict[str, Any]:
        """Compare multiple configurations."""
        print(f"Comparing {len(configs)} configurations...")
        
        results = []
        for config, name in zip(configs, names):
            print(f"  Training {name}...")
            model = train_func(config)
            score = eval_func(model)
            results.append({
                'name': name,
                'config': config,
                'score': score
            })
        
        # Rank by score
        ranked = sorted(results, key=lambda x: x['score'])
        
        return {
            'rankings': ranked,
            'best': ranked[0],
            'worst': ranked[-1],
            'comparison': results
        }
    
    def ab_test(self, model_a: Any, model_b: Any, 
               test_cases: List[Dict[str, Any]],
               judge_func: Callable) -> Dict[str, Any]:
        """A/B test two models on specific test cases."""
        print(f"Running A/B test with {len(test_cases)} cases...")
        
        results = []
        for i, test in enumerate(test_cases):
            output_a = model_a.generate(test['input']) if hasattr(model_a, 'generate') else None
            output_b = model_b.generate(test['input']) if hasattr(model_b, 'generate') else None
            
            judgment = judge_func(test['input'], output_a, output_b, test.get('expected'))
            
            results.append({
                'test_id': i,
                'input': test['input'],
                'output_a': output_a,
                'output_b': output_b,
                'winner': judgment.get('winner', 'unknown'),
                'reason': judgment.get('reason', '')
            })
        
        wins_a = sum(1 for r in results if r['winner'] == 'a')
        wins_b = sum(1 for r in results if r['winner'] == 'b')
        
        return {
            'total_tests': len(test_cases),
            'wins_a': wins_a,
            'wins_b': wins_b,
            'tie': len(test_cases) - wins_a - wins_b,
            'winner': 'a' if wins_a > wins_b else 'b' if wins_b > wins_a else 'tie',
            'detailed_results': results
        }
    
    def generate_report(self, output_path: str = "comparison_report.md"):
        """Generate comparison report."""
        lines = ["# Model Comparison Report\n"]
        
        for comp in self.results:
            lines.append(f"## {comp.model_a_name} vs {comp.model_b_name}")
            lines.append(f"**Winner**: {comp.winner}\n")
            
            lines.append("### Metrics")
            for metric, values in comp.metrics.items():
                lines.append(f"- **{metric}**:")
                lines.append(f"  - {comp.model_a_name}: {values[comp.model_a_name]:.4f}")
                lines.append(f"  - {comp.model_b_name}: {values[comp.model_b_name]:.4f}")
                lines.append(f"  - Difference: {values['difference']:.4f} ({values['relative']*100:.1f}%)")
            
            lines.append("")
        
        report = "\n".join(lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")
        return report


class PerformanceComparator:
    """Compare model performance characteristics."""
    
    @staticmethod
    def measure_speed(model: Any, input_data: List, num_runs: int = 10) -> Dict[str, float]:
        """Measure inference speed."""
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            if hasattr(model, 'forward'):
                _ = model.forward(input_data)
            elif hasattr(model, 'generate'):
                _ = model.generate(input_data[0] if input_data else "")
            times.append(time.perf_counter() - start)
        
        return {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    @staticmethod
    def measure_memory(model: Any) -> Dict[str, int]:
        """Estimate memory usage."""
        # Count parameters
        if hasattr(model, 'parameters'):
            params = len(model.parameters())
        elif hasattr(model, 'state_dict'):
            params = sum(len(v) for v in model.state_dict().values())
        else:
            params = 0
        
        return {
            'parameters': params,
            'size_mb': params * 4 / (1024 * 1024),  # float32
            'size_mb_fp16': params * 2 / (1024 * 1024),  # float16
        }
    
    @staticmethod
    def compare_architectures(architectures: List[Any], names: List[str],
                              input_data: List) -> Dict[str, Any]:
        """Compare different architectures."""
        results = []
        
        for arch, name in zip(architectures, names):
            print(f"Benchmarking {name}...")
            
            speed = PerformanceComparator.measure_speed(arch, input_data)
            memory = PerformanceComparator.measure_memory(arch)
            
            results.append({
                'name': name,
                'speed': speed,
                'memory': memory,
                'efficiency': speed['mean'] / max(memory['size_mb'], 1e-8)
            })
        
        # Rank by efficiency
        ranked = sorted(results, key=lambda x: x['efficiency'])
        
        return {
            'rankings': ranked,
            'fastest': min(results, key=lambda x: x['speed']['mean']),
            'smallest': min(results, key=lambda x: x['memory']['size_mb']),
            'most_efficient': ranked[0]
        }


# Example usage
if __name__ == "__main__":
    from model import GPT, GPTConfig
    
    # Create models to compare
    config_small = GPTConfig(vocab_size=100, n_embd=32, n_layer=1)
    config_large = GPTConfig(vocab_size=100, n_embd=64, n_layer=2)
    
    model_small = GPT(config_small)
    model_large = GPT(config_large)
    
    # Compare
    comparator = ModelComparator()
    
    def eval_loss(model):
        # Dummy evaluation
        return len(model.parameters()) / 10000
    
    comparison = comparator.compare(
        model_small, model_large,
        "small_model", "large_model",
        {'loss': eval_loss, 'params': lambda m: len(m.parameters())}
    )
    
    print(f"\nWinner: {comparison.winner}")
    print(f"Metrics: {comparison.metrics}")
    
    # Performance comparison
    perf_comp = PerformanceComparator()
    speed_small = perf_comp.measure_speed(model_small, [[1, 2, 3]])
    speed_large = perf_comp.measure_speed(model_large, [[1, 2, 3]])
    
    print(f"\nSpeed - Small: {speed_small['mean']*1000:.2f}ms")
    print(f"Speed - Large: {speed_large['mean']*1000:.2f}ms")
    
    # Generate report
    comparator.generate_report()
