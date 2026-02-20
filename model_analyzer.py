"""
Model analyzer for microgpt ecosystem.
Analyzes model architecture, counts parameters, estimates compute.
"""

import math
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class ModelAnalysis:
    """Complete model analysis results."""
    total_params: int
    trainable_params: int
    model_size_mb: float
    flops_per_token: int
    memory_per_token: int
    attention_flops: int
    mlp_flops: int
    embedding_params: int
    layer_params: List[Dict[str, int]]
    recommendations: List[str]


class ModelAnalyzer:
    """Analyze microgpt models."""
    
    def __init__(self, model=None, config=None):
        self.model = model
        self.config = config
    
    def analyze(self) -> ModelAnalysis:
        """Perform complete model analysis."""
        if self.model is None and self.config is None:
            raise ValueError("Must provide model or config")
        
        if self.config is None:
            self.config = self.model.config
        
        # Get config values
        vocab_size = getattr(self.config, 'vocab_size', 100)
        n_embd = getattr(self.config, 'n_embd', 16)
        n_layer = getattr(self.config, 'n_layer', 1)
        n_head = getattr(self.config, 'n_head', 4)
        block_size = getattr(self.config, 'block_size', 16)
        
        # Calculate parameters
        embedding_params = self._embedding_params(vocab_size, n_embd, block_size)
        layer_params = self._layer_params(n_layer, n_embd, n_head)
        output_params = self._output_params(vocab_size, n_embd)
        
        total_params = embedding_params + sum(layer_params.values()) + output_params
        
        # Calculate FLOPs
        attention_flops = self._attention_flops(n_embd, n_head, block_size)
        mlp_flops = self._mlp_flops(n_embd)
        flops_per_token = n_layer * (attention_flops + mlp_flops)
        
        # Memory estimates
        memory_per_token = n_embd * 4  # 4 bytes per float32
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_params, n_layer, n_embd, block_size
        )
        
        return ModelAnalysis(
            total_params=total_params,
            trainable_params=total_params,  # All params are trainable
            model_size_mb=total_params * 4 / (1024 * 1024),  # float32
            flops_per_token=flops_per_token,
            memory_per_token=memory_per_token,
            attention_flops=attention_flops,
            mlp_flops=mlp_flops,
            embedding_params=embedding_params,
            layer_params=[{
                'attention': n_embd * n_embd * 4,  # Wq, Wk, Wv, Wo
                'mlp': n_embd * 4 * n_embd * 2,  # up and down projections
            }] * n_layer,
            recommendations=recommendations
        )
    
    def _embedding_params(self, vocab_size: int, n_embd: int, block_size: int) -> int:
        """Calculate embedding parameters."""
        token_emb = vocab_size * n_embd
        pos_emb = block_size * n_embd
        return token_emb + pos_emb
    
    def _layer_params(self, n_layer: int, n_embd: int, n_head: int) -> Dict[str, int]:
        """Calculate per-layer parameters."""
        # Attention: Wq, Wk, Wv, Wo
        attn_params = 4 * n_embd * n_embd
        
        # MLP: 2 linear layers with 4x expansion
        mlp_params = n_embd * 4 * n_embd + 4 * n_embd * n_embd
        
        return {
            'attention': attn_params,
            'mlp': mlp_params,
            'total': attn_params + mlp_params
        }
    
    def _output_params(self, vocab_size: int, n_embd: int) -> int:
        """Calculate output layer parameters."""
        return vocab_size * n_embd
    
    def _attention_flops(self, n_embd: int, n_head: int, seq_len: int) -> int:
        """Calculate FLOPs for attention."""
        head_dim = n_embd // n_head
        
        # Q, K, V projections: 3 * n_embd * n_embd
        qkv_flops = 3 * n_embd * n_embd
        
        # Attention scores: n_head * seq_len * seq_len * head_dim
        attn_scores = n_head * seq_len * seq_len * head_dim
        
        # Attention application: n_head * seq_len * head_dim * seq_len
        attn_apply = n_head * seq_len * head_dim * seq_len
        
        # Output projection: n_embd * n_embd
        out_proj = n_embd * n_embd
        
        return qkv_flops + attn_scores + attn_apply + out_proj
    
    def _mlp_flops(self, n_embd: int) -> int:
        """Calculate FLOPs for MLP."""
        # Two matmuls with 4x hidden dim
        return 2 * (n_embd * 4 * n_embd)
    
    def _generate_recommendations(self, total_params: int, n_layer: int, 
                                    n_embd: int, block_size: int) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Size recommendations
        if total_params < 10000:
            recommendations.append("Model is very small. Consider increasing n_embd or n_layer for better performance.")
        elif total_params > 10000000:
            recommendations.append("Model is large. Training will be slow in pure Python. Consider using PyTorch for production.")
        
        # Architecture recommendations
        if n_layer < 2:
            recommendations.append("Single layer may have limited capacity. Try n_layer=2-4 for better results.")
        
        if n_embd < 64:
            recommendations.append("Small embedding dimension. Consider n_embd >= 64 for better representations.")
        
        if block_size < 32:
            recommendations.append("Short context window. Consider block_size=64-128 for longer dependencies.")
        
        # Memory recommendations
        model_size_mb = total_params * 4 / (1024 * 1024)
        if model_size_mb > 100:
            recommendations.append(f"Model size ({model_size_mb:.1f} MB) is large for pure Python.")
        
        return recommendations
    
    def print_analysis(self, analysis: ModelAnalysis = None):
        """Print analysis report."""
        if analysis is None:
            analysis = self.analyze()
        
        print("=" * 70)
        print("Model Analysis Report")
        print("=" * 70)
        
        print(f"\nParameters:")
        print(f"  Total: {analysis.total_params:,}")
        print(f"  Trainable: {analysis.trainable_params:,}")
        print(f"  Model size: {analysis.model_size_mb:.2f} MB")
        
        print(f"\nArchitecture:")
        print(f"  Embedding: {analysis.embedding_params:,} params")
        print(f"  Per layer: {analysis.layer_params[0]['total']:,} params")
        print(f"  Attention: {analysis.attention_flops:,} FLOPs/layer")
        print(f"  MLP: {analysis.mlp_flops:,} FLOPs/layer")
        
        print(f"\nCompute:")
        print(f"  FLOPs per token: {analysis.flops_per_token:,}")
        print(f"  Memory per token: {analysis.memory_per_token} bytes")
        
        if analysis.recommendations:
            print(f"\nRecommendations:")
            for rec in analysis.recommendations:
                print(f"  💡 {rec}")
        
        print("=" * 70)
    
    def compare_configs(self, configs: List[Any]) -> Dict[str, Any]:
        """Compare multiple model configurations."""
        analyses = [ModelAnalyzer(config=c).analyze() for c in configs]
        
        comparison = {
            'configs': [],
            'summary': {
                'smallest': None,
                'largest': None,
                'fastest': None,
            }
        }
        
        for i, analysis in enumerate(analyses):
            comparison['configs'].append({
                'index': i,
                'params': analysis.total_params,
                'size_mb': analysis.model_size_mb,
                'flops': analysis.flops_per_token,
            })
        
        # Find extremes
        comparison['summary']['smallest'] = min(analyses, key=lambda x: x.total_params).total_params
        comparison['summary']['largest'] = max(analyses, key=lambda x: x.total_params).total_params
        comparison['summary']['fastest'] = min(analyses, key=lambda x: x.flops_per_token).flops_per_token
        
        return comparison


# Example usage
if __name__ == "__main__":
    from model import GPT, GPTConfig
    
    # Analyze default model
    config = GPTConfig(vocab_size=100, n_embd=64, n_layer=2, n_head=4, block_size=32)
    model = GPT(config)
    
    analyzer = ModelAnalyzer(model)
    analysis = analyzer.analyze()
    analyzer.print_analysis(analysis)
    
    # Compare configurations
    print("\n" + "=" * 70)
    print("Configuration Comparison")
    print("=" * 70)
    
    configs = [
        GPTConfig(vocab_size=100, n_embd=32, n_layer=1),
        GPTConfig(vocab_size=100, n_embd=64, n_layer=2),
        GPTConfig(vocab_size=100, n_embd=128, n_layer=4),
    ]
    
    comparison = analyzer.compare_configs(configs)
    for cfg in comparison['configs']:
        print(f"Config {cfg['index']}: {cfg['params']:,} params, "
              f"{cfg['size_mb']:.2f} MB, {cfg['flops']:,} FLOPs/token")
