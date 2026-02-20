"""
Modern architecture improvements inspired by LLaMA, GPT-4, and other SOTA models.
Includes RoPE, SwiGLU, RMSNorm, better initialization, and more.
"""

import math
import random
from typing import List, Tuple, Optional
from model import Value, GPT as BaseGPT, rmsnorm, softmax


class RoPE:
    """
    Rotary Position Embedding (RoPE) from RoFormer.
    Used in LLaMA, PaLM, and other modern models.
    Better than learned positional embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        inv_freq = [1.0 / (base ** (2 * (i // 2) / dim) for i in range(dim))]
        self.inv_freq = inv_freq
    
    def apply(self, x: List[Value], position: int) -> List[Value]:
        """Apply rotary embeddings to input."""
        result = []
        for i in range(0, len(x), 2):
            if i + 1 < len(x):
                # Rotation matrix
                freq = self.inv_freq[i]
                cos_val = math.cos(position * freq)
                sin_val = math.sin(position * freq)
                
                # Apply rotation
                x0, x1 = x[i].data, x[i+1].data
                result.append(Value(x0 * cos_val - x1 * sin_val))
                result.append(Value(x0 * sin_val + x1 * cos_val))
            else:
                result.append(x[i])
        
        return result


class SwiGLU:
    """
    SwiGLU activation from PaLM.
    Better than ReLU/GELU for transformers.
    """
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)"""
        # Simplified - in practice, this uses gating
        # Swish(x) = x * sigmoid(x)
        result = []
        for xi in x:
            # Swish activation
            sigmoid = 1.0 / (1.0 + math.exp(-xi.data))
            swish = xi.data * sigmoid
            result.append(Value(swish))
        
        return result


class ALiBi:
    """
    Attention with Linear Biases from ALiBi paper.
    Better extrapolation to longer sequences.
    """
    
    def __init__(self, n_heads: int):
        self.n_heads = n_heads
        # Linear bias slopes
        self.slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    
    def apply_bias(self, attn_scores: List[List[float]], head_idx: int) -> List[List[Value]]:
        """Add linear bias to attention scores."""
        slope = self.slopes[head_idx]
        result = []
        
        for i, row in enumerate(attn_scores):
            biased_row = []
            for j, score in enumerate(row):
                # Linear distance penalty
                bias = -slope * abs(i - j)
                biased_row.append(Value(score + bias))
            result.append(biased_row)
        
        return result


class ModernGPT(BaseGPT):
    """
    GPT with modern architectural improvements.
    Based on LLaMA, PaLM, and GPT-4 design choices.
    """
    
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 1,
                 n_embd: int = 16, n_head: int = 4, dropout: float = 0.0,
                 use_rope: bool = True, use_swiglu: bool = True,
                 use_alibi: bool = False, use_rmsnorm: bool = True):
        # Initialize base
        super().__init__(vocab_size, block_size, n_layer, n_embd, n_head, 
                        dropout, use_gelu=False, use_layernorm=not use_rmsnorm)
        
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.use_alibi = use_alibi
        
        # Modern components
        if use_rope:
            self.rope = RoPE(n_embd // n_head, block_size)
        if use_alibi:
            self.alibi = ALiBi(n_head)
        if use_swiglu:
            self.swiglu = SwiGLU()
        
        # Re-initialize with better initialization
        self._modern_init()
    
    def _modern_init(self):
        """Modern weight initialization (Kaiming/He style)."""
        import math
        
        for name, matrix in self.state_dict.items():
            n_in = len(matrix[0]) if matrix else 0
            n_out = len(matrix)
            
            # Kaiming initialization
            std = math.sqrt(2.0 / n_in) if n_in > 0 else 0.01
            
            for i in range(n_out):
                for j in range(n_in):
                    # Use normal distribution
                    self.state_dict[name][i][j].data = random.gauss(0, std)
        
        # Special initialization for embeddings
        if 'wte' in self.state_dict:
            for row in self.state_dict['wte']:
                for v in row:
                    v.data = random.gauss(0, 0.02)
        
        # Zero out bias (if any)
        # In modern transformers, biases are often removed
    
    def _apply_rope(self, q: List[Value], k: List[Value], pos: int) -> Tuple[List[Value], List[Value]]:
        """Apply rotary embeddings to Q and K."""
        if not self.use_rope:
            return q, k
        
        q_rot = self.rope.apply(q, pos)
        k_rot = self.rope.apply(k, pos)
        return q_rot, k_rot
    
    def _activate_modern(self, x: List[Value]) -> List[Value]:
        """Use SwiGLU if enabled, otherwise GELU."""
        if self.use_swiglu:
            return self.swiglu(x)
        # Fallback to GELU
        return [xi.gelu() for xi in x]


class FlashAttention:
    """
    Memory-efficient attention (conceptual implementation).
    Real Flash Attention requires CUDA kernels.
    """
    
    @staticmethod
    def apply(Q: List[List[Value]], K: List[List[Value]], V: List[List[Value]],
              scale: float) -> List[List[Value]]:
        """
        Memory-efficient attention computation.
        Reduces HBM accesses from O(N^2) to O(N).
        """
        # This is a conceptual implementation
        # Real implementation needs custom CUDA kernels
        
        # Simplified: standard attention with tiling
        n = len(Q)
        head_dim = len(Q[0]) if Q else 0
        
        # Tile size for memory efficiency
        tile_size = 64
        
        output = [[Value(0.0) for _ in range(head_dim)] for _ in range(n)]
        
        for i in range(0, n, tile_size):
            i_end = min(i + tile_size, n)
            
            for j in range(0, n, tile_size):
                j_end = min(j + tile_size, n)
                
                # Compute attention for this tile
                for ii in range(i, i_end):
                    # Q @ K^T for this tile
                    scores = []
                    for jj in range(j, j_end):
                        score = sum(Q[ii][k].data * K[jj][k].data for k in range(head_dim)) * scale
                        scores.append((jj, score))
                    
                    # Softmax for this tile (simplified)
                    max_score = max(s for _, s in scores)
                    exp_scores = [(jj, math.exp(s - max_score)) for jj, s in scores]
                    sum_exp = sum(e for _, e in exp_scores)
                    
                    # Accumulate to output
                    for jj, exp_s in exp_scores:
                        for k in range(head_dim):
                            output[ii][k].data += (exp_s / sum_exp) * V[jj][k].data
        
        return output


class GroupedQueryAttention:
    """
    Grouped Query Attention from LLaMA 2.
    Reduces memory and computation while maintaining quality.
    """
    
    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
    
    def repeat_kv(self, x: List[List[Value]]) -> List[List[Value]]:
        """Repeat K/V heads to match Q heads."""
        if self.n_rep == 1:
            return x
        
        result = []
        for head in x:
            for _ in range(self.n_rep):
                result.append(head)
        
        return result


def create_modern_model(size: str = "small") -> ModernGPT:
    """
    Create a modern GPT model with best practices.
    
    Sizes: tiny, small, medium, large, xl
    """
    configs = {
        "tiny": {"vocab_size": 100, "block_size": 128, "n_layer": 2, "n_embd": 128, "n_head": 4},
        "small": {"vocab_size": 1000, "block_size": 512, "n_layer": 6, "n_embd": 384, "n_head": 6},
        "medium": {"vocab_size": 5000, "block_size": 1024, "n_layer": 12, "n_embd": 768, "n_head": 12},
        "large": {"vocab_size": 10000, "block_size": 2048, "n_layer": 24, "n_embd": 1536, "n_head": 16},
        "xl": {"vocab_size": 50000, "block_size": 4096, "n_layer": 48, "n_embd": 2048, "n_head": 32},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    config = configs[size]
    
    return ModernGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        use_rope=True,
        use_swiglu=True,
        use_alibi=False,  # RoPE is generally preferred now
        use_rmsnorm=True
    )
