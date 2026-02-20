"""
Memory-efficient training and inference techniques.
Includes gradient checkpointing, LoRA, QLoRA, and more.
"""

import random
import math
from typing import List, Optional, Dict, Tuple
from model import Value, GPT


class GradientCheckpointing:
    """
    Gradient checkpointing to trade compute for memory.
    Recompute activations during backward pass instead of storing.
    """
    
    def __init__(self, model: GPT, checkpoint_every: int = 1):
        self.model = model
        self.checkpoint_every = checkpoint_every
        self.checkpoints: Dict[int, dict] = {}
    
    def forward_with_checkpoints(self, tokens: List[int]) -> List[Value]:
        """
        Forward pass with selective checkpointing.
        """
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]
        
        # Store checkpoints periodically
        for i, token in enumerate(tokens):
            if i % self.checkpoint_every == 0:
                self.checkpoints[i] = {
                    'token': token,
                    'position': i,
                    'keys': [list(k) for k in keys],
                    'values': [list(v) for v in values]
                }
            
            logits = self.model.forward(token, i, keys, values)
        
        return logits
    
    def backward_with_checkpoints(self, loss: Value, tokens: List[int]):
        """
        Backward pass with recomputation from checkpoints.
        """
        # Standard backward
        loss.backward()
        
        # In real implementation, would recompute from checkpoints
        # to save memory during backward pass


class LoRA:
    """
    Low-Rank Adaptation (LoRA).
    Efficient fine-tuning with low-rank matrices.
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, 
                 alpha: float = 16.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = [[random.gauss(0, 0.01) for _ in range(in_dim)] 
                       for _ in range(rank)]
        self.lora_B = [[0.0 for _ in range(rank)] 
                       for _ in range(out_dim)]  # Initialize B to zero
    
    def forward(self, x: List[Value], base_output: List[Value]) -> List[Value]:
        """
        Forward pass: base_output + scaling * B @ A @ x
        """
        # A @ x
        Ax = [sum(self.lora_A[i][j] * x[j].data for j in range(self.in_dim)) 
              for i in range(self.rank)]
        
        # B @ (A @ x)
        BAx = [sum(self.lora_B[i][j] * Ax[j] for j in range(self.rank)) 
               for i in range(self.out_dim)]
        
        # Add to base output with scaling
        return [Value(bo.data + self.scaling * ba) 
                for bo, ba in zip(base_output, BAx)]
    
    def merge_weights(self, base_weight: List[List[Value]]) -> List[List[Value]]:
        """
        Merge LoRA weights into base weights for inference.
        W_merged = W + scaling * B @ A
        """
        # Compute B @ A
        BA = [[sum(self.lora_B[i][k] * self.lora_A[k][j] 
                  for k in range(self.rank))
               for j in range(self.in_dim)]
              for i in range(self.out_dim)]
        
        # Merge
        merged = [[Value(w.data + self.scaling * ba) 
                  for w, ba in zip(row_w, row_ba)]
                 for row_w, row_ba in zip(base_weight, BA)]
        
        return merged
    
    def parameters(self) -> List[Value]:
        """Get trainable parameters (only A and B)."""
        params = []
        for row in self.lora_A:
            for val in row:
                params.append(Value(val))
        for row in self.lora_B:
            for val in row:
                params.append(Value(val))
        return params


class QLoRA:
    """
    Quantized LoRA (QLoRA).
    LoRA with 4-bit base model quantization.
    """
    
    def __init__(self, base_model: GPT, rank: int = 64, 
                 bits: int = 4, group_size: int = 64):
        self.base_model = base_model
        self.rank = rank
        self.bits = bits
        self.group_size = group_size
        
        # Quantize base model
        self.quantized_weights = self._quantize_model()
        
        # LoRA adapters
        self.lora_layers: Dict[str, LoRA] = {}
        self._init_lora()
    
    def _quantize_model(self) -> Dict[str, dict]:
        """Quantize base model weights."""
        quantized = {}
        
        for name, matrix in self.base_model.state_dict.items():
            # Group-wise quantization
            for g in range(0, len(matrix), self.group_size):
                group = matrix[g:g+self.group_size]
                
                # Find range
                values = [v.data for row in group for v in row]
                min_val = min(values)
                max_val = max(values)
                
                # Quantize
                scale = (max_val - min_val) / (2 ** self.bits - 1)
                quantized[(name, g)] = {
                    'min': min_val,
                    'scale': scale,
                    'values': [
                        [int((v.data - min_val) / scale) for v in row]
                        for row in group
                    ]
                }
        
        return quantized
    
    def _init_lora(self):
        """Initialize LoRA layers for all linear layers."""
        for name, matrix in self.base_model.state_dict.items():
            if 'wte' in name or 'wpe' in name:
                continue  # Don't add LoRA to embeddings
            
            in_dim = len(matrix[0]) if matrix else 0
            out_dim = len(matrix)
            
            self.lora_layers[name] = LoRA(in_dim, out_dim, self.rank)
    
    def forward(self, token_id: int, pos: int, 
                keys: List[List], values: List[List]) -> List[Value]:
        """
        Forward pass with quantized base + LoRA.
        """
        # Dequantize on-the-fly (simplified)
        # Real implementation uses efficient kernels
        
        # Use base model forward with LoRA
        logits = self.base_model.forward(token_id, pos, keys, values)
        
        # Add LoRA outputs
        for name, lora in self.lora_layers.items():
            if 'attn' in name or 'mlp' in name:
                # Apply LoRA to this layer's output
                # Simplified - would need to intercept intermediate activations
                pass
        
        return logits


class DoRA:
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).
    Better than LoRA by decomposing into magnitude and direction.
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        
        # Magnitude component
        self.magnitude = [1.0] * out_dim
        
        # Direction component (low-rank)
        self.lora = LoRA(in_dim, out_dim, rank)
    
    def forward(self, x: List[Value], base_weight: List[List[Value]]) -> List[Value]:
        """
        DoRA forward: (magnitude * direction) @ x
        """
        # Compute direction from base + LoRA
        direction = self.lora.merge_weights(base_weight)
        
        # Normalize direction
        norms = [sum(d[i].data ** 2 for i in range(self.in_dim)) ** 0.5 
                 for d in direction]
        normalized = [[d[i].data / (n + 1e-8) for i in range(self.in_dim)] 
                      for d, n in zip(direction, norms)]
        
        # Scale by magnitude
        scaled = [[self.magnitude[i] * normalized[i][j] 
                  for j in range(self.in_dim)]
                 for i in range(self.out_dim)]
        
        # Apply to input
        output = [sum(scaled[i][j] * x[j].data for j in range(self.in_dim)) 
                 for i in range(self.out_dim)]
        
        return [Value(o) for o in output]


class ReLoRA:
    """
    ReLoRA: Restarting with LoRA for higher rank training.
    """
    
    def __init__(self, model: GPT, rank: int = 8, 
                 restart_interval: int = 1000):
        self.model = model
        self.rank = rank
        self.restart_interval = restart_interval
        self.step_count = 0
        
        self.lora_layers: Dict[str, LoRA] = {}
        self._init_lora()
    
    def _init_lora(self):
        """Initialize LoRA layers."""
        for name, matrix in self.model.state_dict.items():
            in_dim = len(matrix[0]) if matrix else 0
            out_dim = len(matrix)
            self.lora_layers[name] = LoRA(in_dim, out_dim, self.rank)
    
    def step(self):
        """Training step with periodic restarts."""
        self.step_count += 1
        
        if self.step_count % self.restart_interval == 0:
            self._restart()
    
    def _restart(self):
        """
        Merge LoRA into base weights and reinitialize LoRA.
        """
        for name, lora in self.lora_layers.items():
            # Merge
            base_weight = self.model.state_dict[name]
            merged = lora.merge_weights(base_weight)
            self.model.state_dict[name] = merged
            
            # Reinitialize LoRA
            in_dim = lora.in_dim
            out_dim = lora.out_dim
            self.lora_layers[name] = LoRA(in_dim, out_dim, self.rank)


class GaLore:
    """
    Gradient Low-Rank Projection (GaLore).
    Memory-efficient training by projecting gradients to low-rank.
    """
    
    def __init__(self, rank: int = 128, update_freq: int = 200):
        self.rank = rank
        self.update_freq = update_freq
        self.step_count = 0
        
        # Projection matrices
        self.U: Dict[str, List[List[float]]] = {}
        self.V: Dict[str, List[List[float]]] = {}
    
    def project_gradient(self, name: str, grad: List[List[float]]) -> List[List[float]]:
        """
        Project gradient to low-rank subspace.
        """
        if name not in self.U or self.step_count % self.update_freq == 0:
            # Compute SVD (simplified)
            self._update_projection(name, grad)
        
        # Project: U^T @ grad @ V
        # Simplified - real implementation uses efficient matmul
        return grad
    
    def _update_projection(self, name: str, grad: List[List[float]]):
        """Update projection matrices using SVD."""
        # Simplified SVD computation
        # Real implementation uses power iteration or randomized SVD
        
        m = len(grad)
        n = len(grad[0]) if grad else 0
        
        # Initialize random projection matrices
        self.U[name] = [[random.gauss(0, 1) for _ in range(self.rank)] 
                        for _ in range(m)]
        self.V[name] = [[random.gauss(0, 1) for _ in range(n)] 
                        for _ in range(self.rank)]


class UnslothOptimizations:
    """
    Optimizations from Unsloth for faster training.
    """
    
    @staticmethod
    def optimize_attention():
        """
        Optimized attention kernels (conceptual).
        """
        # Unsloth uses hand-optimized CUDA kernels
        # This is a placeholder for the concept
        pass
    
    @staticmethod
    def optimize_embedding():
        """
        Optimized embedding lookup.
        """
        pass
    
    @staticmethod
    def reduce_vram_usage(model: GPT) -> float:
        """
        Estimate VRAM reduction from optimizations.
        """
        # Unsloth claims 70% VRAM reduction
        return 0.7


class LongLoRA:
    """
    LongLoRA: Efficient fine-tuning of long-context LLMs.
    """
    
    def __init__(self, model: GPT, rank: int = 8, 
                 context_extension: int = 8):
        self.model = model
        self.rank = rank
        self.context_extension = context_extension
        
        # Shift short attention for training
        self.shift_size = model.block_size // 4
    
    def shift_attention(self, x: List[List[Value]]) -> List[List[Value]]:
        """
        Shifted sparse attention for efficient training.
        """
        # Group and shift
        groups = []
        for i in range(0, len(x), self.shift_size):
            group = x[i:i+self.shift_size]
            # Shift within group
            shifted = group[self.shift_size//2:] + group[:self.shift_size//2]
            groups.extend(shifted)
        
        return groups
    
    def train_step(self, tokens: List[int]):
        """
        Training step with shifted attention.
        """
        # Use shifted attention during training
        # Use full attention during inference
        pass


def apply_memory_optimizations(model: GPT, 
                                use_lora: bool = True,
                                use_checkpointing: bool = True,
                                use_quantization: bool = False) -> dict:
    """
    Apply all memory optimizations to a model.
    """
    optimizations = {}
    
    if use_checkpointing:
        optimizations['checkpointing'] = GradientCheckpointing(model)
    
    if use_lora:
        optimizations['lora'] = {
            name: LoRA(len(matrix[0]), len(matrix), rank=8)
            for name, matrix in model.state_dict.items()
            if 'wte' not in name and 'wpe' not in name
        }
    
    if use_quantization:
        optimizations['qlora'] = QLoRA(model)
    
    return optimizations
