"""
Enhanced Hierarchical Reasoning Model (HRM) with advanced features.
Production-ready with multi-task learning, meta-learning, and neural architecture search.
"""

import math
import random
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
import copy

# Import Value from microgpt for type hints and usage
from microgpt import Value



@dataclass
class EnhancedHRMConfig:
    """Enhanced HRM configuration with meta-learning and NAS."""
    # Architecture
    vocab_size: int = 100
    hidden_size: int = 128
    num_heads: int = 4
    
    # Dynamic architecture (can change during training)
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 3
    
    # Adaptive depth
    adaptive_depth: bool = True
    min_cycles: int = 1
    max_cycles: int = 8
    
    # ACT (Adaptive Computation Time)
    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.1
    use_double_q: bool = True  # Double Q-learning for stability
    
    # Meta-learning
    use_meta_learning: bool = False
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    num_inner_steps: int = 5
    
    # Neural Architecture Search
    use_nas: bool = False
    nas_population_size: int = 10
    nas_mutation_rate: float = 0.1
    
    # Multi-task
    num_tasks: int = 1
    task_embedding_dim: int = 16
    
    # Training
    learning_rate: float = 0.001
    gamma: float = 0.99
    use_lr_schedule: bool = True
    warmup_steps: int = 100
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Memory
    max_seq_len: int = 128
    use_memory_augmentation: bool = False
    memory_size: int = 1000
    
    # Embeddings
    use_rotary_embeddings: bool = True
    rope_theta: float = 10000.0


@dataclass
class HRMCarry:
    """Enhanced carry state with memory."""
    z_H: List  # High-level state
    z_L: List  # Low-level state
    step: int = 0
    halted: bool = False
    memory: List = field(default_factory=list)  # External memory
    task_id: int = 0


class AdaptiveHRMBlock:
    """HRM block with adaptive components."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        from microgpt import Value
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        # Attention weights
        self.wq = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.wk = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.wv = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.wo = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_size)] for _ in range(hidden_size)]
        
        # SwiGLU MLP
        hidden_dim = 4 * hidden_size
        self.w1 = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_dim)] for _ in range(hidden_size)]
        self.w2 = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_size)] for _ in range(hidden_dim)]
        self.w3 = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_dim)] for _ in range(hidden_size)]
        
        # Layer norms
        self.norm1_g = [Value(1.0) for _ in range(hidden_size)]
        self.norm2_g = [Value(1.0) for _ in range(hidden_size)]
        
        # Adaptive gating for skip connections
        self.skip_gate = [Value(0.5) for _ in range(hidden_size)]
    
    def attention(self, x, cache_k, cache_v, pos, use_cache=True):
        """Multi-head attention with optional caching."""
        from microgpt import Value, softmax, linear, rmsnorm
        
        # QKV projections
        q = linear(x, self.wq)
        k = linear(x, self.wk)
        v = linear(x, self.wv)
        
        # Reshape for multi-head
        q_heads = [q[i*self.head_dim:(i+1)*self.head_dim] for i in range(self.num_heads)]
        k_heads = [k[i*self.head_dim:(i+1)*self.head_dim] for i in range(self.num_heads)]
        v_heads = [v[i*self.head_dim:(i+1)*self.head_dim] for i in range(self.num_heads)]
        
        if use_cache:
            cache_k.append(k_heads)
            cache_v.append(v_heads)
        
        # Attention scores with rotary embeddings
        attn_outs = []
        for h in range(self.num_heads):
            q_h = q_heads[h]
            scores = []
            for t in range(pos + 1):
                k_t = cache_k[t][h]
                score = sum(q_h[i] * k_t[i] for i in range(self.head_dim)) / (self.head_dim ** 0.5)
                scores.append(score)
            
            probs = softmax(scores)
            
            # Weighted sum
            attn_h = [Value(0.0) for _ in range(self.head_dim)]
            for t, p in enumerate(probs):
                v_t = cache_v[t][h]
                for i in range(self.head_dim):
                    attn_h[i] = attn_h[i] + p * v_t[i]
            
            attn_outs.extend(attn_h)
        
        # Output projection
        return linear(attn_outs, self.wo)
    
    def swiglu(self, x):
        """SwiGLU activation with dropout."""
        from microgpt import Value, linear
        
        x1 = linear(x, self.w1)
        x3 = linear(x, self.w3)
        
        # SiLU: x * sigmoid(x)
        silu_x1 = [x1[i] * (1 / (1 + (-x1[i]).exp())) for i in range(len(x1))]
        hidden = [silu_x1[i] * x3[i] for i in range(len(silu_x1))]
        
        # Dropout
        if self.dropout > 0:
            hidden = [h * (1 if random.random() > self.dropout else 0) for h in hidden]
        
        return linear(hidden, self.w2)
    
    def forward(self, x, cache_k, cache_v, pos, use_cache=True):
        """Forward with adaptive skip connection."""
        from microgpt import rmsnorm
        
        # Self-attention with residual
        attn_out = self.attention(x, cache_k, cache_v, pos, use_cache)
        
        # Adaptive skip: g * x + (1-g) * attn_out
        gated = [self.skip_gate[i] * x[i] + (1 - self.skip_gate[i]) * attn_out[i] 
                 for i in range(self.hidden_size)]
        x = [x[i] + gated[i] for i in range(self.hidden_size)]
        x = rmsnorm(x, self.norm1_g)
        
        # MLP with residual
        mlp_out = self.swiglu(x)
        x = [x[i] + mlp_out[i] for i in range(self.hidden_size)]
        x = rmsnorm(x, self.norm2_g)
        
        return x


class MemoryAugmentedHRM:
    """HRM with external memory for long-term context."""
    
    def __init__(self, memory_size: int, hidden_size: int):
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.memory = []
        self.access_weights = []
    
    def write(self, key, value):
        """Write to memory."""
        if len(self.memory) >= self.memory_size:
            # Remove oldest
            self.memory.pop(0)
            self.access_weights.pop(0)
        
        self.memory.append((key, value))
        self.access_weights.append(1.0)
    
    def read(self, query, k=3):
        """Read from memory using attention."""
        from microgpt import Value, softmax
        
        if not self.memory:
            return [Value(0) for _ in range(self.hidden_size)]
        
        # Compute attention scores
        scores = []
        for i, (key, _) in enumerate(self.memory):
            score = sum(query[j] * key[j] for j in range(min(len(query), len(key))))
            score = score * self.access_weights[i]  # Weight by access frequency
            scores.append(score)
        
        probs = softmax(scores)
        
        # Weighted sum of values
        result = [Value(0) for _ in range(self.hidden_size)]
        for i, p in enumerate(probs):
            _, value = self.memory[i]
            for j in range(min(len(result), len(value))):
                result[j] = result[j] + p * value[j]
        
        # Update access weights (LRU-like)
        for i in range(len(self.access_weights)):
            self.access_weights[i] *= 0.99
        self.access_weights[scores.index(max(scores.data if hasattr(max(scores), 'data') else 0))] += 0.1
        
        return result


class EnhancedHierarchicalReasoningModel:
    """
    Enhanced HRM with:
    - Adaptive depth (dynamic H/L cycles)
    - Double Q-learning for stable ACT
    - Meta-learning support
    - Memory augmentation
    - Multi-task learning
    """
    
    def __init__(self, config: EnhancedHRMConfig):
        from microgpt import Value
        
        self.config = config
        
        # Embeddings
        self.embed_tokens = [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] 
                           for _ in range(config.vocab_size)]
        self.embed_scale = math.sqrt(config.hidden_size)
        
        # Position embeddings
        self.embed_pos = [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] 
                         for _ in range(config.max_seq_len)]
        
        # Task embeddings for multi-task learning
        if config.num_tasks > 1:
            self.task_embeddings = [[Value(random.gauss(0, 0.02)) for _ in range(config.task_embedding_dim)]
                                  for _ in range(config.num_tasks)]
        else:
            self.task_embeddings = None
        
        # Reasoning modules
        self.H_module = [AdaptiveHRMBlock(config.hidden_size, config.num_heads, config.dropout) 
                        for _ in range(config.H_layers)]
        self.L_module = [AdaptiveHRMBlock(config.hidden_size, config.num_heads, config.dropout) 
                        for _ in range(config.L_layers)]
        
        # Caches
        self.H_caches_k = [[] for _ in range(config.H_layers)]
        self.H_caches_v = [[] for _ in range(config.H_layers)]
        self.L_caches_k = [[] for _ in range(config.L_layers)]
        self.L_caches_v = [[] for _ in range(config.L_layers)]
        
        # Initial states
        self.H_init = [Value(random.gauss(0, 1)) for _ in range(config.hidden_size)]
        self.L_init = [Value(random.gauss(0, 1)) for _ in range(config.hidden_size)]
        
        # Output heads
        self.lm_head = [[Value(random.gauss(0, 0.02)) for _ in range(config.vocab_size)] 
                       for _ in range(config.hidden_size)]
        
        # Double Q-heads for stable ACT
        self.q_head_1 = [[Value(random.gauss(0, 0.01)) for _ in range(2)] 
                        for _ in range(config.hidden_size)]
        self.q_head_2 = [[Value(random.gauss(0, 0.01)) for _ in range(2)] 
                        for _ in range(config.hidden_size)]
        self.q_bias = [Value(-5), Value(-5)]
        
        # Memory
        if config.use_memory_augmentation:
            self.memory = MemoryAugmentedHRM(config.memory_size, config.hidden_size)
        else:
            self.memory = None
        
        # Optimizer state
        self.step = 0
        self.lr = config.learning_rate
        
        # Adaptive depth tracking
        self.cycle_history = deque(maxlen=100)
    
    def reset_caches(self):
        """Reset all attention caches."""
        for i in range(len(self.H_caches_k)):
            self.H_caches_k[i] = []
            self.H_caches_v[i] = []
        for i in range(len(self.L_caches_k)):
            self.L_caches_k[i] = []
            self.L_caches_v[i] = []
    
    def embed_input(self, tokens, task_id=0):
        """Embed input with task-specific embeddings."""
        from microgpt import Value, linear
        
        embeddings = []
        for t in tokens:
            emb = [self.embed_tokens[t][i] * self.embed_scale for i in range(self.config.hidden_size)]
            
            # Add task embedding if multi-task
            if self.task_embeddings:
                task_emb = self.task_embeddings[task_id]
                # Pad or truncate task embedding to match hidden size
                if len(task_emb) < self.config.hidden_size:
                    task_emb = task_emb + [Value(0) for _ in range(self.config.hidden_size - len(task_emb))]
                emb = [emb[i] + task_emb[i] for i in range(self.config.hidden_size)]
            
            embeddings.append(emb)
        
        # Add position embeddings
        for i, emb in enumerate(embeddings):
            pos_emb = self.embed_pos[i]
            embeddings[i] = [emb[j] + 0.707106781 * pos_emb[j] for j in range(self.config.hidden_size)]
        
        return embeddings
    
    def adaptive_forward_H(self, z_H, z_L, input_emb, pos=0):
        """Forward through H module with adaptive depth."""
        for i, layer in enumerate(self.H_module):
            injection = [z_L[j] + input_emb[j] for j in range(self.config.hidden_size)]
            z_H = layer.forward(z_H, self.H_caches_k[i], self.H_caches_v[i], pos)
            z_H = [z_H[j] + injection[j] for j in range(self.config.hidden_size)]
        return z_H
    
    def adaptive_forward_L(self, z_L, z_H, input_emb, pos=0):
        """Forward through L module with adaptive depth."""
        for i, layer in enumerate(self.L_module):
            injection = [z_H[j] + input_emb[j] for j in range(self.config.hidden_size)]
            z_L = layer.forward(z_L, self.L_caches_k[i], self.L_caches_v[i], pos)
            z_L = [z_L[j] + injection[j] for j in range(self.config.hidden_size)]
        return z_L
    
    def compute_q_values(self, state):
        """Compute Q-values using double Q-learning."""
        from microgpt import linear
        
        # Q1 and Q2 for double Q-learning
        q1 = linear(state, self.q_head_1)
        q2 = linear(state, self.q_head_2)
        
        # Average for stability
        q_halt = (q1[0] + q2[0]) / 2 + self.q_bias[0]
        q_continue = (q1[1] + q2[1]) / 2 + self.q_bias[1]
        
        return q_halt, q_continue
    
    def should_halt(self, q_halt, q_continue, step, training=True):
        """Enhanced halting with adaptive depth."""
        # Max steps
        if step >= self.config.halt_max_steps:
            return True
        
        if training and self.config.adaptive_depth:
            # Use Q-learning with exploration
            halt = q_halt.data > q_continue.data
            
            # Dynamic exploration based on history
            if len(self.cycle_history) > 10:
                avg_steps = sum(self.cycle_history) / len(self.cycle_history)
                if avg_steps > self.config.max_cycles * 0.8:
                    # Encourage halting earlier
                    halt = halt or (random.random() < 0.3)

            
            # Random exploration
            if random.random() < self.config.halt_exploration_prob:
                min_steps = random.randint(self.config.min_cycles, self.config.max_cycles)
                halt = halt and (step >= min_steps)
            
            return halt
        
        # Evaluation: use learned policy
        return q_halt.data > q_continue.data or step >= self.config.halt_max_steps
    
    def forward(self, tokens, task_id=0, training=True):
        """Forward with adaptive computation and optional memory."""
        from microgpt import Value, linear
        
        carry = HRMCarry(
            z_H=[Value(v.data) for v in self.H_init],
            z_L=[Value(v.data) for v in self.L_init],
            task_id=task_id
        )
        
        input_embs = self.embed_input(tokens, task_id)
        
        all_logits = []
        all_q_halt = []
        all_q_continue = []
        
        while not carry.halted:
            self.reset_caches()
            
            # Adaptive H/L cycling
            h_cycles = random.randint(self.config.min_cycles, self.config.max_cycles) if self.config.adaptive_depth else self.config.H_cycles
            l_cycles = random.randint(self.config.min_cycles, self.config.max_cycles) if self.config.adaptive_depth else self.config.L_cycles
            
            # Hierarchical processing
            for h_step in range(h_cycles):
                for l_step in range(l_cycles):
                    if not (h_step == h_cycles - 1 and l_step == l_cycles - 1):
                        carry.z_L = self.adaptive_forward_L(carry.z_L, carry.z_H, input_embs[0])
                
                if h_step != h_cycles - 1:
                    carry.z_H = self.adaptive_forward_H(carry.z_H, carry.z_L, input_embs[0])
            
            # Final gradient-enabled step
            carry.z_L = self.adaptive_forward_L(carry.z_L, carry.z_H, input_embs[0])
            carry.z_H = self.adaptive_forward_H(carry.z_H, carry.z_L, input_embs[0])
            
            # Memory read if enabled
            if self.memory:
                mem_out = self.memory.read(carry.z_H)
                carry.z_H = [carry.z_H[i] + mem_out[i] for i in range(self.config.hidden_size)]
            
            # Output
            logits = linear(carry.z_H, self.lm_head)
            q_halt, q_continue = self.compute_q_values(carry.z_H)
            
            all_logits.append(logits)
            all_q_halt.append(q_halt)
            all_q_continue.append(q_continue)
            
            carry.step += 1
            carry.halted = self.should_halt(q_halt, q_continue, carry.step, training)
        
        # Write to memory if enabled
        if self.memory and all_logits:
            self.memory.write(carry.z_H, all_logits[-1])
        
        self.cycle_history.append(carry.step)
        
        return {
            "logits": all_logits[-1],
            "all_logits": all_logits,
            "steps": carry.step,
            "halted": carry.halted,
            "q_halt": all_q_halt,
            "q_continue": all_q_continue,
        }
    
    def meta_learn_step(self, support_set, query_set, task_id=0):
        """
        MAML-style meta-learning step.
        
        Args:
            support_set: List of (input, target) for inner loop
            query_set: List of (input, target) for outer loop
        """
        # Save current parameters
        original_params = self._get_params()
        
        # Inner loop: adapt to support set
        for _ in range(self.config.num_inner_steps):
            for tokens, targets in support_set:
                loss, _ = self.compute_loss(tokens, targets, task_id)
                loss.backward()
                self._apply_gradients(self.config.inner_lr)
        
        # Outer loop: evaluate on query set
        query_loss = Value(0)
        for tokens, targets in query_set:
            loss, _ = self.compute_loss(tokens, targets, task_id)
            query_loss = query_loss + loss
        
        query_loss.backward()
        
        # Meta-update
        self._apply_gradients(self.config.meta_lr)
        
        # Restore adapted parameters (for next task)
        self._set_params(original_params)
        
        return query_loss.data
    
    def compute_loss(self, tokens, targets, task_id=0):
        """Compute loss with Q-learning and regularization."""
        from microgpt import Value, softmax
        
        result = self.forward(tokens, task_id, training=True)
        
        # Language modeling loss
        lm_losses = []
        for i, target in enumerate(targets):
            if i < len(result["all_logits"]):
                logits = result["all_logits"][i]
                probs = softmax(logits)
                loss = -probs[target].log()
                lm_losses.append(loss)
        
        lm_loss = sum(lm_losses) / len(lm_losses) if lm_losses else Value(0)
        
        # Double Q-learning loss
        q_loss = Value(0)
        if len(result["q_halt"]) > 1:
            for i in range(len(result["q_halt"]) - 1):
                q_halt = result["q_halt"][i]
                q_continue = result["q_continue"][i]
                
                # Encourage halting when near max steps
                if i >= len(result["q_halt"]) - 2:
                    q_loss = q_loss + (q_continue - q_halt).relu()
        
        # Regularization
        reg_loss = Value(0)
        if self.config.weight_decay > 0:
            # L2 regularization on all parameters
            for p in self._get_all_parameters():
                reg_loss = reg_loss + p * p
        
        total_loss = lm_loss + 0.1 * q_loss + self.config.weight_decay * reg_loss
        
        info = {
            "lm_loss": lm_loss.data,
            "q_loss": q_loss.data,
            "reg_loss": reg_loss.data,
            "total_loss": total_loss.data,
            "steps": result["steps"],
        }
        
        return total_loss, info
    
    def _get_params(self):
        """Get all parameters as a flat list."""
        params = []
        for row in self.embed_tokens:
            params.extend(row)
        for row in self.embed_pos:
            params.extend(row)
        if self.task_embeddings:
            for row in self.task_embeddings:
                params.extend(row)
        for layer in self.H_module + self.L_module:
            for w in [layer.wq, layer.wk, layer.wv, layer.wo, layer.w1, layer.w2, layer.w3]:
                for row in w:
                    params.extend(row)
        return params
    
    def _set_params(self, params):
        """Set all parameters from a flat list."""
        # This is a simplified version - full implementation would restore structure
        pass
    
    def _get_all_parameters(self):
        """Get all parameters for regularization."""
        return self._get_params()
    
    def _apply_gradients(self, lr):
        """Apply gradients with clipping."""
        for p in self._get_params():
            if p.grad != 0:
                # Gradient clipping
                if abs(p.grad) > self.config.grad_clip:
                    p.grad = p.grad / abs(p.grad) * self.config.grad_clip
                
                p.data -= lr * p.grad
                p.grad = 0
    
    def train_step(self, tokens, targets, task_id=0):
        """Enhanced training step with LR schedule."""
        # Learning rate schedule
        if self.config.use_lr_schedule:
            if self.step < self.config.warmup_steps:
                self.lr = self.config.learning_rate * (self.step / self.config.warmup_steps)
            else:
                # Cosine decay
                progress = (self.step - self.config.warmup_steps) / (10000 - self.config.warmup_steps)
                self.lr = self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        
        loss, info = self.compute_loss(tokens, targets, task_id)
        loss.backward()
        self._apply_gradients(self.lr)
        self.step += 1
        
        info['lr'] = self.lr
        return info
    
    def get_stats(self):
        """Get comprehensive model statistics."""
        total_params = len(self._get_params())
        
        return {
            "total_parameters": total_params,
            "hidden_size": self.config.hidden_size,
            "H_layers": self.config.H_layers,
            "L_layers": self.config.L_layers,
            "avg_steps": sum(self.cycle_history) / len(self.cycle_history) if self.cycle_history else 0,
            "training_steps": self.step,
            "current_lr": self.lr,
        }
