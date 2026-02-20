"""
Hierarchical Reasoning Model (HRM) Adaptation for microgpt.
Pure Python implementation of the hierarchical recurrent architecture
with adaptive computation time (ACT).

Based on: "Hierarchical Reasoning Model" (Wang et al., 2025)
arXiv:2506.21734
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from microgpt import Value, softmax, rmsnorm, linear, gelu


@dataclass
class HRMConfig:
    """Configuration for Hierarchical Reasoning Model."""
    # Architecture
    vocab_size: int = 100
    hidden_size: int = 128
    num_heads: int = 4
    H_layers: int = 2  # High-level (slow) layers
    L_layers: int = 2  # Low-level (fast) layers
    H_cycles: int = 3  # High-level iterations
    L_cycles: int = 3  # Low-level iterations per H cycle
    
    # ACT (Adaptive Computation Time)
    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.1
    
    # Training
    learning_rate: float = 0.001
    gamma: float = 0.99  # Q-learning discount
    
    # Embeddings
    max_seq_len: int = 128
    puzzle_emb_ndim: int = 0  # For puzzle-specific tasks


@dataclass
class HRMCarry:
    """Carry state for HRM recurrence."""
    z_H: List[Value]  # High-level state
    z_L: List[Value]  # Low-level state
    step: int = 0
    halted: bool = False


class HRMBlock:
    """Single HRM transformer block with attention and SwiGLU."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
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
        
        # Layer norm gains
        self.norm1_g = [Value(1.0) for _ in range(hidden_size)]
        self.norm2_g = [Value(1.0) for _ in range(hidden_size)]
    
    def attention(self, x: List[Value], cache_k: List, cache_v: List, pos: int) -> List[Value]:
        """Multi-head self-attention with caching."""
        # QKV projections
        q = linear(x, self.wq)
        k = linear(x, self.wk)
        v = linear(x, self.wv)
        
        # Reshape for multi-head
        q_heads = [q[i*self.head_dim:(i+1)*self.head_dim] for i in range(self.num_heads)]
        k_heads = [k[i*self.head_dim:(i+1)*self.head_dim] for i in range(self.num_heads)]
        v_heads = [v[i*self.head_dim:(i+1)*self.head_dim] for i in range(self.num_heads)]
        
        # Cache keys and values
        cache_k.append(k_heads)
        cache_v.append(v_heads)
        
        # Attention scores
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
    
    def swiglu(self, x: List[Value]) -> List[Value]:
        """SwiGLU activation."""
        # x1 = x @ w1
        x1 = linear(x, self.w1)
        # x3 = x @ w3
        x3 = linear(x, self.w3)
        # SwiGLU: silu(x1) * x3
        # Approximate silu with: x * sigmoid(x)
        silu_x1 = [x1[i] * (1 / (1 + (-x1[i]).exp())) for i in range(len(x1))]
        hidden = [silu_x1[i] * x3[i] for i in range(len(silu_x1))]
        # Output projection
        return linear(hidden, self.w2)
    
    def forward(self, x: List[Value], cache_k: List, cache_v: List, pos: int) -> List[Value]:
        """Forward through block with post-norm."""
        # Self-attention with residual and norm
        attn_out = self.attention(x, cache_k, cache_v, pos)
        x = [x[i] + attn_out[i] for i in range(self.hidden_size)]
        x = rmsnorm(x, self.norm1_g)
        
        # MLP with residual and norm
        mlp_out = self.swiglu(x)
        x = [x[i] + mlp_out[i] for i in range(self.hidden_size)]
        x = rmsnorm(x, self.norm2_g)
        
        return x


class HRMReasoningModule:
    """High or Low level reasoning module."""
    
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int):
        self.layers = [HRMBlock(hidden_size, num_heads) for _ in range(num_layers)]
        self.caches_k = [[] for _ in range(num_layers)]
        self.caches_v = [[] for _ in range(num_layers)]
    
    def forward(self, x: List[Value], injection: Optional[List[Value]] = None, pos: int = 0) -> List[Value]:
        """Forward through all layers with optional input injection."""
        if injection:
            x = [x[i] + injection[i] for i in range(len(x))]
        
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, self.caches_k[i], self.caches_v[i], pos)
        
        return x
    
    def reset_cache(self):
        """Clear attention caches."""
        for i in range(len(self.caches_k)):
            self.caches_k[i] = []
            self.caches_v[i] = []


class HierarchicalReasoningModel:
    """
    Hierarchical Reasoning Model with Adaptive Computation Time.
    
    Architecture:
    - High-level module: Slow, abstract planning (H_cycles iterations)
    - Low-level module: Fast, detailed computation (L_cycles per H cycle)
    - ACT: Q-learning based dynamic halting
    """
    
    def __init__(self, config: HRMConfig):
        self.config = config
        
        # Embeddings
        self.embed_tokens = [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] for _ in range(config.vocab_size)]
        self.embed_scale = math.sqrt(config.hidden_size)
        
        # Position embeddings (learned)
        self.embed_pos = [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] for _ in range(config.max_seq_len)]
        
        # Puzzle embeddings (for puzzle tasks)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)  # ceil div
            self.puzzle_emb = [[Value(0) for _ in range(config.puzzle_emb_ndim)] for _ in range(1000)]  # Max 1000 puzzles
        else:
            self.puzzle_emb_len = 0
        
        # Reasoning modules
        self.H_module = HRMReasoningModule(config.H_layers, config.hidden_size, config.num_heads)
        self.L_module = HRMReasoningModule(config.L_layers, config.hidden_size, config.num_heads)
        
        # Initial states
        self.H_init = [Value(random.gauss(0, 1)) for _ in range(config.hidden_size)]
        self.L_init = [Value(random.gauss(0, 1)) for _ in range(config.hidden_size)]
        
        # Output heads
        self.lm_head = [[Value(random.gauss(0, 0.02)) for _ in range(config.vocab_size)] for _ in range(config.hidden_size)]
        
        # Q-head for ACT (halt/continue decision)
        self.q_head = [[Value(0) for _ in range(2)] for _ in range(config.hidden_size)]
        self.q_head_bias = [Value(-5), Value(-5)]  # Initialize to favor continuing
        
        # Optimizer state
        self.step = 0
    
    def embed_input(self, tokens: List[int], puzzle_id: Optional[int] = None) -> List[List[Value]]:
        """Embed input tokens with position and optional puzzle embeddings."""
        batch_size = 1  # Simplified for single sequence
        
        # Token embeddings
        embeddings = []
        for t in tokens:
            emb = [self.embed_tokens[t][i] * self.embed_scale for i in range(self.config.hidden_size)]
            embeddings.append(emb)
        
        # Add puzzle embeddings if provided
        if puzzle_id is not None and self.config.puzzle_emb_ndim > 0:
            puzzle_emb = self.puzzle_emb[puzzle_id]
            # Pad to hidden_size
            pad_count = self.config.hidden_size - len(puzzle_emb)
            if pad_count > 0:
                puzzle_emb = puzzle_emb + [Value(0) for _ in range(pad_count)]
            
            # Prepend puzzle embedding
            embeddings = [puzzle_emb] + embeddings
        
        # Add position embeddings
        for i, emb in enumerate(embeddings):
            pos_emb = self.embed_pos[i]
            embeddings[i] = [emb[j] + 0.707106781 * pos_emb[j] for j in range(self.config.hidden_size)]
        
        return embeddings
    
    def forward_step(self, carry: HRMCarry, tokens: List[int], puzzle_id: Optional[int] = None) -> Tuple[HRMCarry, List[Value], Tuple[Value, Value]]:
        """
        Single forward step with H and L module cycling.
        
        Returns:
            new_carry: Updated carry state
            logits: Output logits
            q_values: (halt_logit, continue_logit) for ACT
        """
        # Get input embeddings
        input_embs = self.embed_input(tokens, puzzle_id)
        
        # Initialize or reset if halted
        if carry.halted:
            z_H = self.H_init.copy()
            z_L = self.L_init.copy()
            carry.step = 0
            carry.halted = False
        else:
            z_H = carry.z_H
            z_L = carry.z_L
        
        # Clear caches
        self.H_module.reset_cache()
        self.L_module.reset_cache()
        
        # Hierarchical forward pass
        for h_step in range(self.config.H_cycles):
            for l_step in range(self.config.L_cycles):
                # Not on last step of cycle
                if not (h_step == self.config.H_cycles - 1 and l_step == self.config.L_cycles - 1):
                    # Low-level processes with high-level guidance + input
                    injection = [z_H[i] + input_embs[0][i] for i in range(self.config.hidden_size)]
                    z_L = self.L_module.forward(z_L, injection, pos=0)
            
            # Not on last H cycle
            if h_step != self.config.H_cycles - 1:
                # High-level updates from low-level state
                z_H = self.H_module.forward(z_H, z_L, pos=0)
        
        # Final gradient-enabled step
        injection = [z_H[i] + input_embs[0][i] for i in range(self.config.hidden_size)]
        z_L = self.L_module.forward(z_L, injection, pos=0)
        z_H = self.H_module.forward(z_H, z_L, pos=0)
        
        # Output
        output = linear(z_H, self.lm_head)
        
        # Q-values for halting decision
        q_logits = linear(z_H, self.q_head)
        q_halt = q_logits[0] + self.q_head_bias[0]
        q_continue = q_logits[1] + self.q_head_bias[1]
        
        # New carry (detach for next step)
        new_carry = HRMCarry(
            z_H=[Value(v.data) for v in z_H],  # Detach
            z_L=[Value(v.data) for v in z_L],  # Detach
            step=carry.step + 1,
            halted=False
        )
        
        return new_carry, output, (q_halt, q_continue)
    
    def should_halt(self, q_halt: Value, q_continue: Value, step: int, training: bool = True) -> bool:
        """Determine if should halt based on Q-values and step count."""
        # Max steps reached
        if step >= self.config.halt_max_steps:
            return True
        
        if training and self.config.halt_max_steps > 1:
            # Q-learning decision
            halt = q_halt.data > q_continue.data
            
            # Exploration: random halting
            if random.random() < self.config.halt_exploration_prob:
                min_steps = random.randint(2, self.config.halt_max_steps)
                halt = halt and (step >= min_steps)
            
            return halt
        
        # Evaluation: always use max steps
        return step >= self.config.halt_max_steps
    
    def forward(self, tokens: List[int], puzzle_id: Optional[int] = None, training: bool = True) -> Dict[str, Any]:
        """
        Full forward pass with adaptive computation.
        
        Returns dict with:
            - logits: Final output logits
            - steps: Number of computation steps
            - halted: Whether halted early
            - q_values: Q-values at each step
        """
        carry = HRMCarry(z_H=self.H_init.copy(), z_L=self.L_init.copy())
        
        all_logits = []
        all_q_halt = []
        all_q_continue = []
        
        while not carry.halted:
            carry, logits, (q_halt, q_continue) = self.forward_step(carry, tokens, puzzle_id)
            
            all_logits.append(logits)
            all_q_halt.append(q_halt)
            all_q_continue.append(q_continue)
            
            # Check halting condition
            carry.halted = self.should_halt(q_halt, q_continue, carry.step, training)
        
        return {
            "logits": all_logits[-1],  # Final logits
            "all_logits": all_logits,
            "steps": carry.step,
            "halted": carry.halted,
            "q_halt": all_q_halt,
            "q_continue": all_q_continue,
        }
    
    def compute_loss(self, tokens: List[int], targets: List[int], puzzle_id: Optional[int] = None) -> Tuple[Value, Dict]:
        """
        Compute loss with ACT Q-learning.
        
        Returns:
            loss: Combined LM + Q-learning loss
            info: Dict with metrics
        """
        # Forward pass
        result = self.forward(tokens, puzzle_id, training=True)
        
        # Language modeling loss
        lm_losses = []
        for i, target in enumerate(targets):
            if i < len(result["all_logits"]):
                logits = result["all_logits"][i]
                probs = softmax(logits)
                loss = -probs[target].log()
                lm_losses.append(loss)
        
        lm_loss = sum(lm_losses) / len(lm_losses) if lm_losses else Value(0)
        
        # Q-learning loss (simplified)
        q_loss = Value(0)
        if len(result["q_halt"]) > 1:
            # Simple Q-learning: encourage halting when correct
            for i in range(len(result["q_halt"]) - 1):
                q_halt = result["q_halt"][i]
                q_continue = result["q_continue"][i]
                
                # Loss: want halt > continue when near end
                if i >= len(result["q_halt"]) - 2:
                    q_loss = q_loss + (q_continue - q_halt).relu()
        
        # Combined loss
        total_loss = lm_loss + 0.1 * q_loss
        
        info = {
            "lm_loss": lm_loss.data,
            "q_loss": q_loss.data,
            "total_loss": total_loss.data,
            "steps": result["steps"],
        }
        
        return total_loss, info
    
    def train_step(self, tokens: List[int], targets: List[int], puzzle_id: Optional[int] = None) -> Dict[str, float]:
        """Single training step."""
        loss, info = self.compute_loss(tokens, targets, puzzle_id)
        
        # Backward
        loss.backward()
        
        # Update parameters (simple SGD)
        lr = self.config.learning_rate
        
        # Update embeddings
        for row in self.embed_tokens:
            for p in row:
                if p.grad != 0:
                    p.data -= lr * p.grad
                    p.grad = 0
        
        # Update reasoning modules
        for module in [self.H_module, self.L_module]:
            for layer in module.layers:
                for param_list in [layer.wq, layer.wk, layer.wv, layer.wo, layer.w1, layer.w2, layer.w3]:
                    for row in param_list:
                        for p in row:
                            if p.grad != 0:
                                p.data -= lr * p.grad
                                p.grad = 0
        
        # Update output heads
        for row in self.lm_head:
            for p in row:
                if p.grad != 0:
                    p.data -= lr * p.grad
                    p.grad = 0
        
        for p in self.q_head_bias:
            if p.grad != 0:
                p.data -= lr * p.grad
                p.grad = 0
        
        self.step += 1
        
        return info
    
    def generate(
        self,
        prompt_tokens: List[int],
        max_length: int = 100,
        puzzle_id: Optional[int] = None,
        temperature: float = 0.8,
    ) -> List[int]:
        """Generate tokens autoregressively."""
        generated = prompt_tokens.copy()
        
        for _ in range(max_length):
            # Forward with ACT
            result = self.forward(generated, puzzle_id, training=False)
            
            # Sample from logits
            logits = result["logits"]
            scaled = [l / temperature for l in logits]
            probs = softmax(scaled)
            
            # Sample
            prob_values = [p.data for p in probs]
            next_token = random.choices(range(self.config.vocab_size), weights=prob_values)[0]
            
            generated.append(next_token)
            
            # Stop token (assuming 0 is EOS)
            if next_token == 0:
                break
        
        return generated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        total_params = (
            len(self.embed_tokens) * len(self.embed_tokens[0]) +
            len(self.embed_pos) * len(self.embed_pos[0]) +
            sum(
                len(layer.wq) * len(layer.wq[0]) +
                len(layer.wk) * len(layer.wk[0]) +
                len(layer.wv) * len(layer.wv[0]) +
                len(layer.wo) * len(layer.wo[0]) +
                len(layer.w1) * len(layer.w1[0]) +
                len(layer.w2) * len(layer.w2[0]) +
                len(layer.w3) * len(layer.w3[0])
                for module in [self.H_module, self.L_module]
                for layer in module.layers
            ) +
            len(self.lm_head) * len(self.lm_head[0])
        )
        
        return {
            "total_parameters": total_params,
            "hidden_size": self.config.hidden_size,
            "H_layers": self.config.H_layers,
            "L_layers": self.config.L_layers,
            "H_cycles": self.config.H_cycles,
            "L_cycles": self.config.L_cycles,
            "training_steps": self.step,
        }
