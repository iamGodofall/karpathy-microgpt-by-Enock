"""
Integration of Hierarchical Reasoning Model (HRM) into microgpt ecosystem.
Combines HRM's adaptive computation with microgpt's pure Python implementation.
"""

import random
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from microgpt import Value, softmax, rmsnorm, linear, gelu
from hrm_adapter import HierarchicalReasoningModel, HRMConfig, HRMCarry


@dataclass
class HRMIntegratedConfig:
    """Configuration combining microgpt and HRM features."""
    # Model architecture
    vocab_size: int = 100
    hidden_size: int = 64
    num_heads: int = 4
    n_layer: int = 2  # Standard transformer layers (optional)
    H_layers: int = 2  # HRM high-level layers
    L_layers: int = 2  # HRM low-level layers
    H_cycles: int = 2
    L_cycles: int = 2
    
    # ACT settings
    use_act: bool = True
    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.1
    
    # Training
    learning_rate: float = 0.001
    num_steps: int = 1000
    batch_size: int = 1
    
    # Generation
    temperature: float = 0.8
    max_length: int = 100
    
    # Data
    max_seq_len: int = 128
    data_path: str = "input.txt"


class HybridGPTWithHRM:
    """
    Hybrid model combining standard GPT with Hierarchical Reasoning.
    
    Can operate in three modes:
    1. Standard GPT: Traditional transformer
    2. HRM mode: Hierarchical reasoning with ACT
    3. Hybrid: Both architectures with learned gating
    """
    
    def __init__(self, config: HRMIntegratedConfig):
        self.config = config
        
        # Initialize HRM component
        hrm_config = HRMConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            H_layers=config.H_layers,
            L_layers=config.L_layers,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            halt_max_steps=config.halt_max_steps,
            halt_exploration_prob=config.halt_exploration_prob,
            learning_rate=config.learning_rate,
            max_seq_len=config.max_seq_len,
        )
        self.hrm = HierarchicalReasoningModel(hrm_config)
        
        # Optional: Standard GPT layers for hybrid mode
        self.use_hybrid = config.n_layer > 0
        
        if self.use_hybrid:
            # Standard transformer layers
            self.layers = []
            for _ in range(config.n_layer):
                layer = {
                    'wq': [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] for _ in range(config.hidden_size)],
                    'wk': [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] for _ in range(config.hidden_size)],
                    'wv': [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] for _ in range(config.hidden_size)],
                    'wo': [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] for _ in range(config.hidden_size)],
                    'w1': [[Value(random.gauss(0, 0.02)) for _ in range(4 * config.hidden_size)] for _ in range(config.hidden_size)],
                    'w2': [[Value(random.gauss(0, 0.02)) for _ in range(config.hidden_size)] for _ in range(4 * config.hidden_size)],
                    'norm1_g': [Value(1.0) for _ in range(config.hidden_size)],
                    'norm2_g': [Value(1.0) for _ in range(config.hidden_size)],
                }
                self.layers.append(layer)
            
            # Gating mechanism to combine HRM and GPT outputs
            self.gate = [Value(0.5) for _ in range(config.hidden_size)]
        
        # Shared embeddings (use HRM's)
        self.embed_tokens = self.hrm.embed_tokens
        self.embed_pos = self.hrm.embed_pos
        self.embed_scale = self.hrm.embed_scale
        
        # Shared output head (use HRM's)
        self.lm_head = self.hrm.lm_head
        
        self.step = 0
    
    def standard_transformer_forward(self, tokens: List[int], pos: int) -> List[Value]:
        """Standard transformer forward pass."""
        # Embeddings
        x = [self.embed_tokens[tokens[pos]][i] * self.embed_scale + self.embed_pos[pos][i] 
             for i in range(self.config.hidden_size)]
        
        # Transformer layers
        for layer in self.layers:
            # Attention
            q = linear(x, layer['wq'])
            k = linear(x, layer['wk'])
            v = linear(x, layer['wv'])
            
            # Simple attention (no KV cache for now)
            head_dim = self.config.hidden_size // self.config.num_heads
            scores = []
            for i in range(self.config.hidden_size):
                score = q[i] * k[i] / (head_dim ** 0.5)
                scores.append(score)
            
            probs = softmax(scores)
            attn_out = [sum(probs[i] * v[i] for i in range(self.config.hidden_size))]
            attn_out = linear(attn_out, layer['wo'])
            
            # Residual + norm
            x = [x[i] + attn_out[i] for i in range(self.config.hidden_size)]
            x = rmsnorm(x, layer['norm1_g'])
            
            # MLP
            h = linear(x, layer['w1'])
            h = [gelu(v) for v in h]
            mlp_out = linear(h, layer['w2'])
            
            # Residual + norm
            x = [x[i] + mlp_out[i] for i in range(self.config.hidden_size)]
            x = rmsnorm(x, layer['norm2_g'])
        
        return x
    
    def forward_hrm(self, tokens: List[int], puzzle_id: Optional[int] = None) -> Dict[str, Any]:
        """Forward using HRM with ACT."""
        return self.hrm.forward(tokens, puzzle_id, training=True)
    
    def forward_hybrid(self, tokens: List[int], pos: int, puzzle_id: Optional[int] = None) -> List[Value]:
        """
        Hybrid forward combining standard transformer and HRM.
        Uses gating to blend outputs.
        """
        # Standard transformer output
        gpt_out = self.standard_transformer_forward(tokens, pos)
        
        # HRM output (single step)
        result = self.hrm.forward(tokens, puzzle_id, training=True)
        hrm_out = result["logits"]
        
        # Gate combination
        if self.use_hybrid:
            combined = []
            for i in range(self.config.hidden_size):
                g = self.gate[i].data
                val = g * gpt_out[i] + (1 - g) * hrm_out[i]
                combined.append(val)
            return combined
        
        return hrm_out
    
    def generate(
        self,
        prompt: str,
        tokenizer: Any,
        max_length: int = None,
        temperature: float = None,
        use_hrm: bool = True,
    ) -> str:
        """
        Generate text using HRM or hybrid mode.
        
        Args:
            prompt: Input text
            tokenizer: Tokenizer with encode/decode methods
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            use_hrm: Whether to use HRM (True) or hybrid (False)
        
        Returns:
            Generated text
        """
        if max_length is None:
            max_length = self.config.max_length
        if temperature is None:
            temperature = self.config.temperature
        
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        
        generated = tokens.copy()
        
        for i in range(max_length):
            if use_hrm:
                # HRM generation with ACT
                result = self.hrm.forward(generated, training=False)
                logits = result["logits"]
            else:
                # Hybrid mode
                logits = self.forward_hybrid(generated, len(generated) - 1)
            
            # Sample
            scaled = [l / temperature for l in logits]
            probs = softmax(scaled)
            prob_values = [p.data for p in probs]
            
            next_token = random.choices(range(self.config.vocab_size), weights=prob_values)[0]
            generated.append(next_token)
            
            # Stop on EOS (token 0)
            if next_token == 0:
                break
        
        return tokenizer.decode(generated)
    
    def train_step_hrm(
        self,
        tokens: List[int],
        targets: List[int],
        puzzle_id: Optional[int] = None
    ) -> Dict[str, float]:
        """Training step using HRM loss."""
        return self.hrm.train_step(tokens, targets, puzzle_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined model statistics."""
        hrm_stats = self.hrm.get_stats()
        
        if self.use_hybrid:
            gpt_params = sum(
                len(layer['wq']) * len(layer['wq'][0]) +
                len(layer['wk']) * len(layer['wk'][0]) +
                len(layer['wv']) * len(layer['wv'][0]) +
                len(layer['wo']) * len(layer['wo'][0]) +
                len(layer['w1']) * len(layer['w1'][0]) +
                len(layer['w2']) * len(layer['w2'][0])
                for layer in self.layers
            )
            hrm_stats['gpt_parameters'] = gpt_params
            hrm_stats['total_parameters'] += gpt_params
        
        hrm_stats['mode'] = 'hybrid' if self.use_hybrid else 'hrm_only'
        return hrm_stats


class HRMTrainer:
    """Trainer for HRM models."""
    
    def __init__(self, model: HybridGPTWithHRM, config: HRMIntegratedConfig):
        self.model = model
        self.config = config
    
    def train(
        self,
        dataset: List[Tuple[List[int], List[int]]],
        num_steps: int = None,
        eval_interval: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Train on dataset.
        
        Args:
            dataset: List of (input_tokens, target_tokens) pairs
            num_steps: Number of training steps
            eval_interval: Evaluate every N steps
        
        Returns:
            Training history
        """
        if num_steps is None:
            num_steps = self.config.num_steps
        
        history = []
        
        for step in range(num_steps):
            # Sample batch
            tokens, targets = random.choice(dataset)
            
            # Train step
            info = self.model.train_step_hrm(tokens, targets)
            info['step'] = step
            
            history.append(info)
            
            # Log
            if step % 10 == 0:
                print(f"Step {step}: loss={info['total_loss']:.4f}, "
                      f"lm_loss={info['lm_loss']:.4f}, "
                      f"steps={info['steps']}")
            
            # Eval
            if step % eval_interval == 0:
                print(f"\n--- Step {step} Stats ---")
                stats = self.model.get_stats()
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                print()
        
        return history


# Simple character-level tokenizer for testing
class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.chars)}  # 0 reserved for padding/EOS
        self.stoi['<pad>'] = 0
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join(self.itos.get(t, '') for t in tokens)


def create_demo_dataset(text: str, seq_len: int = 10) -> List[Tuple[List[int], List[int]]]:
    """Create simple next-token prediction dataset."""
    tokenizer = CharTokenizer(text)
    
    data = []
    for i in range(0, len(text) - seq_len, seq_len):
        chunk = text[i:i + seq_len + 1]
        tokens = tokenizer.encode(chunk)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        data.append((input_tokens, target_tokens))
    
    return data, tokenizer


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("HRM + microgpt Integration Demo")
    print("=" * 60)
    
    # Simple demo text
    text = "hello world this is a test of hierarchical reasoning with adaptive computation"
    
    # Create dataset
    data, tokenizer = create_demo_dataset(text, seq_len=5)
    print(f"\nDataset size: {len(data)} sequences")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Config
    config = HRMIntegratedConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_heads=4,
        H_layers=1,
        L_layers=1,
        H_cycles=2,
        L_cycles=2,
        halt_max_steps=4,
        num_steps=100,
    )
    
    # Create model
    print("\nInitializing HRM model...")
    model = HybridGPTWithHRM(config)
    
    # Stats
    stats = model.get_stats()
    print("\nModel Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Train
    print("\nTraining...")
    trainer = HRMTrainer(model, config)
    history = trainer.train(data, num_steps=50, eval_interval=25)
    
    # Generate
    print("\n" + "=" * 60)
    print("Generation Test")
    print("=" * 60)
    
    prompt = "hello"
    print(f"\nPrompt: '{prompt}'")
    
    result = model.generate(prompt, tokenizer, max_length=10, use_hrm=True)
    print(f"Generated: '{result}'")
    
    print("\n" + "=" * 60)
    print("HRM Integration Complete!")
    print("=" * 60)
