"""
Full integration of OpenClaw architecture into microgpt.
Combines microgpt's pure Python GPT with OpenClaw's session management,
auth profiles, and tool systems.
"""

import os
import json
import math
import random
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Import microgpt components
from microgpt import Value, softmax, cross_entropy, rmsnorm, linear, gelu, swiglu

# Import OpenClaw adapter
from openclaw_adapter import (
    OpenClawAdapter, Session, SessionManager, 
    AuthProfile, AuthProfileStore, ThinkLevel
)


@dataclass
class MicroGPTConfig:
    """Configuration for microgpt with OpenClaw features."""
    # Model architecture
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.0
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 4
    num_steps: int = 5000
    
    # OpenClaw features
    max_context_tokens: int = 2048
    compaction_threshold: float = 0.8
    enable_fallback: bool = True
    
    # Generation
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9


class MicroGPTWithOpenClaw:
    """
    microgpt enhanced with OpenClaw architecture patterns.
    """
    
    def __init__(self, config: Optional[MicroGPTConfig] = None, storage_dir: str = ".microgpt"):
        self.config = config or MicroGPTConfig()
        self.adapter = OpenClawAdapter(storage_dir)
        
        # Model parameters (initialized on first use)
        self.params: Dict[str, List[List[Value]]] = {}
        self.vocab_size: int = 0
        self.uchars: List[str] = []
        self.BOS: int = 0
        
        # Training state
        self.step: int = 0
        self.best_loss: float = float('inf')
        
        # Initialize if data exists
        self._init_data()
    
    def _init_data(self):
        """Initialize dataset and tokenizer."""
        if not os.path.exists('input.txt'):
            import urllib.request
            names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
            urllib.request.urlretrieve(names_url, 'input.txt')
        
        with open('input.txt') as f:
            docs = [line.strip() for line in f if line.strip()]
        
        random.seed(42)
        random.shuffle(docs)
        self.docs = docs
        
        # Tokenizer
        self.uchars = sorted(set(''.join(docs)))
        self.BOS = len(self.uchars)
        self.vocab_size = len(self.uchars) + 1
        
        print(f"Loaded {len(docs)} documents, vocab size: {self.vocab_size}")
    
    def _init_params(self):
        """Initialize model parameters."""
        C = self.config
        n_embd, n_head, n_layer, block_size = C.n_embd, C.n_head, C.n_layer, C.block_size
        head_dim = n_embd // n_head
        vocab_size = self.vocab_size
        
        # Embeddings
        self.params['wte'] = [[Value(random.gauss(0, 0.02)) for _ in range(n_embd)] for _ in range(vocab_size)]
        self.params['wpe'] = [[Value(random.gauss(0, 0.02)) for _ in range(n_embd)] for _ in range(block_size)]
        
        # Transformer layers
        for i in range(n_layer):
            # Attention
            self.params[f'attn_{i}_wq'] = [[Value(random.gauss(0, 0.02)) for _ in range(n_embd)] for _ in range(n_embd)]
            self.params[f'attn_{i}_wk'] = [[Value(random.gauss(0, 0.02)) for _ in range(n_embd)] for _ in range(n_embd)]
            self.params[f'attn_{i}_wv'] = [[Value(random.gauss(0, 0.02)) for _ in range(n_embd)] for _ in range(n_embd)]
            self.params[f'attn_{i}_wo'] = [[Value(random.gauss(0, 0.02)) for _ in range(n_embd)] for _ in range(n_embd)]
            
            # MLP (SwiGLU)
            hidden_dim = 4 * n_embd
            self.params[f'mlp_{i}_w1'] = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_dim)] for _ in range(n_embd)]
            self.params[f'mlp_{i}_w2'] = [[Value(random.gauss(0, 0.02)) for _ in range(n_embd)] for _ in range(hidden_dim)]
            self.params[f'mlp_{i}_w3'] = [[Value(random.gauss(0, 0.02)) for _ in range(hidden_dim)] for _ in range(n_embd)]
            
            # Layer norms
            self.params[f'ln1_{i}_g'] = [Value(1.0) for _ in range(n_embd)]
            self.params[f'ln2_{i}_g'] = [Value(1.0) for _ in range(n_embd)]
        
        # Output
        self.params['ln_f_g'] = [Value(1.0) for _ in range(n_embd)]
        self.params['lm_head'] = [[Value(random.gauss(0, 0.02)) for _ in range(vocab_size)] for _ in range(n_embd)]
        
        print(f"Initialized {len(self.params)} parameter matrices")
    
    def gpt(self, token_id: int, pos_id: int, keys_cache: List, values_cache: List) -> List[Value]:
        """Forward pass through GPT model."""
        C = self.config
        n_embd, n_head, n_layer = C.n_embd, C.n_head, C.n_layer
        head_dim = n_embd // n_head
        
        # Embeddings
        x = [self.params['wte'][token_id][i] + self.params['wpe'][pos_id][i] for i in range(n_embd)]
        
        # Transformer layers
        for layer in range(n_layer):
            # Attention
            ln1_out = rmsnorm(x, self.params[f'ln1_{layer}_g'])
            
            # QKV projections
            q = linear(ln1_out, self.params[f'attn_{layer}_wq'])
            k = linear(ln1_out, self.params[f'attn_{layer}_wk'])
            v = linear(ln1_out, self.params[f'attn_{layer}_wv'])
            
            # Reshape for multi-head
            q_heads = [q[i*head_dim:(i+1)*head_dim] for i in range(n_head)]
            k_heads = [k[i*head_dim:(i+1)*head_dim] for i in range(n_head)]
            v_heads = [v[i*head_dim:(i+1)*head_dim] for i in range(n_head)]
            
            # Causal self-attention
            attn_outs = []
            for h in range(n_head):
                q_h = q_heads[h]
                
                # Compute attention scores
                scores = []
                for t in range(pos_id + 1):
                    if len(keys_cache[layer]) > t:
                        k_t = keys_cache[layer][t][h]
                        score = sum(q_h[i] * k_t[i] for i in range(head_dim)) / (head_dim ** 0.5)
                        scores.append(score)
                
                # Softmax
                probs = softmax(scores)
                
                # Weighted sum
                attn_out = [Value(0.0) for _ in range(head_dim)]
                for t, p in enumerate(probs):
                    if len(values_cache[layer]) > t:
                        v_t = values_cache[layer][t][h]
                        for i in range(head_dim):
                            attn_out[i] = attn_out[i] + p * v_t[i]
                
                attn_outs.extend(attn_out)
            
            # Output projection
            attn_proj = linear(attn_outs, self.params[f'attn_{layer}_wo'])
            x = [x[i] + attn_proj[i] for i in range(n_embd)]
            
            # Cache keys and values for this position
            if len(keys_cache[layer]) <= pos_id:
                keys_cache[layer].append(k_heads)
                values_cache[layer].append(v_heads)
            
            # MLP
            ln2_out = rmsnorm(x, self.params[f'ln2_{layer}_g'])
            mlp_hidden = swiglu(ln2_out, self.params[f'mlp_{layer}_w1'], self.params[f'mlp_{layer}_w3'])
            mlp_out = linear(mlp_hidden, self.params[f'mlp_{layer}_w2'])
            x = [x[i] + mlp_out[i] for i in range(n_embd)]
        
        # Output
        x = rmsnorm(x, self.params['ln_f_g'])
        logits = linear(x, self.params['lm_head'])
        
        return logits
    
    def train_step(self, doc: str) -> float:
        """Single training step on a document."""
        if not self.params:
            self._init_params()
        
        # Tokenize
        tokens = [self.BOS] + [self.uchars.index(ch) for ch in doc if ch in self.uchars] + [self.BOS]
        n = min(self.config.block_size, len(tokens) - 1)
        
        # Forward pass
        keys_cache = [[] for _ in range(self.config.n_layer)]
        values_cache = [[] for _ in range(self.config.n_layer)]
        losses = []
        
        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]
            
            logits = self.gpt(token_id, pos_id, keys_cache, values_cache)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        
        loss = (1 / n) * sum(losses, Value(0))
        
        # Backward pass
        loss.backward()
        
        # Update parameters (AdamW)
        lr = self.config.learning_rate * (1 - self.step / self.config.num_steps)
        
        for name, matrix in self.params.items():
            for row in matrix:
                for p in row:
                    if p.grad != 0:
                        p.data -= lr * p.grad
                        p.grad = 0
        
        self.step += 1
        return loss.data
    
    def generate(
        self,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text with OpenClaw-style sampling."""
        if not self.params:
            self._init_params()
        
        C = self.config
        max_length = max_length or C.block_size
        temperature = temperature or C.temperature
        top_k = top_k or C.top_k
        top_p = top_p or C.top_p
        
        # Start with BOS or encode prompt
        if prompt:
            tokens = [self.uchars.index(ch) for ch in prompt if ch in self.uchars]
            if not tokens:
                tokens = [self.BOS]
        else:
            tokens = [self.BOS]
        
        keys_cache = [[] for _ in range(C.n_layer)]
        values_cache = [[] for _ in range(C.n_layer)]
        
        result = []
        
        for pos_id in range(max_length):
            token_id = tokens[pos_id] if pos_id < len(tokens) else self.BOS
            
            logits = self.gpt(token_id, pos_id, keys_cache, values_cache)
            
            # Apply temperature
            scaled_logits = [l / temperature for l in logits]
            probs = softmax(scaled_logits)
            
            # Top-k filtering
            if top_k > 0:
                prob_values = [p.data for p in probs]
                sorted_probs = sorted(enumerate(prob_values), key=lambda x: x[1], reverse=True)
                top_k_indices = [i for i, _ in sorted_probs[:top_k]]
                
                filtered_probs = [Value(0) for _ in probs]
                for i in top_k_indices:
                    filtered_probs[i] = probs[i]
                probs = softmax(filtered_probs)
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                prob_values = [(i, p.data) for i, p in enumerate(probs)]
                sorted_probs = sorted(prob_values, key=lambda x: x[1], reverse=True)
                
                cumsum = 0
                nucleus_indices = []
                for i, p in sorted_probs:
                    cumsum += p
                    nucleus_indices.append(i)
                    if cumsum >= top_p:
                        break
                
                filtered_probs = [Value(0) for _ in probs]
                for i in nucleus_indices:
                    filtered_probs[i] = probs[i]
                probs = softmax(filtered_probs)
            
            # Sample
            prob_values = [p.data for p in probs]
            token_id = random.choices(range(self.vocab_size), weights=prob_values)[0]
            
            if token_id == self.BOS:
                break
            
            result.append(self.uchars[token_id])
        
        return ''.join(result)
    
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Chat interface with session management.
        OpenClaw-style conversation handling.
        """
        # Get or create session
        session = self.adapter.get_or_create_session(session_id, self.config.max_context_tokens)
        
        # Add system prompt if new session
        if system_prompt and len(session.messages) == 0:
            session.add_message("system", system_prompt)
        
        # Add user message
        session.add_message("user", message)
        
        # Generate response (simplified - in real use, would use context)
        # For now, just use the message as seed
        response = self.generate(prompt=message[:50])
        
        # Add assistant response
        session.add_message("assistant", response)
        
        # Save session
        self.adapter.sessions._save_session(session)
        
        return {
            "response": response,
            "session_id": session.session_id,
            "tokens_used": session.estimate_tokens(),
            "compaction_count": session.compaction_count,
        }
    
    def train(self, num_steps: Optional[int] = None, eval_interval: int = 100):
        """Train the model with progress tracking."""
        num_steps = num_steps or self.config.num_steps
        
        print(f"Training for {num_steps} steps...")
        
        for step in range(num_steps):
            doc = self.docs[step % len(self.docs)]
            loss = self.train_step(doc)
            
            if step % eval_interval == 0:
                print(f"Step {step:5d} | loss: {loss:.4f}")
                
                # Save best model
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_checkpoint("best_model.json")
        
        print(f"Training complete. Best loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(".microgpt/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "step": self.step,
            "best_loss": self.best_loss,
            "config": asdict(self.config),
            "vocab_size": self.vocab_size,
            "uchars": self.uchars,
            "BOS": self.BOS,
            "params": {
                name: [[v.data for v in row] for row in matrix]
                for name, matrix in self.params.items()
            }
        }
        
        path = checkpoint_dir / filename
        path.write_text(json.dumps(checkpoint, indent=2))
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = Path(".microgpt/checkpoints") / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = json.loads(path.read_text())
        
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        self.vocab_size = checkpoint["vocab_size"]
        self.uchars = checkpoint["uchars"]
        self.BOS = checkpoint["BOS"]
        
        # Restore parameters
        self.params = {}
        for name, matrix_data in checkpoint["params"].items():
            self.params[name] = [[Value(v) for v in row] for row in matrix_data]
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model and session statistics."""
        return {
            "model": {
                "step": self.step,
                "best_loss": self.best_loss,
                "vocab_size": self.vocab_size,
                "num_parameters": sum(
                    len(row) for matrix in self.params.values() for row in matrix
                ) if self.params else 0,
            },
            "sessions": {
                "count": len(self.adapter.sessions.sessions),
                "ids": list(self.adapter.sessions.sessions.keys())[:10],  # First 10
            },
            "config": asdict(self.config),
        }


# CLI interface
def main():
    """CLI for microgpt with OpenClaw features."""
    import argparse
    
    parser = argparse.ArgumentParser(description="microgpt with OpenClaw architecture")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--chat", type=str, help="Chat message")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--session", type=str, help="Session ID for chat")
    parser.add_argument("--checkpoint", type=str, help="Load checkpoint")
    parser.add_argument("--save", type=str, help="Save checkpoint after training")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--prompt", type=str, help="Generation prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    # Initialize
    config = MicroGPTConfig(num_steps=args.steps)
    model = MicroGPTWithOpenClaw(config)
    
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    
    if args.train:
        model.train(num_steps=args.steps)
        if args.save:
            model.save_checkpoint(args.save)
    
    elif args.chat:
        result = model.chat(args.chat, session_id=args.session)
        print(f"\nResponse: {result['response']}")
        print(f"Session: {result['session_id']}")
        print(f"Tokens: {result['tokens_used']}")
    
    elif args.generate:
        prompt = args.prompt or ""
        result = model.generate(
            prompt=prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"Generated: {result}")
    
    elif args.stats:
        stats = model.get_stats()
        print(json.dumps(stats, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
