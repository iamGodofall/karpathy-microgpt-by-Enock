"""
Advanced features for microgpt:
- Mixed precision training
- Gradient accumulation
- Beam search generation
- Repetition penalty
- Contrastive search
"""

import math
import random
from typing import List, Tuple, Optional, Dict
from collections import Counter
from model import Value, GPT, softmax


class MixedPrecisionTrainer:
    """Simulated mixed precision training (loss scaling for numerical stability)."""
    
    def __init__(self, loss_scale: float = 2.0 ** 16):
        self.loss_scale = loss_scale
        self.max_scale = 2.0 ** 24
        self.min_scale = 2.0 ** 8
    
    def scale_loss(self, loss: Value) -> Value:
        """Scale up loss for gradient computation."""
        return loss * self.loss_scale
    
    def unscale_gradients(self, params: List[Value]):
        """Unscale gradients after backward pass."""
        for p in params:
            p.grad /= self.loss_scale
    
    def update_scale(self, has_inf_or_nan: bool):
        """Adjust loss scale based on gradient health."""
        if has_inf_or_nan:
            self.loss_scale = max(self.loss_scale / 2, self.min_scale)
        else:
            self.loss_scale = min(self.loss_scale * 2, self.max_scale)


class GradientAccumulator:
    """Accumulate gradients over multiple steps for effective larger batch size."""
    
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads: Dict[int, float] = {}
        self.step_count = 0
    
    def accumulate(self, params: List[Value]):
        """Accumulate gradients from current step."""
        for i, p in enumerate(params):
            if i not in self.accumulated_grads:
                self.accumulated_grads[i] = 0.0
            self.accumulated_grads[i] += p.grad
    
    def step(self, params: List[Value]):
        """Apply accumulated gradients if ready."""
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            # Apply accumulated gradients
            for i, p in enumerate(params):
                p.grad = self.accumulated_grads.get(i, 0.0) / self.accumulation_steps
                self.accumulated_grads[i] = 0.0
            return True  # Ready for optimizer step
        
        # Zero out current gradients for next accumulation
        for p in params:
            p.grad = 0
        
        return False  # Not ready yet


class BeamSearchDecoder:
    """Beam search for higher quality generation."""
    
    def __init__(self, beam_width: int = 5, max_length: int = 50):
        self.beam_width = beam_width
        self.max_length = max_length
    
    def decode(self, model: GPT, start_token: int, 
               temperature: float = 1.0) -> Tuple[List[int], float]:
        """
        Beam search decoding.
        Returns: (best_sequence, score)
        """
        model.set_training(False)
        
        # Each beam: (sequence, score, keys, values)
        beams = [([start_token], 0.0, [[] for _ in range(model.n_layer)], 
                 [[] for _ in range(model.n_layer)])]
        
        for _ in range(self.max_length):
            new_beams = []
            
            for seq, score, keys, values in beams:
                if seq[-1] == model.vocab_size - 1:  # BOS token
                    new_beams.append((seq, score, keys, values))
                    continue
                
                # Get logits for last token
                logits = model.forward(seq[-1], len(seq) - 1, keys, values)
                
                # Apply temperature and get probabilities
                scaled = [l / temperature for l in logits]
                probs = softmax(scaled)
                probs_data = [p.data for p in probs]
                
                # Get top-k candidates
                top_k = min(self.beam_width, len(probs_data))
                top_indices = sorted(range(len(probs_data)), 
                                   key=lambda i: probs_data[i], reverse=True)[:top_k]
                
                for idx in top_indices:
                    new_seq = seq + [idx]
                    # Log probability (using log for numerical stability)
                    new_score = score + math.log(probs_data[idx] + 1e-10)
                    # Deep copy keys and values
                    new_keys = [k.copy() for k in keys]
                    new_values = [v.copy() for v in values]
                    new_beams.append((new_seq, new_score, new_keys, new_values))
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_width]
        
        # Return best sequence (excluding start token)
        best_seq, best_score, _, _ = beams[0]
        return best_seq[1:], best_score


class RepetitionPenaltyLogitsProcessor:
    """Penalize repeated tokens to reduce repetition."""
    
    def __init__(self, penalty: float = 1.2):
        self.penalty = penalty
    
    def __call__(self, logits: List[Value], generated_tokens: List[int]) -> List[Value]:
        """Apply repetition penalty to logits."""
        if not generated_tokens or self.penalty == 1.0:
            return logits
        
        # Count token frequencies
        token_counts = Counter(generated_tokens)
        
        # Apply penalty
        penalized = []
        for i, logit in enumerate(logits):
            if i in token_counts:
                # Divide probability by penalty (subtract in log space)
                penalty_factor = self.penalty ** token_counts[i]
                penalized.append(Value(logit.data / penalty_factor))
            else:
                penalized.append(logit)
        
        return penalized


class ContrastiveSearchDecoder:
    """
    Contrastive search: balance between confidence and diversity.
    From: "A Contrastive Framework for Neural Text Generation" (Su et al., 2022)
    """
    
    def __init__(self, k: int = 5, alpha: float = 0.6):
        self.k = k  # Top-k candidates to consider
        self.alpha = alpha  # Balance between confidence and diversity
    
    def decode(self, model: GPT, start_token: int, 
               max_length: int = 50) -> List[int]:
        """
        Contrastive search decoding.
        """
        model.set_training(False)
        
        generated = [start_token]
        keys = [[] for _ in range(model.n_layer)]
        values = [[] for _ in range(model.n_layer)]
        
        for pos in range(max_length):
            # Get logits
            logits = model.forward(generated[-1], pos, keys, values)
            probs = softmax(logits)
            probs_data = [p.data for p in probs]
            
            # Get top-k candidates
            top_k_indices = sorted(range(len(probs_data)), 
                                  key=lambda i: probs_data[i], 
                                  reverse=True)[:self.k]
            
            if not top_k_indices:
                break
            
            # Select token that maximizes: (1-alpha) * log_prob - alpha * max_similarity
            best_token = top_k_indices[0]
            best_score = float('-inf')
            
            for idx in top_k_indices:
                # Confidence score
                confidence = math.log(probs_data[idx] + 1e-10)
                
                # Similarity penalty (simplified - in practice use cosine similarity)
                # Here we use a simple heuristic: penalize if token appeared recently
                similarity_penalty = 0.0
                if idx in generated[-5:]:  # Last 5 tokens
                    similarity_penalty = 0.5 * (5 - generated[-5:][::-1].index(idx))
                
                score = (1 - self.alpha) * confidence - self.alpha * similarity_penalty
                
                if score > best_score:
                    best_score = score
                    best_token = idx
            
            generated.append(best_token)
            
            if best_token == model.vocab_size - 1:  # BOS token
                break
        
        return generated[1:]  # Exclude start token


class TopA_Sampling:
    """Top-a sampling: dynamic threshold based on maximum probability."""
    
    def __init__(self, a: float = 0.2):
        self.a = a
    
    def sample(self, logits: List[Value], temperature: float = 1.0) -> int:
        """Sample using top-a threshold."""
        scaled = [l / temperature for l in logits]
        probs = softmax(scaled)
        probs_data = [p.data for p in probs]
        
        max_prob = max(probs_data)
        threshold = self.a * max_prob
        
        # Filter and renormalize
        filtered = [(i, p) for i, p in enumerate(probs_data) if p >= threshold]
        if not filtered:
            return random.choices(range(len(probs_data)), weights=probs_data)[0]
        
        indices, probs_filtered = zip(*filtered)
        total = sum(probs_filtered)
        probs_normalized = [p / total for p in probs_filtered]
        
        return random.choices(indices, weights=probs_normalized)[0]


class MirostatSampling:
    """
    Mirostat sampling: maintain constant perplexity during generation.
    From: "Mirostat: A Neural Text Decoding Algorithm" (Basu et al., 2021)
    """
    
    def __init__(self, target_perplexity: float = 8.0, tau: float = 3.0, 
                 max_surprise: float = 5.0):
        self.target_perplexity = target_perplexity
        self.tau = tau  # Learning rate for adjustment
        self.max_surprise = max_surprise
        self.current_surprise = 0.0
    
    def sample(self, logits: List[Value]) -> int:
        """Sample with mirostat control."""
        # Calculate surprise for each token
        probs = softmax(logits)
        probs_data = [p.data for p in probs]
        
        # Filter by maximum surprise
        log_probs = [math.log(p + 1e-10) for p in probs_data]
        surprises = [-lp for lp in log_probs]
        
        valid_indices = [i for i, s in enumerate(surprises) 
                        if s <= self.current_surprise + self.max_surprise]
        
        if not valid_indices:
            valid_indices = list(range(len(probs_data)))
        
        # Sample from valid tokens
        valid_probs = [probs_data[i] for i in valid_indices]
        total = sum(valid_probs)
        valid_probs = [p / total for p in valid_probs]
        
        chosen_idx = random.choices(valid_indices, weights=valid_probs)[0]
        
        # Update surprise estimate
        observed_surprise = surprises[chosen_idx]
        self.current_surprise = self.tau * observed_surprise + \
                               (1 - self.tau) * math.log(self.target_perplexity)
        
        return chosen_idx
