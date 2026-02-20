"""
Advanced training techniques from SOTA models.
Includes Lion optimizer, schedule-free training, Muon, and more.
"""

import math
import random
from typing import List, Optional, Callable
from .model import Value


class LionOptimizer:
    """
    Lion optimizer from Google Brain.
    Memory efficient, often outperforms AdamW.
    Uses only momentum, no second moment.
    """

    def __init__(
        self,
        params: List[Value],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay

        # Momentum buffer only (no second moment like Adam)
        self.m = [0.0] * len(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def step(self, step_num: int):
        for i, p in enumerate(self.params):
            # Weight decay
            if self.weight_decay > 0:
                p.grad += self.weight_decay * p.data

            # Update momentum
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

            # Lion update: sign of momentum
            update = math.copysign(1, self.m[i]) * self.lr

            # Interpolate with second momentum
            update *= self.beta2

            p.data -= update


class ScheduleFreeOptimizer:
    """
    Schedule-Free Learning from Meta.
    No learning rate schedule needed!
    """

    def __init__(
        self, params: List[Value], lr: float = 1.0, momentum: float = 0.9, weight_decay: float = 0.0
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Dual averaging variables
        self.z = [p.data for p in params]  # Auxiliary variable
        self.m = [0.0] * len(params)  # Momentum

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def step(self, step_num: int):
        for i, p in enumerate(self.params):
            # Weight decay
            grad = p.grad
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data

            # Update auxiliary variable
            self.z[i] -= self.lr * grad

            # Momentum update
            self.m[i] = self.momentum * self.m[i] + grad

            # Update parameter
            p.data = self.z[i] - self.lr * self.m[i]


class MuonOptimizer:
    """
    Muon optimizer (Momentum Orthogonalization).
    Orthogonalizes gradients for better optimization.
    """

    def __init__(
        self, params: List[Value], lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

        self.m = [0.0] * len(params)

    def _orthogonalize(self, grad: List[float]) -> List[float]:
        """Orthogonalize gradient (simplified)."""
        # Real implementation uses SVD
        # This is a placeholder
        norm = sum(g**2 for g in grad) ** 0.5
        if norm > 0:
            return [g / norm for g in grad]
        return grad

    def step(self, step_num: int):
        for i, p in enumerate(self.params):
            # Update momentum
            self.m[i] = self.momentum * self.m[i] + p.grad

            # Orthogonalize
            orth_m = self._orthogonalize([self.m[i]])

            # Nesterov update
            if self.nesterov:
                update = self.momentum * orth_m[0] + p.grad
            else:
                update = orth_m[0]

            p.data -= self.lr * update


class SophiaOptimizer:
    """
    Sophia: Second-order clipped optimizer.
    Uses Hessian diagonal approximation.
    """

    def __init__(
        self,
        params: List[Value],
        lr: float = 1e-3,
        betas: tuple = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1e-1,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.rho = rho
        self.weight_decay = weight_decay

        self.m = [0.0] * len(params)
        self.h = [0.0] * len(params)  # Hessian diagonal estimate

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def step(self, step_num: int):
        for i, p in enumerate(self.params):
            # Weight decay
            if self.weight_decay > 0:
                p.grad += self.weight_decay * p.data

            # Update momentum
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

            # Update Hessian estimate (simplified)
            # Real implementation uses Hutchinson's method
            self.h[i] = self.beta2 * self.h[i] + (1 - self.beta2) * (p.grad**2)

            # Clipped update
            h_clip = max(self.h[i], 1e-10)
            update = self.m[i] / max(1, self.rho * h_clip)

            p.data -= self.lr * update


class ChinchillaScaling:
    """
    Chinchilla scaling laws for optimal model/data sizing.
    """

    @staticmethod
    def optimal_params(tokens: int) -> int:
        """
        Given tokens, return optimal parameter count.
        N_optimal = C^0.5 where C is compute in FLOPs
        """
        # Simplified: ~20 tokens per parameter
        return int(tokens / 20)

    @staticmethod
    def optimal_tokens(params: int) -> int:
        """
        Given parameters, return optimal token count.
        """
        return int(params * 20)

    @staticmethod
    def compute_flops(params: int, tokens: int) -> float:
        """Estimate training FLOPs."""
        # ~6 * params * tokens for forward+backward
        return 6 * params * tokens

    @staticmethod
    def recommend_config(compute_budget: float) -> dict:
        """
        Recommend model size and training tokens for compute budget.
        compute_budget in FLOPs
        """
        # From Chinchilla paper: N_opt ∝ C^0.5, D_opt ∝ C^0.5
        # where C is compute budget

        optimal_params = int((compute_budget / 6) ** 0.5)
        optimal_tokens = int((compute_budget / 6) ** 0.5 * 20)

        return {"params": optimal_params, "tokens": optimal_tokens, "compute_flops": compute_budget}


class CurriculumLearning:
    """
    Curriculum learning: start easy, increase difficulty.
    """

    def __init__(self, schedule: str = "linear"):
        self.schedule = schedule
        self.step_count = 0

    def get_difficulty(self, max_difficulty: float) -> float:
        """Get current difficulty level."""
        if self.schedule == "linear":
            return min(1.0, self.step_count / 10000) * max_difficulty
        elif self.schedule == "exponential":
            return max_difficulty * (1 - math.exp(-self.step_count / 5000))
        else:
            return max_difficulty

    def step(self):
        self.step_count += 1


class MixtureOfExperts:
    """
    Mixture of Experts (MoE) layer.
    Used in Mixtral, GPT-4, etc.
    """

    def __init__(
        self, n_experts: int, top_k: int = 2, input_dim: int = 512, hidden_dim: int = 2048
    ):
        self.n_experts = n_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Router
        self.router = [[random.gauss(0, 0.02) for _ in range(n_experts)] for _ in range(input_dim)]

        # Experts (FFN layers)
        self.experts = []
        for _ in range(n_experts):
            w1 = [[random.gauss(0, 0.02) for _ in range(input_dim)] for _ in range(hidden_dim)]
            w2 = [[random.gauss(0, 0.02) for _ in range(hidden_dim)] for _ in range(input_dim)]
            self.experts.append((w1, w2))

    def forward(self, x: List[Value]) -> List[Value]:
        """Forward pass with top-k routing."""
        # Compute routing weights
        router_logits = [sum(r[i] * x[i].data for i in range(len(x))) for r in self.router]

        # Top-k selection
        top_k_indices = sorted(
            range(len(router_logits)), key=lambda i: router_logits[i], reverse=True
        )[: self.top_k]

        # Compute weighted sum of expert outputs
        output = [Value(0.0) for _ in range(len(x))]

        for idx in top_k_indices:
            # Expert forward
            w1, w2 = self.experts[idx]
            hidden = [sum(w1[i][j] * x[j].data for j in range(len(x))) for i in range(len(w1))]
            # ReLU activation
            hidden = [max(0, h) for h in hidden]
            expert_out = [
                sum(w2[i][j] * hidden[j] for j in range(len(hidden))) for i in range(len(x))
            ]

            # Add to output with routing weight
            weight = math.exp(router_logits[idx])
            for i in range(len(output)):
                output[i].data += weight * expert_out[i]

        return output


class SlidingWindowAttention:
    """
    Sliding Window Attention from Mistral.
    Efficient long-context modeling.
    """

    def __init__(self, window_size: int = 4096):
        self.window_size = window_size

    def create_mask(self, seq_len: int) -> List[List[bool]]:
        """Create sliding window attention mask."""
        mask = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                # Attend only to window_size previous tokens
                if j <= i and i - j <= self.window_size:
                    row.append(True)
                else:
                    row.append(False)
            mask.append(row)
        return mask


class SpeculativeDecoding:
    """
    Speculative decoding for faster inference.
    Draft small model, verify with large model.
    """

    def __init__(self, draft_model, target_model, gamma: int = 4):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma  # Number of draft tokens

    def generate(self, prompt_tokens: List[int], max_length: int) -> List[int]:
        """
        Generate with speculative decoding.
        """
        result = list(prompt_tokens)

        while len(result) < max_length:
            # Draft model generates gamma tokens
            draft_tokens = []
            for _ in range(self.gamma):
                # Simplified - would use actual draft model
                next_token = random.randint(0, self.draft_model.vocab_size - 1)
                draft_tokens.append(next_token)

            # Target model verifies
            # Accept tokens until first mismatch
            accepted = 0
            for draft_token in draft_tokens:
                # Verify with target model
                # Simplified - would compute actual probabilities
                if random.random() > 0.1:  # 90% acceptance rate
                    result.append(draft_token)
                    accepted += 1
                else:
                    # Reject - sample from target
                    result.append(random.randint(0, self.target_model.vocab_size - 1))
                    break

            if accepted == self.gamma:
                # All accepted, sample one more from target
                result.append(random.randint(0, self.target_model.vocab_size - 1))

        return result[:max_length]


def get_optimizer(name: str, params: List[Value], lr: float = 0.001):
    """Factory function for optimizers."""
    optimizers = {
        "adam": lambda: __import__("trainer", fromlist=["AdamOptimizer"]).AdamOptimizer(
            params,
            __import__("trainer", fromlist=["TrainingConfig"]).TrainingConfig(learning_rate=lr),
        ),
        "lion": lambda: LionOptimizer(params, lr),
        "schedule_free": lambda: ScheduleFreeOptimizer(params, lr),
        "muon": lambda: MuonOptimizer(params, lr),
        "sophia": lambda: SophiaOptimizer(params, lr),
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizers[name]()
