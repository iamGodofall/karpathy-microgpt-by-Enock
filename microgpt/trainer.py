"""
Training infrastructure for microgpt.
Includes Adam optimizer, learning rate scheduling, and gradient clipping.
"""

import math
from typing import List, Optional, Callable
from .model import Value, GPT
from .config import TrainingConfig


class AdamOptimizer:
    """Adam optimizer with optional weight decay and gradient clipping."""

    def __init__(self, params: List[Value], config: TrainingConfig):
        self.params = params
        self.lr = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps_adam
        self.weight_decay = config.weight_decay
        self.grad_clip = config.grad_clip

        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def zero_grad(self):
        """Reset gradients."""
        for p in self.params:
            p.grad = 0

    def step(self, step_num: int):
        """Perform one optimization step."""
        self.t = step_num + 1

        # Gradient clipping
        if self.grad_clip > 0:
            self._clip_gradients()

        # Weight decay (L2 regularization)
        if self.weight_decay > 0:
            for p in self.params:
                p.grad += self.weight_decay * p.data

        # Adam update
        lr_t = self._get_lr(step_num)

        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= lr_t * m_hat / (v_hat**0.5 + self.eps)

    def _clip_gradients(self):
        """Clip gradients to prevent exploding gradients."""
        total_norm = sum(p.grad**2 for p in self.params) ** 0.5
        clip_coef = self.grad_clip / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.params:
                p.grad *= clip_coef

    def _get_lr(self, step: int) -> float:
        """Get learning rate for current step based on schedule."""
        if hasattr(self, "_lr_schedule_fn"):
            return self._lr_schedule_fn(step)
        return self.lr


class LRScheduler:
    """Learning rate schedulers."""

    @staticmethod
    def linear(initial_lr: float, total_steps: int):
        """Linear decay from initial_lr to 0."""

        def schedule(step: int) -> float:
            return initial_lr * (1 - step / total_steps)

        return schedule

    @staticmethod
    def cosine(initial_lr: float, total_steps: int, min_lr: float = 0.0, warmup_steps: int = 0):
        """Cosine annealing with optional warmup."""

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return initial_lr * step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        return schedule

    @staticmethod
    def constant(initial_lr: float):
        """Constant learning rate."""

        def schedule(step: int) -> float:
            return initial_lr

        return schedule


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Main training loop for GPT."""

    def __init__(
        self, model: GPT, config: TrainingConfig, optimizer: Optional[AdamOptimizer] = None
    ):
        self.model = model
        self.config = config

        if optimizer is None:
            self.optimizer = AdamOptimizer(model.parameters(), config)
        else:
            self.optimizer = optimizer

        # Set up learning rate schedule
        if config.lr_schedule == "linear":
            self.optimizer._lr_schedule_fn = LRScheduler.linear(
                config.learning_rate, config.num_steps
            )
        elif config.lr_schedule == "cosine":
            self.optimizer._lr_schedule_fn = LRScheduler.cosine(
                config.learning_rate, config.num_steps, warmup_steps=config.warmup_steps
            )
        else:
            self.optimizer._lr_schedule_fn = LRScheduler.constant(config.learning_rate)

        self.early_stopping = EarlyStopping() if config.val_split > 0 else None

    def compute_loss(self, tokens: List[int], keys: List[List], values: List[List]) -> Value:
        """Compute cross-entropy loss for a sequence."""
        n = min(self.model.block_size, len(tokens) - 1)
        losses = []

        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]

            logits = self.model.forward(token_id, pos_id, keys, values)
            probs = self._softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        return (1.0 / n) * sum(losses) if losses else Value(0.0)

    def _softmax(self, logits: List[Value]) -> List[Value]:
        """Numerically stable softmax."""
        from .model import softmax

        return softmax(logits)

    def train_step(self, tokens: List[int], step: int) -> float:
        """Single training step."""
        # Reset KV cache for new sequence
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        # Forward pass
        loss = self.compute_loss(tokens, keys, values)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Update parameters
        self.optimizer.step(step)

        return loss.data

    def validate(self, val_docs: List[str], char_to_idx: dict, bos_token: int) -> float:
        """Compute validation loss."""
        self.model.set_training(False)

        total_loss = 0.0
        count = 0

        for doc in val_docs:
            tokens = (
                [bos_token] + [char_to_idx[ch] for ch in doc if ch in char_to_idx] + [bos_token]
            )
            if len(tokens) < 2:
                continue

            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]

            loss = self.compute_loss(tokens, keys, values)
            total_loss += loss.data
            count += 1

        self.model.set_training(True)

        return total_loss / count if count > 0 else float("inf")

    def train(
        self,
        train_docs: List[str],
        val_docs: Optional[List[str]],
        char_to_idx: dict,
        bos_token: int,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ):
        """Full training loop."""
        best_val_loss = float("inf")

        for step in range(self.config.num_steps):
            # Sample a document
            doc = train_docs[step % len(train_docs)]
            tokens = (
                [bos_token] + [char_to_idx[ch] for ch in doc if ch in char_to_idx] + [bos_token]
            )

            if len(tokens) < 2:
                continue

            # Training step
            loss = self.train_step(tokens, step)
            lr = self.optimizer._get_lr(step)

            # Validation
            val_loss = None
            if val_docs and (step + 1) % self.config.eval_interval == 0:
                val_loss = self.validate(val_docs, char_to_idx, bos_token)

                # Early stopping check
                if self.early_stopping and self.early_stopping(val_loss):
                    print(f"\nEarly stopping at step {step + 1}")
                    break

                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            # Callback for logging
            if callback:
                callback(step + 1, loss, lr, val_loss)

        return best_val_loss
