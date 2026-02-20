"""
Model compression techniques.
Pruning, knowledge distillation, and weight sharing.
"""

import random
import math
from typing import List, Dict, Tuple, Optional
from .model import GPT, Value


class MagnitudePruning:
    """
    Prune weights by magnitude.
    """

    def __init__(self, sparsity: float = 0.5):
        self.sparsity = sparsity  # Fraction to prune
        self.masks: Dict[str, List[List[bool]]] = {}

    def prune(self, model: GPT) -> GPT:
        """
        Prune model weights.
        """
        pruned = GPT(
            vocab_size=model.vocab_size,
            block_size=model.block_size,
            n_layer=model.n_layer,
            n_embd=model.n_embd,
            n_head=model.n_head,
        )

        for name, matrix in model.state_dict.items():
            # Find threshold
            all_values = [abs(v.data) for row in matrix for v in row]
            threshold = sorted(all_values)[int(len(all_values) * self.sparsity)]

            # Create mask
            mask = [[abs(v.data) > threshold for v in row] for row in matrix]
            self.masks[name] = mask

            # Apply mask
            for i, row in enumerate(matrix):
                for j, v in enumerate(row):
                    if mask[i][j]:
                        pruned.state_dict[name][i][j].data = v.data
                    else:
                        pruned.state_dict[name][i][j].data = 0

        return pruned

    def count_nonzero(self, model: GPT) -> int:
        """Count non-zero parameters."""
        count = 0
        for name, matrix in model.state_dict.items():
            for row in matrix:
                for v in row:
                    if abs(v.data) > 1e-10:
                        count += 1
        return count


class StructuredPruning:
    """
    Prune entire neurons/channels.
    """

    def __init__(self, target_size: float = 0.5):
        self.target_size = target_size

    def prune_attention_heads(self, model: GPT, heads_to_keep: List[int]) -> GPT:
        """
        Prune to specified attention heads.
        """
        pruned = GPT(
            vocab_size=model.vocab_size,
            block_size=model.block_size,
            n_layer=model.n_layer,
            n_embd=model.n_embd,
            n_head=len(heads_to_keep),
        )

        # Copy only selected heads
        # Simplified - would need to restructure weight matrices

        return pruned

    def prune_layers(self, model: GPT, layers_to_keep: List[int]) -> GPT:
        """
        Prune to specified layers.
        """
        pruned = GPT(
            vocab_size=model.vocab_size,
            block_size=model.block_size,
            n_layer=len(layers_to_keep),
            n_embd=model.n_embd,
            n_head=model.n_head,
        )

        # Copy only selected layers
        for new_idx, old_idx in enumerate(layers_to_keep):
            for name in ["attn_wq", "attn_wk", "attn_wv", "attn_wo", "mlp_fc1", "mlp_fc2"]:
                old_name = f"layer{old_idx}.{name}"
                new_name = f"layer{new_idx}.{name}"
                if old_name in model.state_dict:
                    pruned.state_dict[new_name] = model.state_dict[old_name]

        return pruned


class KnowledgeDistillation:
    """
    Distill knowledge from teacher to student.
    """

    def __init__(self, teacher: GPT, student: GPT, temperature: float = 2.0, alpha: float = 0.5):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft vs hard loss

    def distillation_loss(
        self, student_logits: List[Value], teacher_logits: List[Value], target: int
    ) -> Value:
        """
        Compute distillation loss.
        """
        # Soft targets from teacher
        teacher_probs = self._softmax_with_temp(teacher_logits, self.temperature)

        # Student soft predictions
        student_probs = self._softmax_with_temp(student_logits, self.temperature)

        # KL divergence
        kl_loss = sum(
            t.data * math.log(t.data / (s.data + 1e-10) + 1e-10)
            for t, s in zip(teacher_probs, student_probs)
        )

        # Hard target loss
        hard_loss = -student_logits[target].log()

        # Combined
        return Value(self.alpha * kl_loss + (1 - self.alpha) * hard_loss.data)

    def _softmax_with_temp(self, logits: List[Value], temperature: float) -> List[Value]:
        """Softmax with temperature."""
        scaled = [logit.data / temperature for logit in logits]

        max_val = max(scaled)
        exps = [math.exp(s - max_val) for s in scaled]
        total = sum(exps)
        return [Value(e / total) for e in exps]

    def train_step(self, tokens: List[int]) -> float:
        """
        One distillation training step.
        """
        # Teacher forward (no grad)
        teacher_keys = [[] for _ in range(self.teacher.n_layer)]
        teacher_values = [[] for _ in range(self.teacher.n_layer)]

        student_keys = [[] for _ in range(self.student.n_layer)]
        student_values = [[] for _ in range(self.student.n_layer)]

        total_loss = 0.0

        for i in range(len(tokens) - 1):
            # Teacher prediction
            teacher_logits = self.teacher.forward(tokens[i], i, teacher_keys, teacher_values)

            # Student prediction
            student_logits = self.student.forward(tokens[i], i, student_keys, student_values)

            # Distillation loss
            loss = self.distillation_loss(student_logits, teacher_logits, tokens[i + 1])
            total_loss += loss.data

        return total_loss / (len(tokens) - 1)


class WeightSharing:
    """
    Share weights across layers to reduce parameters.
    """

    @staticmethod
    def share_across_layers(model: GPT, share_every: int = 2) -> GPT:
        """
        Share weights every N layers.
        """
        shared = GPT(
            vocab_size=model.vocab_size,
            block_size=model.block_size,
            n_layer=model.n_layer,
            n_embd=model.n_embd,
            n_head=model.n_head,
        )

        # Copy base layers
        for i in range(model.n_layer):
            source_idx = (i // share_every) * share_every

            for name in ["attn_wq", "attn_wk", "attn_wv", "attn_wo", "mlp_fc1", "mlp_fc2"]:
                src_name = f"layer{source_idx}.{name}"
                dst_name = f"layer{i}.{name}"

                if src_name in model.state_dict:
                    # Share reference (in real implementation)
                    # Here we just copy
                    shared.state_dict[dst_name] = model.state_dict[src_name]

        return shared


class QuantizationAwareTraining:
    """
    Train with quantization in the loop.
    """

    def __init__(self, model: GPT, bits: int = 8):
        self.model = model
        self.bits = bits

    def fake_quantize(self, tensor: List[List[Value]]) -> List[List[Value]]:
        """
        Simulate quantization during training.
        """
        # Find range
        all_vals = [v.data for row in tensor for v in row]
        min_val = min(all_vals)
        max_val = max(all_vals)

        # Quantize and dequantize
        scale = (max_val - min_val) / (2**self.bits - 1)

        result = []
        for row in tensor:
            new_row = []
            for v in row:
                q = round((v.data - min_val) / scale)
                dq = q * scale + min_val
                new_row.append(Value(dq))
            result.append(new_row)

        return result

    def forward_with_quantization(
        self, token_id: int, pos: int, keys: List[List], values: List[List]
    ) -> List[Value]:
        """
        Forward pass with fake quantization.
        """
        # Quantize weights before use
        for name in ["wte", "wpe"]:
            self.model.state_dict[name] = self.fake_quantize(self.model.state_dict[name])

        # Normal forward
        return self.model.forward(token_id, pos, keys, values)


class DynamicInference:
    """
    Dynamic inference with early exit.
    """

    def __init__(self, model: GPT, confidence_threshold: float = 0.9):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def should_exit_early(self, logits: List[Value], layer_idx: int) -> bool:
        """
        Check if we can exit early from this layer.
        """
        # Confidence = max probability
        probs = [logit.data for logit in logits]

        max_prob = max(probs) / sum(probs)

        # Also consider layer depth
        depth_factor = layer_idx / self.model.n_layer

        return max_prob > self.confidence_threshold and depth_factor > 0.5

    def generate_with_early_exit(self, token_id: int, max_length: int) -> List[int]:
        """
        Generate with early exit.
        """
        generated = []

        for pos in range(max_length):
            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]

            for layer_idx in range(self.model.n_layer):
                logits = self.model.forward(token_id, pos, keys, values)

                # Check early exit
                if self.should_exit_early(logits, layer_idx):
                    break

            # Sample
            probs = [logit.data for logit in logits]

            total = sum(probs)
            probs = [p / total for p in probs]
            token_id = random.choices(range(len(probs)), weights=probs)[0]
            generated.append(token_id)

        return generated


def compress_model(model: GPT, method: str = "magnitude", **kwargs) -> GPT:
    """
    Compress model using specified method.
    """
    if method == "magnitude":
        pruner = MagnitudePruning(kwargs.get("sparsity", 0.5))
        return pruner.prune(model)
    elif method == "structured":
        pruner = StructuredPruning(kwargs.get("target_size", 0.5))
        return pruner.prune_layers(model, kwargs.get("layers_to_keep", [0, 1]))
    elif method == "sharing":
        return WeightSharing.share_across_layers(model, kwargs.get("share_every", 2))
    else:
        raise ValueError(f"Unknown compression method: {method}")


def distill_model(
    teacher: GPT, student_config: dict, data: List[List[int]], epochs: int = 10
) -> GPT:
    """
    Distill teacher model to student.
    """
    student = GPT(**student_config)
    distiller = KnowledgeDistillation(teacher, student)

    for epoch in range(epochs):
        total_loss = 0
        for tokens in data:
            loss = distiller.train_step(tokens)
            total_loss += loss

        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return student
