"""
Enhanced microgpt model with configurable architecture.
Includes GELU, dropout, LayerNorm, and multi-layer support.
"""

import math
import random
from typing import List, Optional


class Value:
    """Scalar with automatic differentiation."""

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        # Clamp to avoid log(0) or log(negative)
        clamped = max(self.data, 1e-10)
        return Value(math.log(clamped), (self,), (1 / clamped,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def gelu(self):
        """Gaussian Error Linear Unit approximation."""
        # GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) *
        # (x + 0.044715 * x³)))
        x = self.data
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        cdf = 0.5 * (1.0 + math.tanh(inner))
        # Derivative of GELU
        tanh_val = math.tanh(inner)
        local_grad = cdf + x * 0.5 * math.sqrt(2.0 / math.pi) * (1 - tanh_val**2) * (
            1 + 3 * 0.044715 * x**2
        )
        return Value(x * cdf, (self,), (local_grad,))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


class Dropout:
    """Dropout regularization."""

    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True

    def __call__(self, x: List[Value]) -> List[Value]:
        if not self.training or self.p == 0:
            return x

        # Inverted dropout: scale by 1/(1-p) during training
        scale = 1.0 / (1.0 - self.p)
        return [xi * scale if random.random() > self.p else Value(0.0) for xi in x]


def linear(x: List[Value], w: List[List[Value]]) -> List[Value]:
    """Linear transformation: x @ W^T"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits: List[Value]) -> List[Value]:
    """Numerically stable softmax."""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(e.data for e in exps)
    # Clamp to avoid division by zero
    total = max(total, 1e-10)
    return [e / total for e in exps]


def rmsnorm(x: List[Value], eps: float = 1e-5) -> List[Value]:
    """Root Mean Square Layer Normalization."""
    ms = sum(xi.data * xi.data for xi in x) / len(x)
    scale = (ms + eps) ** -0.5
    return [xi * scale for xi in x]


def layernorm(x: List[Value], eps: float = 1e-5) -> List[Value]:
    """Layer Normalization with mean centering."""
    mean = sum(xi.data for xi in x) / len(x)
    var = sum((xi.data - mean) ** 2 for xi in x) / len(x)
    scale = (var + eps) ** -0.5
    return [(xi - mean) * scale for xi in x]


class GPT:
    """GPT model with configurable architecture."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 1,
        n_embd: int = 16,
        n_head: int = 4,
        dropout: float = 0.0,
        use_gelu: bool = False,
        use_layernorm: bool = False,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_gelu = use_gelu
        self.use_layernorm = use_layernorm

        # Initialize parameters
        self._init_parameters()

        # Dropout layers
        self.drop = Dropout(dropout)
        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)

    def _init_parameters(self):
        """Initialize all model parameters."""
        std = 0.08

        self.state_dict = {
            "wte": self._matrix(self.vocab_size, self.n_embd, std),
            "wpe": self._matrix(self.block_size, self.n_embd, std),
            "lm_head": self._matrix(self.vocab_size, self.n_embd, std),
        }

        for i in range(self.n_layer):
            # Attention weights
            self.state_dict[f"layer{i}.attn_wq"] = self._matrix(self.n_embd, self.n_embd, std)
            self.state_dict[f"layer{i}.attn_wk"] = self._matrix(self.n_embd, self.n_embd, std)
            self.state_dict[f"layer{i}.attn_wv"] = self._matrix(self.n_embd, self.n_embd, std)
            self.state_dict[f"layer{i}.attn_wo"] = self._matrix(self.n_embd, self.n_embd, std)

            # MLP weights
            self.state_dict[f"layer{i}.mlp_fc1"] = self._matrix(4 * self.n_embd, self.n_embd, std)
            self.state_dict[f"layer{i}.mlp_fc2"] = self._matrix(self.n_embd, 4 * self.n_embd, std)

    def _matrix(self, nout: int, nin: int, std: float) -> List[List[Value]]:
        """Create a matrix of Values with Gaussian initialization."""
        return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    def parameters(self) -> List[Value]:
        """Get all parameters as a flat list."""
        return [p for mat in self.state_dict.values() for row in mat for p in row]

    def num_params(self) -> int:
        """Count total number of parameters."""
        return len(self.parameters())

    def _norm(self, x: List[Value]) -> List[Value]:
        """Apply normalization (LayerNorm or RMSNorm)."""
        if self.use_layernorm:
            return layernorm(x)
        return rmsnorm(x)

    def _activate(self, x: List[Value]) -> List[Value]:
        """Apply activation function (GELU or ReLU)."""
        if self.use_gelu:
            return [xi.gelu() for xi in x]
        return [xi.relu() for xi in x]

    def forward(
        self, token_id: int, pos_id: int, keys: List[List], values: List[List]
    ) -> List[Value]:
        """Forward pass for a single token position."""
        # Embeddings
        tok_emb = self.state_dict["wte"][token_id]
        pos_emb = self.state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = self._norm(x)
        x = self.drop(x)

        # Transformer layers
        for li in range(self.n_layer):
            # Multi-head Attention
            x_residual = x
            x = self._norm(x)

            q = linear(x, self.state_dict[f"layer{li}.attn_wq"])
            k = linear(x, self.state_dict[f"layer{li}.attn_wk"])
            v = linear(x, self.state_dict[f"layer{li}.attn_wv"])

            keys[li].append(k)
            values[li].append(v)

            # Multi-head attention computation
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                k_h = [ki[hs : hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + self.head_dim] for vi in values[li]]

                # Scaled dot-product attention
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / (self.head_dim**0.5)
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                attn_weights = self.attn_drop(attn_weights)

                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            x = linear(x_attn, self.state_dict[f"layer{li}.attn_wo"])
            x = self.resid_drop(x)
            x = [a + b for a, b in zip(x, x_residual)]

            # MLP
            x_residual = x
            x = self._norm(x)
            x = linear(x, self.state_dict[f"layer{li}.mlp_fc1"])
            x = self._activate(x)
            x = self.drop(x)
            x = linear(x, self.state_dict[f"layer{li}.mlp_fc2"])
            x = self.resid_drop(x)
            x = [a + b for a, b in zip(x, x_residual)]

        # Output logits
        logits = linear(x, self.state_dict["lm_head"])
        return logits

    def set_training(self, mode: bool = True):
        """Set training mode (affects dropout)."""
        self.drop.training = mode
        self.attn_drop.training = mode
        self.resid_drop.training = mode

    def generate(
        self,
        token_id: int,
        max_length: int,
        temperature: float = 0.5,
        top_k: int = 0,
        top_p: float = 1.0,
        keys: Optional[List[List]] = None,
        values: Optional[List[List]] = None,
    ) -> List[int]:
        """Generate tokens autoregressively."""
        if keys is None:
            keys = [[] for _ in range(self.n_layer)]
        if values is None:
            values = [[] for _ in range(self.n_layer)]

        generated = []
        for pos_id in range(max_length):
            logits = self.forward(token_id, pos_id, keys, values)

            # Apply temperature
            scaled_logits = [logit / temperature for logit in logits]

            # Convert to probabilities
            probs = softmax(scaled_logits)
            probs_data = [p.data for p in probs]

            # Top-k filtering
            if top_k > 0:
                sorted_probs = sorted(probs_data, reverse=True)
                threshold = sorted_probs[min(top_k, len(sorted_probs)) - 1]
                probs_data = [p if p >= threshold else 0 for p in probs_data]
                # Re-normalize
                total = sum(probs_data)
                probs_data = [p / total for p in probs_data]

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs = sorted(enumerate(probs_data), key=lambda x: x[1], reverse=True)
                cumsum = 0
                keep_indices = set()
                for idx, p in sorted_probs:
                    cumsum += p
                    keep_indices.add(idx)
                    if cumsum >= top_p:
                        break
                probs_data = [p if i in keep_indices else 0 for i, p in enumerate(probs_data)]
                # Re-normalize
                total = sum(probs_data)
                probs_data = [p / total for p in probs_data]

            # Sample
            token_id = random.choices(range(self.vocab_size), weights=probs_data)[0]
            generated.append(token_id)

        return generated
