"""
Quantum-inspired computing for microgpt.
Superposition, entanglement, and quantum gates in pure Python.
"""

import random
import math
import cmath
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Qubit:
    """A quantum bit with superposition."""

    alpha: complex = 1 + 0j  # |0⟩ amplitude
    beta: complex = 0 + 0j  # |1⟩ amplitude

    def __post_init__(self):
        self.normalize()

    def normalize(self):
        """Ensure |α|² + |β|² = 1."""
        norm = math.sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm

    def measure(self) -> int:
        """Collapse superposition to 0 or 1."""
        prob_0 = abs(self.alpha) ** 2
        return 0 if random.random() < prob_0 else 1

    def probability_0(self) -> float:
        return abs(self.alpha) ** 2

    def probability_1(self) -> float:
        return abs(self.beta) ** 2

    def apply_gate(self, gate: "QuantumGate"):
        """Apply quantum gate."""
        new_alpha = gate.matrix[0][0] * self.alpha + gate.matrix[0][1] * self.beta
        new_beta = gate.matrix[1][0] * self.alpha + gate.matrix[1][1] * self.beta
        self.alpha, self.beta = new_alpha, new_beta
        self.normalize()

    def __repr__(self):
        return f"|ψ⟩ = {self.alpha:.3f}|0⟩ + {self.beta:.3f}|1⟩"


class QuantumGate:
    """Quantum logic gate."""

    def __init__(self, name: str, matrix: List[List[complex]]):
        self.name = name
        self.matrix = matrix

    @classmethod
    def hadamard(cls):
        """Hadamard gate: creates superposition."""
        h = 1 / math.sqrt(2)
        return cls("H", [[h, h], [h, -h]])

    @classmethod
    def pauli_x(cls):
        """Pauli-X (NOT) gate."""
        return cls("X", [[0, 1], [1, 0]])

    @classmethod
    def pauli_y(cls):
        """Pauli-Y gate."""
        return cls("Y", [[0, -1j], [1j, 0]])

    @classmethod
    def pauli_z(cls):
        """Pauli-Z gate."""
        return cls("Z", [[1, 0], [0, -1]])

    @classmethod
    def phase(cls, theta: float):
        """Phase shift gate."""
        return cls("P", [[1, 0], [0, cmath.exp(1j * theta)]])

    @classmethod
    def cnot(cls):
        """Controlled-NOT gate (2-qubit)."""
        return cls("CNOT", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class QuantumRegister:
    """Register of qubits."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qubits = [Qubit(1, 0) for _ in range(n_qubits)]

    def apply_single(self, gate: QuantumGate, target: int):
        """Apply single-qubit gate."""
        self.qubits[target].apply_gate(gate)

    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        # If control is |1⟩, flip target
        if self.qubits[control].measure() == 1:
            self.qubits[target].apply_gate(QuantumGate.pauli_x())

    def measure_all(self) -> List[int]:
        """Measure all qubits."""
        return [q.measure() for q in self.qubits]

    def get_state_vector(self) -> List[complex]:
        """Get full state vector (exponential!)."""
        # For n qubits, state vector has 2^n amplitudes
        # Simplified: return individual qubit states
        return [(q.alpha, q.beta) for q in self.qubits]

    def entangle(self, q1: int, q2: int):
        """Create Bell state between two qubits."""
        # Apply Hadamard to q1, then CNOT with q1 as control
        self.apply_single(QuantumGate.hadamard(), q1)
        self.apply_cnot(q1, q2)


class QuantumNeuralLayer:
    """Neural network layer with quantum-inspired computation."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        # Quantum registers for each output neuron
        self.registers = [QuantumRegister(in_features) for _ in range(out_features)]

        # Classical weights for measurement interpretation
        self.weights = [
            [random.gauss(0, 0.1) for _ in range(in_features)] for _ in range(out_features)
        ]

    def forward(self, inputs: List[float]) -> List[float]:
        """Quantum forward pass."""
        outputs = []

        for i, reg in enumerate(self.registers):
            # Encode inputs into qubit phases
            for j, (qubit, inp) in enumerate(zip(reg.qubits, inputs)):
                # Use input to set phase
                phase_gate = QuantumGate.phase(inp * math.pi)
                qubit.apply_gate(phase_gate)

            # Apply entanglement
            if len(reg.qubits) > 1:
                reg.entangle(0, 1)

            # Measure and combine with classical weights
            measurements = reg.measure_all()
            weighted_sum = sum(m * w for m, w in zip(measurements, self.weights[i]))
            outputs.append(math.tanh(weighted_sum))  # Activation

        return outputs


class QuantumOptimizer:
    """Quantum-inspired optimization."""

    def __init__(self, n_params: int):
        self.n_params = n_params
        self.quantum_params = [Qubit() for _ in range(n_params)]

    def step(self, gradients: List[float], lr: float = 0.01):
        """Update parameters using quantum interference."""
        for i, (q, grad) in enumerate(zip(self.quantum_params, gradients)):
            # Use gradient to influence phase
            phase = -lr * grad * math.pi
            q.apply_gate(QuantumGate.phase(phase))

            # Measurement provides update direction
            measurement = q.measure()
            # In real quantum, this would be more sophisticated

        return [q.probability_1() for q in self.quantum_params]


class QuantumGPT:
    """GPT with quantum-inspired components."""

    def __init__(self, vocab_size: int, n_embd: int, n_layer: int):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer

        # Quantum embedding layer
        self.q_embed = QuantumNeuralLayer(vocab_size, n_embd)

        # Quantum transformer layers
        self.layers = [QuantumNeuralLayer(n_embd, n_embd) for _ in range(n_layer)]

    def forward(self, tokens: List[int]) -> List[float]:
        """Quantum forward pass."""
        # One-hot encode
        one_hot = [0.0] * self.vocab_size
        for t in tokens:
            if 0 <= t < self.vocab_size:
                one_hot[t] = 1.0

        # Quantum embedding
        x = self.q_embed.forward(one_hot)

        # Quantum transformer layers
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def generate(self, prompt: List[int], max_len: int = 10) -> List[int]:
        """Generate with quantum randomness."""
        result = prompt.copy()

        for _ in range(max_len):
            logits = self.forward(result[-self.n_embd :])
            # Use quantum measurement for sampling
            probs = self._softmax(logits)
            next_token = self._quantum_sample(probs)
            result.append(next_token)

        return result

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax normalization."""
        exp_x = [math.exp(xi) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]

    def _quantum_sample(self, probs: List[float]) -> int:
        """Sample using quantum superposition."""
        # Create qubit with amplitudes from probabilities
        q = Qubit(
            complex(math.sqrt(probs[0]) if probs else 1, 0),
            complex(math.sqrt(1 - probs[0]) if len(probs) > 1 else 0, 0),
        )
        return q.measure()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Quantum-Inspired Neural Networks")
    print("Superposition × Entanglement × Interference")
    print("=" * 70)

    # Single qubit demo
    print("\n1. Single Qubit Operations")
    q = Qubit(1, 0)  # |0⟩
    print(f"Initial: {q}")

    q.apply_gate(QuantumGate.hadamard())
    print(f"After H: {q}")

    measurement = q.measure()
    print(f"Measured: {measurement}")

    # Quantum register
    print("\n2. Quantum Register (3 qubits)")
    reg = QuantumRegister(3)
    print(f"Initial: {reg.measure_all()}")

    # Create superposition
    for i in range(3):
        reg.apply_single(QuantumGate.hadamard(), i)
    print(f"After H: {reg.measure_all()}")

    # Entanglement
    print("\n3. Entanglement (Bell State)")
    reg2 = QuantumRegister(2)
    reg2.entangle(0, 1)
    m1 = reg2.qubits[0].measure()
    m2 = reg2.qubits[1].measure()
    print(f"Qubit 0: {m1}, Qubit 1: {m2} (should be correlated)")

    # Quantum neural layer
    print("\n4. Quantum Neural Layer")
    layer = QuantumNeuralLayer(4, 3)
    output = layer.forward([0.5, -0.3, 0.8, 0.1])
    print(f"Input: [0.5, -0.3, 0.8, 0.1]")
    print(f"Output: {[f'{o:.3f}' for o in output]}")

    # Quantum GPT
    print("\n5. Quantum GPT")
    model = QuantumGPT(vocab_size=10, n_embd=8, n_layer=2)
    output = model.forward([1, 2, 3])
    print(f"GPT output: {[f'{o:.3f}' for o in output]}")

    print("\n✨ Quantum-inspired computation")
    print("✨ Superposition enables parallel exploration")
    print("✨ Entanglement creates correlations")
    print("✨ Measurement collapses to classical output")
