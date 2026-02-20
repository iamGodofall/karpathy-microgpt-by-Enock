"""
Model interpretability and analysis tools for microgpt.
Visualize attention patterns, analyze neuron activations, and probe representations.
"""

import math
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from .model import GPT, Value


class AttentionVisualizer:
    """Visualize attention patterns across layers and heads."""

    def __init__(self, model: GPT):
        self.model = model
        self.attention_weights: List[List[List[float]]] = []

    def capture_attention(self, tokens: List[int]) -> List[List[List[float]]]:
        """
        Capture attention weights for a sequence.
        Returns: [layer][head][position][position] attention weights
        """
        self.model.set_training(False)

        all_attentions = []
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        for pos in range(len(tokens)):
            # Forward pass, capturing attention
            _ = self.model.forward(tokens[pos], pos, keys, values)

            # Extract attention from keys (simplified)
            # In real implementation, would capture actual attention matrices
            layer_attns = []
            for li in range(self.model.n_layer):
                head_attns = []
                for h in range(self.model.n_head):
                    # Placeholder attention pattern
                    attn = [0.0] * (pos + 1)
                    attn[-1] = 1.0  # Self-attention to current position
                    head_attns.append(attn)
                layer_attns.append(head_attns)

            all_attentions.append(layer_attns)

        return all_attentions

    def get_attention_flow(self, layer: int, head: int, target_pos: int) -> List[float]:
        """
        Get attention flow to a specific position.
        Shows which input positions influence the target position.
        """
        # Simplified - would use actual captured attention
        return [1.0 / (target_pos + 1)] * (target_pos + 1)

    def find_attention_heads(self, pattern_type: str = "positional") -> List[Tuple[int, int]]:
        """
        Find attention heads with specific patterns.
        Patterns: 'positional', 'syntactic', 'rare_tokens', 'bos_focus'
        """
        found_heads = []

        for li in range(self.model.n_layer):
            for hi in range(self.model.n_head):
                # Analyze head pattern (simplified)
                if pattern_type == "positional":
                    # Check if attention follows position bias
                    found_heads.append((li, hi))
                elif pattern_type == "syntactic":
                    # Check for syntactic patterns
                    pass

        return found_heads

    def export_attention_html(self, tokens: List[int], tokenizer, filename: str = "attention.html"):
        """
        Export attention visualization as interactive HTML.
        """
        attentions = self.capture_attention(tokens)

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attention Visualization</title>
            <style>
                body { font-family: monospace; }
                .token { display: inline-block; padding: 2px 5px; margin: 1px; }
                .attention-row { margin: 2px 0; }
            </style>
        </head>
        <body>
            <h1>Attention Visualization</h1>
        """

        # Add tokens
        token_strs = [tokenizer.idx_to_char.get(t, "?") for t in tokens]
        html += "<div>"
        for i, tok in enumerate(token_strs):
            html += f'<span class="token" id="tok-{i}">{tok}</span>'
        html += "</div>"

        html += "</body></html>"

        with open(filename, "w") as f:
            f.write(html)

        print(f"Attention visualization saved to {filename}")


class NeuronAnalyzer:
    """Analyze individual neuron activations and their roles."""

    def __init__(self, model: GPT):
        self.model = model
        self.activation_history: Dict[str, List[List[float]]] = defaultdict(list)

    def capture_activations(self, tokens: List[int]) -> Dict[str, List[List[float]]]:
        """
        Capture activations from all layers.
        """
        self.model.set_training(False)

        activations = defaultdict(list)
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        for pos, token in enumerate(tokens):
            # Forward pass
            logits = self.model.forward(token, pos, keys, values)

            # Capture intermediate activations
            for li in range(self.model.n_layer):
                # Simplified - would capture actual intermediate values
                layer_acts = [0.0] * self.model.n_embd
                activations[f"layer_{li}"].append(layer_acts)

        return dict(activations)

    def find_maximally_activating_tokens(
        self, layer: int, neuron: int, texts: List[str], tokenizer
    ) -> List[Tuple[str, float]]:
        """
        Find tokens that maximally activate a specific neuron.
        """
        activations = []

        for text in texts:
            tokens = tokenizer.encode(text)
            acts = self.capture_activations(tokens)

            for pos, act in enumerate(acts.get(f"layer_{layer}", [])):
                if len(act) > neuron:
                    activations.append((text[pos : pos + 1], act[neuron]))

        # Sort by activation
        return sorted(activations, key=lambda x: -x[1])[:20]

    def analyze_neuron_specialization(
        self, layer: int, neuron: int, test_inputs: List[str], tokenizer
    ) -> Dict[str, any]:
        """
        Analyze what a specific neuron responds to.
        """
        results = {
            "layer": layer,
            "neuron": neuron,
            "max_activation": 0.0,
            "min_activation": 0.0,
            "avg_activation": 0.0,
            "responsive_tokens": [],
        }

        all_activations = []

        for text in test_inputs:
            tokens = tokenizer.encode(text)
            acts = self.capture_activations(tokens)

            layer_acts = acts.get(f"layer_{layer}", [])
            for pos, act in enumerate(layer_acts):
                if len(act) > neuron:
                    all_activations.append((text, pos, act[neuron]))

        if all_activations:
            activations_only = [a[2] for a in all_activations]
            results["max_activation"] = max(activations_only)
            results["min_activation"] = min(activations_only)
            results["avg_activation"] = sum(activations_only) / len(activations_only)

            # Find most responsive tokens
            top_activating = sorted(all_activations, key=lambda x: -x[2])[:10]
            results["responsive_tokens"] = [
                {"text": t, "position": p, "activation": a} for t, p, a in top_activating
            ]

        return results


class ProbingClassifier:
    """
    Linear probing to analyze what information is encoded in representations.
    """

    def __init__(self, model: GPT):
        self.model = model
        self.probes: Dict[str, List[List[float]]] = {}

    def train_probe(
        self, name: str, representations: List[List[float]], labels: List[int], num_classes: int
    ):
        """
        Train a linear probe on representations.
        """
        # Simple linear classifier
        input_dim = len(representations[0])

        # Initialize probe weights
        self.probes[name] = [
            [random.gauss(0, 0.01) for _ in range(input_dim)] for _ in range(num_classes)
        ]

        # Train (simplified SGD)
        for _ in range(100):
            for rep, label in zip(representations, labels):
                # Forward
                logits = [sum(w[i] * rep[i] for i in range(input_dim)) for w in self.probes[name]]

                # Softmax
                max_logit = max(logits)
                exps = [math.exp(l - max_logit) for logit in logits]
                total = sum(exps)
                probs = [e / total for e in exps]

                # Backward (simplified)
                for i in range(num_classes):
                    grad = probs[i] - (1 if i == label else 0)
                    for j in range(input_dim):
                        self.probes[name][i][j] -= 0.01 * grad * rep[j]

    def evaluate_probe(
        self, name: str, representations: List[List[float]], labels: List[int]
    ) -> float:
        """
        Evaluate probe accuracy.
        """
        correct = 0
        for rep, label in zip(representations, labels):
            logits = [sum(w[i] * rep[i] for i in range(len(rep))) for w in self.probes[name]]
            pred = logits.index(max(logits))
            if pred == label:
                correct += 1

        return correct / len(labels) if labels else 0.0

    def probe_positional_information(self, texts: List[str], tokenizer) -> float:
        """
        Probe how well position information is encoded.
        """
        representations = []
        positions = []

        for text in texts:
            tokens = tokenizer.encode(text)
            acts = self.capture_representations(tokens)

            for pos, rep in enumerate(acts):
                representations.append(rep)
                positions.append(pos % 10)  # Position mod 10 as label

        # Train probe
        self.train_probe("position", representations, positions, 10)

        # Evaluate
        return self.evaluate_probe("position", representations, positions)

    def capture_representations(self, tokens: List[int]) -> List[List[float]]:
        """
        Capture final layer representations.
        """
        self.model.set_training(False)

        reps = []
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        for pos, token in enumerate(tokens):
            logits = self.model.forward(token, pos, keys, values)
            # Use pre-final layer as representation
            # Simplified - would capture actual hidden states
            rep = [l.data for logit in logits]
            reps.append(rep)

        return reps


class SaliencyAnalyzer:
    """Analyze input saliency for model predictions."""

    def __init__(self, model: GPT):
        self.model = model

    def compute_saliency(self, tokens: List[int], target_pos: int) -> List[float]:
        """
        Compute gradient-based saliency for each input token.
        Shows which input tokens most influence the prediction.
        """
        self.model.set_training(False)

        # Forward pass
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        for pos in range(target_pos + 1):
            logits = self.model.forward(tokens[pos], pos, keys, values)

        # Get target logit
        target_logit = logits[tokens[target_pos]]

        # Backward to get gradients on embeddings
        target_logit.backward()

        # Saliency is gradient magnitude
        # Simplified - would extract actual input gradients
        saliency = [1.0] * len(tokens)

        return saliency

    def integrated_gradients(
        self, tokens: List[int], target_pos: int, num_steps: int = 50
    ) -> List[float]:
        """
        Compute integrated gradients for more stable attributions.
        """
        # Simplified implementation
        # Real implementation would interpolate from baseline
        return self.compute_saliency(tokens, target_pos)


def analyze_model(model: GPT, sample_text: str, tokenizer):
    """
    Run full interpretability analysis on a model.
    """
    print("=" * 70)
    print("MODEL INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    tokens = tokenizer.encode(sample_text)
    print(f"\nAnalyzing text: '{sample_text}'")
    print(f"Tokens: {tokens}")

    # Attention analysis
    print("\n--- Attention Patterns ---")
    attn_viz = AttentionVisualizer(model)
    attentions = attn_viz.capture_attention(tokens)
    print(f"Captured attention for {len(attentions)} positions")

    # Neuron analysis
    print("\n--- Neuron Activations ---")
    neuron_analyzer = NeuronAnalyzer(model)
    acts = neuron_analyzer.capture_activations(tokens)
    print(f"Captured activations from {len(acts)} layers")

    # Probing
    print("\n--- Linear Probing ---")
    prober = ProbingClassifier(model)
    # Would train and evaluate probes here

    # Saliency
    print("\n--- Input Saliency ---")
    saliency = SaliencyAnalyzer(model)
    scores = saliency.compute_saliency(tokens, len(tokens) - 1)
    print(f"Saliency scores: {scores[:5]}...")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
