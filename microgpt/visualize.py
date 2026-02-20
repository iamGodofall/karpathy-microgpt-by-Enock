"""
Visualization tools for training metrics and model analysis.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import math

from .model import GPT


def generate_training_plot(log_file: str, output_file: str = "training_curves.html"):
    """Generate interactive training curves using Plotly (if available) or matplotlib."""
    # Load metrics
    metrics = []
    with open(log_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if "step" in data:
                metrics.append(data)

    if not metrics:
        print("No metrics found in log file")
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    perplexities = [m["perplexity"] for m in metrics]
    learning_rates = [m["learning_rate"] for m in metrics]

    # Try plotly first (interactive)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Loss", "Perplexity", "Learning Rate"),
            vertical_spacing=0.1,
        )

        # Loss curve
        fig.add_trace(
            go.Scatter(x=steps, y=losses, mode="lines", name="Loss", line=dict(color="blue")),
            row=1,
            col=1,
        )

        # Perplexity curve
        fig.add_trace(
            go.Scatter(
                x=steps, y=perplexities, mode="lines", name="Perplexity", line=dict(color="green")
            ),
            row=2,
            col=1,
        )

        # Learning rate curve
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=learning_rates,
                mode="lines",
                name="Learning Rate",
                line=dict(color="red"),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(title="Training Curves", height=800, showlegend=False)

        fig.write_html(output_file)
        print(f"Interactive plot saved to {output_file}")

    except ImportError:
        # Fallback to matplotlib
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            # Loss
            axes[0].plot(steps, losses, "b-", label="Loss")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].grid(True)

            # Perplexity
            axes[1].plot(steps, perplexities, "g-", label="Perplexity")
            axes[1].set_ylabel("Perplexity")
            axes[1].set_title("Perplexity")
            axes[1].grid(True)

            # Learning rate
            axes[2].plot(steps, learning_rates, "r-", label="Learning Rate")
            axes[2].set_ylabel("Learning Rate")
            axes[2].set_xlabel("Step")
            axes[2].set_title("Learning Rate Schedule")
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig(output_file.replace(".html", ".png"), dpi=150)
            print(f"Static plot saved to {output_file.replace('.html', '.png')}")

        except ImportError:
            print("Neither plotly nor matplotlib available. Install one to generate plots.")


def analyze_generation(samples: List[str], output_file: Optional[str] = None):
    """Analyze generated text for diversity and patterns."""
    from collections import Counter

    if not samples:
        print("No samples to analyze")
        return

    # Basic statistics
    lengths = [len(s) for s in samples]
    unique_samples = len(set(samples))

    print(f"\n=== Generation Analysis ===")
    print(f"Total samples: {len(samples)}")
    print(f"Unique samples: {unique_samples} ({100*unique_samples/len(samples):.1f}%)")
    print(f"Avg length: {sum(lengths)/len(lengths):.1f}")
    print(f"Length range: {min(lengths)} - {max(lengths)}")

    # Character frequency
    all_chars = "".join(samples)
    char_freq = Counter(all_chars)
    print(f"\nTop 10 characters:")
    for char, count in char_freq.most_common(10):
        print(f"  '{char}': {count} ({100*count/len(all_chars):.1f}%)")

    # Repetition analysis
    repeats = sum(1 for s in samples if len(set(s)) < len(s) * 0.5)
    print(f"\nSamples with high repetition: {repeats} ({100*repeats/len(samples):.1f}%)")

    # N-gram analysis
    bigrams = Counter()
    for s in samples:
        for i in range(len(s) - 1):
            bigrams[s[i : i + 2]] += 1

    print(f"\nTop 10 bigrams:")
    for bigram, count in bigrams.most_common(10):
        print(f"  '{bigram}': {count}")

    # Save to file if requested
    if output_file:
        analysis = {
            "total_samples": len(samples),
            "unique_samples": unique_samples,
            "avg_length": sum(lengths) / len(lengths),
            "char_frequency": dict(char_freq.most_common(20)),
            "top_bigrams": dict(bigrams.most_common(20)),
        }

        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {output_file}")


def compare_checkpoints(checkpoint_paths: List[str], data_path: str = "input.txt"):
    """Compare multiple checkpoints on the same validation set."""
    from data import DataLoader
    from model import GPT
    from trainer import Trainer
    from config import Config, ModelConfig, TrainingConfig, GenerationConfig
    from checkpoint import CheckpointManager
    import pickle

    loader = DataLoader()
    _, val_docs = loader.load_file(data_path, val_split=1.0)
    tokenizer = loader.tokenizer

    results = []

    for path in checkpoint_paths:
        print(f"\nEvaluating {path}...")

        # Load checkpoint
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        # Reconstruct model
        config_data = checkpoint["config"]
        config = Config(
            model=ModelConfig(**config_data.get("model", {})),
            training=TrainingConfig(**config_data.get("training", {})),
            generation=GenerationConfig(**config_data.get("generation", {})),
        )

        model = GPT(
            vocab_size=tokenizer.vocab_size,
            block_size=config.model.block_size,
            n_layer=config.model.n_layer,
            n_embd=config.model.n_embd,
            n_head=config.model.n_head,
            dropout=0.0,
            use_gelu=config.model.use_gelu,
            use_layernorm=config.model.use_layernorm,
        )

        # Load weights
        for key, matrix in checkpoint["state_dict"].items():
            for i, row in enumerate(matrix):
                for j, val in enumerate(row):
                    model.state_dict[key][i][j].data = val

        model.set_training(False)

        # Evaluate
        trainer = Trainer(model, config.training)
        val_loss = trainer.validate(val_docs, tokenizer.char_to_idx, tokenizer.bos_token)
        perplexity = math.exp(val_loss)

        results.append(
            {
                "checkpoint": path,
                "step": checkpoint.get("step", 0),
                "loss": val_loss,
                "perplexity": perplexity,
            }
        )

        print(f"  Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

    # Summary
    print(f"\n=== Comparison Summary ===")
    best = min(results, key=lambda x: x["loss"])
    print(f"Best checkpoint: {best['checkpoint']} (step {best['step']})")
    print(f"  Loss: {best['loss']:.4f}, Perplexity: {best['perplexity']:.2f}")

    return results


def export_to_onnx(model: GPT, output_path: str = "model.onnx"):
    """Export model to ONNX format (requires PyTorch)."""
    try:
        import torch
        import torch.nn as nn

        # This is a placeholder - full implementation would require
        # converting the pure Python model to PyTorch first
        print("ONNX export requires PyTorch implementation")
        print("Consider using the PyTorch version of microgpt for production")

    except ImportError:
        print("PyTorch not available. Install with: pip install torch")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize.py plot <log_file> [output_file]")
        print("  python visualize.py analyze <samples_file>")
        print("  python visualize.py compare <checkpoint1> <checkpoint2> ...")
        sys.exit(1)

    command = sys.argv[1]

    if command == "plot":
        log_file = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else "training_curves.html"
        generate_training_plot(log_file, output)

    elif command == "analyze":
        samples_file = sys.argv[2]
        with open(samples_file, "r") as f:
            samples = [line.strip() for line in f if line.strip()]
        analyze_generation(samples)

    elif command == "compare":
        checkpoints = sys.argv[2:]
        compare_checkpoints(checkpoints)

    else:
        print(f"Unknown command: {command}")
