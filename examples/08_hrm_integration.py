#!/usr/bin/env python3
"""
Example 08: Hierarchical Reasoning Model (HRM) Integration

This example demonstrates the HRM architecture integrated with microgpt:
- Hierarchical reasoning with high-level (slow) and low-level (fast) modules
- Adaptive Computation Time (ACT) with Q-learning
- Pure Python implementation of the architecture from:
  "Hierarchical Reasoning Model" (Wang et al., 2025)

Key Features:
- Dual-timescale processing (H_cycles × L_cycles)
- Dynamic halting based on confidence
- No explicit chain-of-thought required
- Single forward pass complex reasoning
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microgpt_hrm_integration import (
    HybridGPTWithHRM,
    HRMIntegratedConfig,
    HRMTrainer,
    CharTokenizer,
    create_demo_dataset,
)


def demo_basic_hrm():
    """Demonstrate basic HRM training and generation."""
    print("=" * 70)
    print("HRM Basic Demo")
    print("=" * 70)

    # Training text
    text = """
    the quick brown fox jumps over the lazy dog
    pack my box with five dozen liquor jugs
    how vexingly quick daft zebras jump
    """

    # Create dataset
    data, tokenizer = create_demo_dataset(text, seq_len=8)
    print(f"\nDataset: {len(data)} sequences")
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")

    # Configure HRM
    config = HRMIntegratedConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=48,
        num_heads=4,
        H_layers=1,  # High-level (planning)
        L_layers=2,  # Low-level (execution)
        H_cycles=2,  # Planning iterations
        L_cycles=3,  # Execution iterations per planning step
        halt_max_steps=6,
        halt_exploration_prob=0.15,
        learning_rate=0.01,
        num_steps=200,
    )

    # Create model
    print("\nInitializing HRM...")
    model = HybridGPTWithHRM(config)

    # Show architecture
    stats = model.get_stats()
    print("\nArchitecture:")
    print(f"  Hidden size: {stats['hidden_size']}")
    print(f"  H layers: {stats['H_layers']} (planning)")
    print(f"  L layers: {stats['L_layers']} (execution)")
    print(f"  H cycles: {stats['H_cycles']}")
    print(f"  L cycles: {stats['L_cycles']}")
    print(f"  Total parameters: {stats['total_parameters']:,}")

    # Train
    print("\nTraining...")
    trainer = HRMTrainer(model, config)
    history = trainer.train(data, num_steps=100, eval_interval=50)

    # Generate
    print("\n" + "-" * 70)
    print("Generation (HRM mode):")
    print("-" * 70)

    prompts = ["the quick", "pack my", "how vex"]
    for prompt in prompts:
        result = model.generate(prompt, tokenizer, max_length=15, use_hrm=True)
        print(f"  '{prompt}' -> '{result}'")

    return model, tokenizer


def demo_act_behavior():
    """Demonstrate Adaptive Computation Time behavior."""
    print("\n" + "=" * 70)
    print("ACT (Adaptive Computation Time) Demo")
    print("=" * 70)

    text = "reasoning requires thinking step by step through problems carefully"
    data, tokenizer = create_demo_dataset(text, seq_len=6)

    config = HRMIntegratedConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_heads=4,
        H_layers=1,
        L_layers=1,
        H_cycles=2,
        L_cycles=2,
        halt_max_steps=8,
        halt_exploration_prob=0.1,
    )

    model = HybridGPTWithHRM(config)

    print("\nTesting ACT behavior on different inputs...")

    test_inputs = [
        "reason",  # Short, might halt early
        "reasoning requires",  # Medium
        "reasoning requires thinking step",  # Longer
    ]

    for inp in test_inputs:
        tokens = tokenizer.encode(inp)
        result = model.forward_hrm(tokens)
        steps = result["steps"]
        print(f"  Input length {len(tokens):2d}: {steps} computation steps")

    print("\nACT allows dynamic computation based on input complexity!")


def demo_comparison():
    """Compare HRM vs standard transformer."""
    print("\n" + "=" * 70)
    print("HRM vs Standard Transformer")
    print("=" * 70)

    text = """
    in machine learning neural networks learn patterns from data
    deep learning uses multiple layers to extract features
    transformers use attention mechanisms for sequence modeling
    """

    data, tokenizer = create_demo_dataset(text, seq_len=10)

    # HRM config
    hrm_config = HRMIntegratedConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_heads=4,
        H_layers=2,
        L_layers=2,
        H_cycles=2,
        L_cycles=2,
        n_layer=0,  # No standard layers
        halt_max_steps=6,
        num_steps=150,
    )

    # Hybrid config (HRM + standard)
    hybrid_config = HRMIntegratedConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_heads=4,
        H_layers=1,
        L_layers=1,
        H_cycles=2,
        L_cycles=2,
        n_layer=2,  # Add standard layers
        halt_max_steps=6,
        num_steps=150,
    )

    print("\nTraining HRM-only model...")
    hrm_model = HybridGPTWithHRM(hrm_config)
    hrm_trainer = HRMTrainer(hrm_model, hrm_config)
    hrm_history = hrm_trainer.train(data, num_steps=100, eval_interval=50)

    print("\nTraining Hybrid model (HRM + Transformer)...")
    hybrid_model = HybridGPTWithHRM(hybrid_config)
    hybrid_trainer = HRMTrainer(hybrid_model, hybrid_config)
    hybrid_history = hybrid_trainer.train(data, num_steps=100, eval_interval=50)

    # Compare
    print("\n" + "-" * 70)
    print("Comparison:")
    print("-" * 70)

    hrm_final = hrm_history[-1]
    hybrid_final = hybrid_history[-1]

    print(f"HRM-only:     loss={hrm_final['total_loss']:.4f}, steps={hrm_final['steps']}")
    print(f"Hybrid:       loss={hybrid_final['total_loss']:.4f}, steps={hybrid_final['steps']}")

    hrm_stats = hrm_model.get_stats()
    hybrid_stats = hybrid_model.get_stats()

    print(f"\nParameters:")
    print(f"  HRM-only:   {hrm_stats['total_parameters']:,}")
    print(f"  Hybrid:     {hybrid_stats['total_parameters']:,}")


def demo_puzzle_solving():
    """Demonstrate HRM on simple puzzle-like tasks."""
    print("\n" + "=" * 70)
    print("Puzzle Solving with HRM")
    print("=" * 70)

    # Simple pattern: a^n b^n (counting task)
    # e.g., "aaabbb", "aaaaabbbbb"
    patterns = []
    for n in range(1, 6):
        pattern = "a" * n + "b" * n
        patterns.append(pattern)

    print("\nLearning pattern: a^n b^n (balanced brackets-like)")
    print("Examples:", patterns[:3])

    # Create dataset
    all_text = " ".join(patterns)
    data, tokenizer = create_demo_dataset(all_text, seq_len=6)

    config = HRMIntegratedConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=48,
        num_heads=4,
        H_layers=2,  # More layers for reasoning
        L_layers=2,
        H_cycles=3,  # More cycles for complex patterns
        L_cycles=3,
        halt_max_steps=10,
        learning_rate=0.02,
        num_steps=300,
    )

    model = HybridGPTWithHRM(config)

    print("\nTraining on pattern task...")
    trainer = HRMTrainer(model, config)
    history = trainer.train(data, num_steps=200, eval_interval=100)

    # Test generalization
    print("\nTesting generalization to unseen lengths:")
    test_patterns = ["aaaaaabbbbbb", "aaabbb"]  # n=6 and n=3

    for pattern in test_patterns:
        tokens = tokenizer.encode(pattern[:3])  # Give first half
        result = model.forward_hrm(tokens)

        # Generate continuation
        full_result = model.generate(pattern[:3], tokenizer, max_length=10, use_hrm=True)
        print(f"  Input: '{pattern[:3]}' -> Generated: '{full_result}'")
        print(f"  Computation steps used: {result['steps']}")


def main():
    """Run all HRM demos."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL REASONING MODEL (HRM) EXAMPLES")
    print("Pure Python Implementation for microgpt")
    print("=" * 70)

    # Run demos
    demo_basic_hrm()
    demo_act_behavior()
    demo_comparison()
    demo_puzzle_solving()

    print("\n" + "=" * 70)
    print("All HRM demos completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • HRM uses hierarchical processing (H-level planning, L-level execution)")
    print("  • ACT enables dynamic computation based on problem complexity")
    print("  • No explicit chain-of-thought required")
    print("  • Pure Python implementation compatible with microgpt")
    print("\nFor more details, see:")
    print("  - hrm_adapter.py: Core HRM implementation")
    print("  - microgpt_hrm_integration.py: Integration layer")
    print("  - Paper: arXiv:2506.21734 (Wang et al., 2025)")


if __name__ == "__main__":
    main()
