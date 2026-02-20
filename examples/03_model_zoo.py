"""
Example 3: Using Model Zoo
Create and compare different model configurations.
"""

from model_zoo import ModelZoo, create_model
from benchmark import ComparativeBenchmark


def main():
    print("=" * 60)
    print("Example 3: Model Zoo")
    print("=" * 60)

    # List available models
    print("\n1. Available Model Configurations")
    print("-" * 40)
    ModelZoo.list_models()

    # Create different models
    print("\n2. Creating Models")
    print("-" * 40)

    configs = ["tiny", "small", "medium"]

    for config_name in configs:
        print(f"\n   Creating {config_name} model...")
        model, config = create_model(config_name)
        print(f"   Parameters: {model.num_params():,}")
        print(f"   Layers: {model.n_layer}")
        print(f"   Embedding dim: {model.n_embd}")

    # Compare configurations
    print("\n3. Comparative Benchmark")
    print("-" * 40)
    print("   (This would run speed benchmarks on each config)")
    print("   Skipping for quick demo...")

    # Custom configuration
    print("\n4. Custom Configuration")
    print("-" * 40)

    from config import Config, ModelConfig, TrainingConfig

    custom_config = Config(
        model=ModelConfig(
            n_layer=3, n_embd=48, n_head=6, block_size=48, dropout=0.1, use_gelu=True
        ),
        training=TrainingConfig(
            num_steps=3000, learning_rate=0.005, lr_schedule="cosine", warmup_steps=300
        ),
    )

    print("   Custom config created!")
    print(f"   Model layers: {custom_config.model.n_layer}")
    print(f"   Training steps: {custom_config.training.num_steps}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
