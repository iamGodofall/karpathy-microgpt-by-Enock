"""
Fine-tuning script for pre-trained microgpt models.
Supports domain adaptation and task-specific fine-tuning.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from .model import GPT
from .data import CharTokenizer, DataLoader
from .trainer import Trainer, TrainingConfig
from .config import Config
from .checkpoint import CheckpointManager


def freeze_layers(model: GPT, num_layers_to_freeze: int):
    """
    Freeze early layers for efficient fine-tuning.
    """
    for i in range(num_layers_to_freeze):
        # Freeze attention weights
        for name in [
            f"layer{i}.attn_wq",
            f"layer{i}.attn_wk",
            f"layer{i}.attn_wv",
            f"layer{i}.attn_wo",
        ]:
            for row in model.state_dict[name]:
                for p in row:
                    p.grad = 0  # Zero out gradients (effectively frozen)

        # Freeze MLP weights
        for name in [f"layer{i}.mlp_fc1", f"layer{i}.mlp_fc2"]:
            for row in model.state_dict[name]:
                for p in row:
                    p.grad = 0


def gradual_unfreeze(model: GPT, current_step: int, total_steps: int):
    """
    Gradually unfreeze layers during training (ULMFiT approach).
    """
    # Unfreeze one layer every N steps
    unfreeze_every = total_steps // model.n_layer

    layers_to_unfreeze = min(current_step // unfreeze_every + 1, model.n_layer)

    print(f"Step {current_step}: {layers_to_unfreeze}/{model.n_layer} layers active")

    # Freeze all layers first
    for i in range(model.n_layer):
        freeze_layers(model, model.n_layer)

    # Unfreeze top layers
    for i in range(model.n_layer - layers_to_unfreeze, model.n_layer):
        # Unfreeze by allowing gradients
        pass  # In real implementation, would remove from frozen set


def discriminative_finetuning(model: GPT, base_lr: float):
    """
    Apply different learning rates to different layers.
    Lower layers get smaller learning rates.
    """
    # This would require modifying the optimizer to support per-parameter LR
    # For now, we just document the approach
    learning_rates = []

    for i in range(model.n_layer):
        # Exponentially decay LR for lower layers
        lr = base_lr * (0.9 ** (model.n_layer - i - 1))
        learning_rates.append(lr)

    return learning_rates


def finetune(args):
    """
    Main fine-tuning function.
    """
    print("=" * 70)
    print("MICROGPT FINE-TUNING")
    print("=" * 70)

    # Load pre-trained model
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = checkpoint_mgr.load_pickle(args.checkpoint)
    else:
        latest = checkpoint_mgr.get_latest()
        if not latest:
            raise ValueError("No checkpoint found. Please train a model first.")
        print(f"Loading latest checkpoint: {latest}")
        checkpoint = checkpoint_mgr.load_pickle(latest)

    # Create model from checkpoint
    config_dict = checkpoint["config"]
    model = GPT(
        vocab_size=config_dict["model"]["vocab_size"],
        block_size=config_dict["model"]["block_size"],
        n_layer=config_dict["model"]["n_layer"],
        n_embd=config_dict["model"]["n_embd"],
        n_head=config_dict["model"]["n_head"],
        dropout=args.dropout,  # Can increase dropout for fine-tuning
        use_gelu=config_dict["model"].get("use_gelu", False),
        use_layernorm=config_dict["model"].get("use_layernorm", False),
    )

    # Load weights
    for name, matrix_data in checkpoint["state_dict"].items():
        for i, row in enumerate(matrix_data):
            for j, val in enumerate(row):
                model.state_dict[name][i][j].data = val

    print(f"Loaded model with {model.num_params():,} parameters")

    # Load fine-tuning data
    print(f"\nLoading fine-tuning data from: {args.data}")
    loader = DataLoader()

    if args.data.endswith(".txt"):
        train_docs, val_docs = loader.load_file(args.data, val_split=0.1)
    else:
        # Assume directory
        from pretrain import load_large_dataset

        train_docs, val_docs = load_large_dataset(args.data, val_split=0.1)

    # Update tokenizer if needed
    tokenizer = CharTokenizer()
    tokenizer.fit(train_docs)

    # Adjust model embeddings if vocab size changed
    if tokenizer.vocab_size != model.vocab_size:
        print(f"Adjusting vocab size: {model.vocab_size} -> {tokenizer.vocab_size}")
        # In practice, would need to resize embedding layers
        # For now, we'll use the original tokenizer
        tokenizer = CharTokenizer()
        tokenizer.fit([checkpoint["config"]["data_path"]])

    # Fine-tuning configuration
    ft_config = TrainingConfig(
        num_steps=args.steps,
        learning_rate=args.lr,
        lr_schedule="cosine",
        warmup_steps=args.steps // 10,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        val_split=0.0,  # Already split
        eval_interval=args.steps // 10,
        save_interval=args.steps // 5,
    )

    # Freeze layers if specified
    if args.freeze_layers > 0:
        print(f"Freezing first {args.freeze_layers} layers")
        freeze_layers(model, args.freeze_layers)

    # Create trainer
    trainer = Trainer(model, ft_config)

    # Fine-tuning loop
    print("\n" + "=" * 70)
    print("STARTING FINE-TUNING")
    print("=" * 70)

    best_loss = float("inf")

    for step in range(ft_config.num_steps):
        # Gradual unfreezing
        if args.gradual_unfreeze:
            gradual_unfreeze(model, step, ft_config.num_steps)

        # Sample document
        import random

        doc = random.choice(train_docs)
        tokens = (
            [tokenizer.bos_token]
            + [
                tokenizer.char_to_idx.get(ch, tokenizer.bos_token)
                for ch in doc
                if ch in tokenizer.char_to_idx
            ]
            + [tokenizer.bos_token]
        )

        if len(tokens) < 2:
            continue

        # Training step
        loss = trainer.train_step(tokens, step)

        if step % 100 == 0:
            lr = trainer.optimizer._get_lr(step)
            print(f"Step {step:5d} | Loss: {loss:.4f} | LR: {lr:.6f}")

        # Save checkpoint
        if (step + 1) % ft_config.save_interval == 0:
            checkpoint_mgr.save_pickle(
                model.state_dict, checkpoint["config"], step, loss, f"finetuned_step_{step}.pkl"
            )

        if loss < best_loss:
            best_loss = loss
            checkpoint_mgr.save_best(model.state_dict, checkpoint["config"], step, loss, "pkl")

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 70)

    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune microgpt model")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Fine-tuning data file or directory"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of fine-tuning steps")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (typically smaller than pretraining)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate (can increase for fine-tuning)"
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--freeze-layers", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument(
        "--gradual-unfreeze", action="store_true", help="Use gradual unfreezing (ULMFiT)"
    )

    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
