"""
Command-line interface for microgpt.
Supports train, generate, and eval modes.
"""

import argparse
import sys
import random
from pathlib import Path

from .config import Config, ModelConfig, TrainingConfig, GenerationConfig
from .model import GPT
from .trainer import Trainer
from .data import DataLoader, CharTokenizer, BPETokenizer
from .checkpoint import CheckpointManager
from .logger import TrainingLogger, Metrics


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)


def train_command(args):
    """Execute training command."""
    # Load or create config
    if args.config:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = Config.from_yaml(args.config)
        else:
            config = Config.from_json(args.config)
    else:
        config = Config()

    # Override with command line args
    if args.steps:
        config.training.num_steps = args.steps
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.batch_size = args.batch_size

    set_seed(config.generation.seed)

    # Load data
    loader = DataLoader()
    if args.data:
        train_docs, val_docs = loader.load_file(args.data, config.training.val_split)
    else:
        train_docs, val_docs = loader.load_names(config.training.val_split)

    tokenizer = loader.tokenizer

    # Create model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
        use_gelu=config.model.use_gelu,
        use_layernorm=config.model.use_layernorm,
    )

    print(f"Model parameters: {model.num_params():,}")

    # Setup training
    trainer = Trainer(model, config.training)
    checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
    logger = TrainingLogger(config.log_dir, args.experiment_name)

    # Training callback
    def callback(step, loss, lr, val_loss=None):
        perplexity = 2.71828**loss  # e^loss approximation
        metrics = Metrics(step=step, loss=loss, perplexity=perplexity, learning_rate=lr)
        logger.log(metrics)

        # Save checkpoint
        if step % config.training.save_interval == 0:
            path = checkpoint_mgr.save_pickle(
                model.state_dict, config, step, loss, f"checkpoint_step_{step}.pkl"
            )
            logger.log_event("checkpoint_saved", {"path": str(path)})

        # Save best model
        if val_loss is not None and val_loss == min([m.loss for m in logger.metrics_history]):
            checkpoint_mgr.save_best(model.state_dict, config, step, val_loss, "pkl")
            logger.log_event("best_model_saved", {"val_loss": val_loss})

    # Train
    print("Starting training...")
    best_val_loss = trainer.train(
        train_docs,
        val_docs if val_docs else None,
        tokenizer.char_to_idx,
        tokenizer.bos_token,
        callback,
    )

    # Finalize
    logger.save_summary()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    # Save final model
    final_path = checkpoint_mgr.save_pickle(
        model.state_dict,
        config,
        config.training.num_steps,
        logger.metrics_history[-1].loss if logger.metrics_history else float("inf"),
        "final_model.pkl",
    )
    print(f"Final model saved to {final_path}")


def generate_command(args):
    """Execute generation command."""
    # Load checkpoint
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    if args.checkpoint:
        checkpoint = checkpoint_mgr.load_pickle(args.checkpoint)
    else:
        latest = checkpoint_mgr.get_latest()
        if not latest:
            print("No checkpoint found!")
            sys.exit(1)
        checkpoint = checkpoint_mgr.load_pickle(latest)

    # Reconstruct config
    config_data = checkpoint["config"]
    config = Config(
        model=ModelConfig(**config_data.get("model", {})),
        training=TrainingConfig(**config_data.get("training", {})),
        generation=GenerationConfig(**config_data.get("generation", {})),
    )

    # Override with command line args
    if args.temperature:
        config.generation.temperature = args.temperature
    if args.num_samples:
        config.generation.num_samples = args.num_samples
    if args.max_length:
        config.generation.max_length = args.max_length
    if args.top_k:
        config.generation.top_k = args.top_k
    if args.top_p:
        config.generation.top_p = args.top_p
    if args.seed:
        config.generation.seed = args.seed

    set_seed(config.generation.seed)

    # Load data to get tokenizer
    loader = DataLoader()
    loader.load_names(val_split=0)
    tokenizer = loader.tokenizer

    # Create and load model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=0.0,  # No dropout during inference
        use_gelu=config.model.use_gelu,
        use_layernorm=config.model.use_layernorm,
    )

    # Load weights
    for key, matrix in checkpoint["state_dict"].items():
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                model.state_dict[key][i][j].data = val

    model.set_training(False)

    # Generate
    print(f"\n--- Generating {config.generation.num_samples} samples ---")
    print(
        f"Temperature: {config.generation.temperature}, Top-k: {config.generation.top_k}, Top-p: {config.generation.top_p}"
    )
    print("-" * 50)

    for i in range(config.generation.num_samples):
        tokens = model.generate(
            token_id=tokenizer.bos_token,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
        )
        text = tokenizer.decode(tokens)
        print(f"Sample {i+1:2d}: {text}")

    # Interactive mode
    if args.interactive:
        print("\n--- Interactive Mode (type 'quit' to exit) ---")
        while True:
            prompt = input("\nPrompt (or 'quit'): ").strip()
            if prompt.lower() == "quit":
                break

            # Encode prompt
            prompt_tokens = tokenizer.encode(prompt)

            # Generate continuation
            keys = [[] for _ in range(model.n_layer)]
            values = [[] for _ in range(model.n_layer)]

            # Feed prompt through model
            for pos_id, token_id in enumerate(prompt_tokens):
                _ = model.forward(token_id, pos_id, keys, values)

            # Generate continuation
            continuation = model.generate(
                token_id=prompt_tokens[-1] if prompt_tokens else tokenizer.bos_token,
                max_length=config.generation.max_length,
                temperature=config.generation.temperature,
                top_k=config.generation.top_k,
                top_p=config.generation.top_p,
                keys=keys,
                values=values,
            )

            result = tokenizer.decode(continuation)
            print(f"Generated: {result}")


def eval_command(args):
    """Execute evaluation command."""
    # Load checkpoint
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    if args.checkpoint:
        checkpoint = checkpoint_mgr.load_pickle(args.checkpoint)
    else:
        latest = checkpoint_mgr.get_latest()
        if not latest:
            print("No checkpoint found!")
            sys.exit(1)
        checkpoint = checkpoint_mgr.load_pickle(latest)

    # Load data
    loader = DataLoader()
    if args.data:
        _, val_docs = loader.load_file(args.data, val_split=1.0)
    else:
        _, val_docs = loader.load_names(val_split=1.0)

    tokenizer = loader.tokenizer

    # Reconstruct config
    config_data = checkpoint["config"]
    config = Config(
        model=ModelConfig(**config_data.get("model", {})),
        training=TrainingConfig(**config_data.get("training", {})),
        generation=GenerationConfig(**config_data.get("generation", {})),
    )

    # Create and load model
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
    perplexity = 2.71828**val_loss

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

    # Additional metrics
    if args.detailed:
        print("\n--- Detailed Metrics ---")

        # Sequence length distribution
        lengths = [len(doc) for doc in val_docs]
        print(f"Avg sequence length: {sum(lengths)/len(lengths):.1f}")
        print(f"Min/Max length: {min(lengths)}/{max(lengths)}")

        # Vocabulary coverage
        all_chars = set("".join(val_docs))
        coverage = len(all_chars) / len(tokenizer.char_to_idx) * 100
        print(f"Vocab coverage: {coverage:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="microgpt - Minimal GPT in pure Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python cli.py train --steps 2000 --lr 0.005
  
  # Generate text from trained model
  python cli.py generate --temperature 0.7 --num-samples 10
  
  # Evaluate model on validation set
  python cli.py eval --detailed
  
  # Interactive generation
  python cli.py generate --interactive
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Config file (YAML or JSON)")
    train_parser.add_argument("--data", type=str, help="Training data file")
    train_parser.add_argument("--steps", type=int, help="Number of training steps")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument(
        "--experiment-name", type=str, default="experiment", help="Experiment name for logging"
    )

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--checkpoint", type=str, help="Checkpoint file to load")
    gen_parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    gen_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    gen_parser.add_argument("--num-samples", type=int, help="Number of samples")
    gen_parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    gen_parser.add_argument("--top-k", type=int, help="Top-k sampling (0=disabled)")
    gen_parser.add_argument("--top-p", type=float, help="Top-p/nucleus sampling (1.0=disabled)")
    gen_parser.add_argument("--seed", type=int, help="Random seed")
    gen_parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, help="Checkpoint file to load")
    eval_parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    eval_parser.add_argument("--data", type=str, help="Evaluation data file")
    eval_parser.add_argument("--detailed", action="store_true", help="Show detailed metrics")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "eval":
        eval_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
