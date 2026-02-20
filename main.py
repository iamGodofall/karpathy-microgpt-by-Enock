"""
Main entry point for microgpt ecosystem.
Unified interface to all features.
"""

import argparse
import sys
from pathlib import Path

# Import all modules
from config import Config, ModelConfig, TrainingConfig, GenerationConfig
from model import GPT
from trainer import Trainer
from data import CharTokenizer, load_data
from checkpoint import CheckpointManager
from logger import TrainingLogger
from model_zoo import create_model, list_models
from tokenizers import create_tokenizer, list_tokenizers


def train_command(args):
    """Execute training command."""
    print("=" * 60)
    print("microgpt Training")
    print("=" * 60)
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with CLI args
    if args.model:
        config.model = ModelConfig(**args.model)
    if args.training:
        config.training = TrainingConfig(**args.training)
    
    # Load data
    print(f"Loading data from {config.data_path}...")
    docs = load_data(config.data_path)
    print(f"Loaded {len(docs)} documents")
    
    # Split train/val
    split_idx = int(len(docs) * (1 - config.training.val_split))
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(docs)
    
    # Create model
    print(f"Creating model: {config.model.n_layer} layers, "
          f"{config.model.n_embd} dim, {config.model.n_head} heads")
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
        use_gelu=config.model.use_gelu,
        use_layernorm=config.model.use_layernorm
    )
    print(f"Model parameters: {model.num_params():,}")
    
    # Setup training
    trainer = Trainer(model, config.training)
    logger = TrainingLogger(config.log_dir)
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    
    # Training callback
    def callback(step, loss, lr, val_loss=None):
        from logger import Metrics
        import math
        
        perplexity = math.exp(loss) if loss < 10 else float('inf')
        metrics = Metrics(step=step, loss=loss, perplexity=perplexity, 
                         learning_rate=lr)
        logger.log(metrics)
        
        # Save checkpoint
        if step % config.training.save_interval == 0:
            path = checkpoint_manager.save_pickle(
                model.state_dict, config, step, loss
            )
            logger.log_event('checkpoint', {'path': str(path)})
    
    # Train
    print("\nStarting training...")
    best_loss = trainer.train(train_docs, val_docs, 
                               tokenizer.char_to_idx, 
                               tokenizer.bos_token,
                               callback=callback)
    
    # Save final
    checkpoint_manager.save_best(model.state_dict, config, 
                                  config.training.num_steps, best_loss)
    logger.save_summary()
    
    print(f"\nTraining complete! Best validation loss: {best_loss:.4f}")


def generate_command(args):
    """Execute generation command."""
    print("=" * 60)
    print("microgpt Generation")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    
    if args.checkpoint:
        checkpoint = checkpoint_manager.load_pickle(args.checkpoint)
    else:
        latest = checkpoint_manager.get_latest()
        if not latest:
            print("No checkpoint found!")
            return
        checkpoint = checkpoint_manager.load_pickle(latest)
    
    # Reconstruct model
    config_dict = checkpoint['config']
    model_config = ModelConfig(**config_dict.get('model', {}))
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    # Would load vocab from checkpoint
    
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=model_config.block_size,
        n_layer=model_config.n_layer,
        n_embd=model_config.n_embd,
        n_head=model_config.n_head
    )
    
    # Load weights
    for name, matrix_data in checkpoint['state_dict'].items():
        for i, row in enumerate(matrix_data):
            for j, val in enumerate(row):
                model.state_dict[name][i][j].data = val
    
    print(f"Loaded model from step {checkpoint['step']}")
    print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    
    # Generate
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length,
        num_samples=args.num_samples
    )
    
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"Temperature: {gen_config.temperature}, Top-k: {gen_config.top_k}, Top-p: {gen_config.top_p}")
    print("-" * 60)
    
    for i in range(gen_config.num_samples):
        tokens = model.generate(
            tokenizer.bos_token,
            max_length=gen_config.max_length,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p
        )
        
        text = tokenizer.decode(tokens)
        print(f"\nSample {i+1}: {text}")


def eval_command(args):
    """Execute evaluation command."""
    print("=" * 60)
    print("microgpt Evaluation")
    print("=" * 60)
    
    from evaluation import run_evaluation
    
    # Load model
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    latest = checkpoint_manager.get_latest()
    
    if not latest:
        print("No checkpoint found!")
        return
    
    checkpoint = checkpoint_manager.load_pickle(latest)
    
    # Reconstruct
    tokenizer = CharTokenizer()
    model_config = ModelConfig(**checkpoint['config'].get('model', {}))
    
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=model_config.block_size,
        n_layer=model_config.n_layer,
        n_embd=model_config.n_embd,
        n_head=model_config.n_head
    )
    
    # Load weights
    for name, matrix_data in checkpoint['state_dict'].items():
        for i, row in enumerate(matrix_data):
            for j, val in enumerate(row):
                model.state_dict[name][i][j].data = val
    
    # Run evaluation
    results = run_evaluation(model, tokenizer, args.test_file)
    
    return results


def chat_command(args):
    """Execute interactive chat."""
    print("=" * 60)
    print("microgpt Chat")
    print("=" * 60)
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 60)
    
    # Load model
    from chat import ChatSession
    
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    session = ChatSession(checkpoint_manager, args.checkpoint)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'help':
                print("Commands: quit, help, clear, save, load")
                continue
            
            if user_input.lower() == 'clear':
                session.clear_history()
                print("History cleared")
                continue
            
            response = session.chat(user_input)
            print(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def zoo_command(args):
    """List available model configurations."""
    print("=" * 60)
    print("microgpt Model Zoo")
    print("=" * 60)
    
    models = list_models()
    
    print(f"\nAvailable models ({len(models)}):")
    for name, config in models.items():
        params = estimate_params(config)
        print(f"  {name:15} | {config['n_layer']}L/{config['n_embd']}D/{config['n_head']}H | ~{params:,} params")
    
    if args.info:
        print(f"\nDetails for '{args.info}':")
        config = models.get(args.info)
        if config:
            for k, v in config.items():
                print(f"  {k}: {v}")


def estimate_params(config: dict) -> int:
    """Estimate parameter count."""
    v = config.get('vocab_size', 256)
    d = config.get('n_embd', 16)
    l = config.get('n_layer', 1)
    h = config.get('n_head', 4)
    
    # Embeddings
    params = v * d * 2  # wte + wpe
    
    # Per layer
    params += l * (
        4 * d * d +  # 4 linear in attention
        2 * 4 * d * d  # 2 MLP layers (4x expansion)
    )
    
    # Output
    params += v * d
    
    return params


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='microgpt - Pure Python GPT Ecosystem',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --config config.yaml
  
  # Generate text
  python main.py generate --checkpoint best_model.pkl --num-samples 10
  
  # Interactive chat
  python main.py chat
  
  # Evaluate model
  python main.py eval --test-file test.txt
  
  # List model zoo
  python main.py zoo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Config file')
    train_parser.add_argument('--data', type=str, default='input.txt', help='Training data')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    gen_parser.add_argument('--temperature', type=float, default=0.7)
    gen_parser.add_argument('--top-k', type=int, default=40)
    gen_parser.add_argument('--top-p', type=float, default=0.9)
    gen_parser.add_argument('--max-length', type=int, default=100)
    gen_parser.add_argument('--num-samples', type=int, default=5)
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    eval_parser.add_argument('--test-file', type=str, help='Test data file')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    
    # Zoo command
    zoo_parser = subparsers.add_parser('zoo', help='Model zoo')
    zoo_parser.add_argument('--info', type=str, help='Show model details')
    
    # Parse
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute
    commands = {
        'train': train_command,
        'generate': generate_command,
        'eval': eval_command,
        'chat': chat_command,
        'zoo': zoo_command,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
