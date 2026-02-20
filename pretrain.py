"""
Large-scale pretraining script for microgpt.
Supports multi-file datasets, checkpoint resumption, and distributed training.
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Optional

from model import GPT
from data import CharTokenizer, DataLoader
from trainer import Trainer, TrainingConfig
from config import Config
from checkpoint import CheckpointManager
from logger import TrainingLogger, Metrics
from benchmark import MemoryProfiler


def load_large_dataset(data_dir: str, val_split: float = 0.01) -> tuple:
    """
    Load large dataset from directory of text files.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    all_texts = []
    
    # Load all .txt files
    for txt_file in data_path.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split into chunks if very large
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            all_texts.extend(chunks)
    
    # Shuffle and split
    random.shuffle(all_texts)
    split_idx = int(len(all_texts) * (1 - val_split))
    
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    print(f"Loaded {len(train_texts)} training chunks, {len(val_texts)} validation chunks")
    
    return train_texts, val_texts


def pretrain(args):
    """
    Main pretraining function.
    """
    print("="*70)
    print("MICROGPT PRETRAINING")
    print("="*70)
    
    # Load or create config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command line args
    if args.steps:
        config.training.num_steps = args.steps
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    print(f"\nConfiguration:")
    print(f"  Steps: {config.training.num_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Model: {config.model.n_layer} layers, {config.model.n_embd} dims")
    
    # Load data
    print("\nLoading dataset...")
    if args.data_dir:
        train_docs, val_docs = load_large_dataset(args.data_dir, config.training.val_split)
    else:
        loader = DataLoader()
        train_docs, val_docs = loader.load_names(config.training.val_split)
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(train_docs)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    print("\nInitializing model...")
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
    
    # Memory profiling
    profiler = MemoryProfiler(model)
    profiler.print_memory_report()
    
    # Setup checkpointing
    checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        latest = checkpoint_mgr.get_latest()
        if latest:
            print(f"\nResuming from checkpoint: {latest}")
            checkpoint = checkpoint_mgr.load_pickle(latest)
            start_step = checkpoint['step']
            # Load weights
            for name, matrix_data in checkpoint['state_dict'].items():
                for i, row in enumerate(matrix_data):
                    for j, val in enumerate(row):
                        model.state_dict[name][i][j].data = val
        else:
            print("\nNo checkpoint found, starting from scratch")
    
    # Setup logging
    logger = TrainingLogger(config.log_dir, args.experiment_name)
    
    # Create trainer
    trainer = Trainer(model, config.training)
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_val_loss = float('inf')
    
    def log_callback(step, loss, lr, val_loss=None):
        # Calculate perplexity
        perplexity = 2.71828 ** loss
        
        metrics = Metrics(
            step=step,
            loss=loss,
            perplexity=perplexity,
            learning_rate=lr
        )
        logger.log(metrics)
        
        # Save checkpoint periodically
        if step % config.training.save_interval == 0:
            checkpoint_mgr.save_pickle(
                model.state_dict,
                config,
                step,
                loss,
                f"checkpoint_step_{step}.pkl"
            )
            logger.log_event('checkpoint_saved', {'step': step, 'loss': loss})
        
        # Save best model
        if val_loss and val_loss < best_val_loss:
            checkpoint_mgr.save_best(model.state_dict, config, step, val_loss, 'pkl')
            logger.log_event('best_model_saved', {'step': step, 'val_loss': val_loss})
    
    # Run training
    final_val_loss = trainer.train(
        train_docs, 
        val_docs if val_docs else None,
        tokenizer.char_to_idx,
        tokenizer.bos_token,
        callback=log_callback
    )
    
    # Final checkpoint
    checkpoint_mgr.save_pickle(
        model.state_dict,
        config,
        config.training.num_steps,
        final_val_loss if final_val_loss != float('inf') else 0,
        "final_model.pkl"
    )
    
    # Save summary
    summary = logger.save_summary()
    
    print("\n" + "="*70)
    print("PRETRAINING COMPLETE")
    print("="*70)
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: {config.log_dir}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Pretrain microgpt on large dataset')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Directory containing training data')
    parser.add_argument('--steps', type=int, help='Number of training steps')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--experiment-name', type=str, default='pretrain', 
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    pretrain(args)


if __name__ == '__main__':
    main()
