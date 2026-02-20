"""
Example 1: Basic Training
Train a simple model on the names dataset.
"""

import random
from model import GPT
from data import DataLoader
from trainer import Trainer, TrainingConfig


def main():
    print("="*60)
    print("Example 1: Basic Training")
    print("="*60)
    
    # 1. Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    train_docs, val_docs = loader.load_names(val_split=0.1)
    print(f"   Train: {len(train_docs)} documents")
    print(f"   Val: {len(val_docs)} documents")
    print(f"   Vocab size: {loader.tokenizer.vocab_size}")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = GPT(
        vocab_size=loader.tokenizer.vocab_size,
        block_size=16,
        n_layer=2,
        n_embd=32,
        n_head=4,
        dropout=0.1
    )
    print(f"   Parameters: {model.num_params():,}")
    
    # 3. Configure training
    print("\n3. Configuring training...")
    config = TrainingConfig(
        num_steps=500,
        learning_rate=0.01,
        lr_schedule='linear',
        eval_interval=100
    )
    
    # 4. Train
    print("\n4. Training...")
    trainer = Trainer(model, config)
    
    for step in range(config.num_steps):
        # Sample document
        doc = random.choice(train_docs)
        tokens = [loader.tokenizer.bos_token] + [
            loader.tokenizer.char_to_idx.get(ch, loader.tokenizer.bos_token)
            for ch in doc
        ] + [loader.tokenizer.bos_token]
        
        if len(tokens) < 2:
            continue
        
        # Training step
        loss = trainer.train_step(tokens, step)
        
        if step % 100 == 0:
            print(f"   Step {step:4d}: loss = {loss:.4f}")
    
    # 5. Generate
    print("\n5. Generating samples...")
    model.set_training(False)
    
    for i in range(5):
        tokens = model.generate(
            loader.tokenizer.bos_token,
            max_length=15,
            temperature=0.8
        )
        name = loader.tokenizer.decode(tokens)
        print(f"   Sample {i+1}: {name}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == '__main__':
    main()
