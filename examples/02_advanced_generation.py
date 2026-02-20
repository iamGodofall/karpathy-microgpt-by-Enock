"""
Example 2: Advanced Generation Techniques
Demonstrates different sampling methods.
"""

import random
from model import GPT
from data import DataLoader
from advanced_features import (
    BeamSearchDecoder,
    ContrastiveSearchDecoder,
    RepetitionPenaltyLogitsProcessor,
    TopA_Sampling,
    MirostatSampling
)


def main():
    print("="*60)
    print("Example 2: Advanced Generation")
    print("="*60)
    
    # Load data and create model
    loader = DataLoader()
    loader.load_names(val_split=0)
    
    model = GPT(
        vocab_size=loader.tokenizer.vocab_size,
        block_size=32,
        n_layer=2,
        n_embd=32,
        n_head=4
    )
    model.set_training(False)
    
    start_token = loader.tokenizer.bos_token
    
    print("\n1. Temperature Sampling")
    print("-" * 40)
    for temp in [0.5, 0.8, 1.2]:
        tokens = model.generate(start_token, max_length=20, temperature=temp)
        text = loader.tokenizer.decode(tokens)
        print(f"Temp {temp}: {text}")
    
    print("\n2. Top-k Sampling")
    print("-" * 40)
    for k in [5, 20, 50]:
        tokens = model.generate(start_token, max_length=20, temperature=0.8, top_k=k)
        text = loader.tokenizer.decode(tokens)
        print(f"Top-{k}: {text}")
    
    print("\n3. Top-p (Nucleus) Sampling")
    print("-" * 40)
    for p in [0.5, 0.9, 0.95]:
        tokens = model.generate(start_token, max_length=20, temperature=0.8, top_p=p)
        text = loader.tokenizer.decode(tokens)
        print(f"Top-p {p}: {text}")
    
    print("\n4. Beam Search")
    print("-" * 40)
    decoder = BeamSearchDecoder(beam_width=3, max_length=20)
    tokens, score = decoder.decode(model, start_token, temperature=0.8)
    text = loader.tokenizer.decode(tokens)
    print(f"Beam search (score={score:.2f}): {text}")
    
    print("\n5. Contrastive Search")
    print("-" * 40)
    decoder = ContrastiveSearchDecoder(k=5, alpha=0.6)
    tokens = decoder.decode(model, start_token, max_length=20)
    text = loader.tokenizer.decode(tokens)
    print(f"Contrastive: {text}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == '__main__':
    main()
