"""
Data loading and tokenization for microgpt.
Includes character-level and BPE tokenization.
"""

import os
import random
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.bos_token: int = 0
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        chars = sorted(set(''.join(texts)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.bos_token = len(chars)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx.get(ch, self.bos_token) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.idx_to_char.get(t, '') for t in tokens if t != self.bos_token)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including BOS token)."""
        return len(self.char_to_idx) + 1


class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.base_vocab_size = 256  # Start with bytes
        self.merges: List[Tuple[int, int]] = []
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Count frequency of adjacent pairs."""
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
    
    def _merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Merge all occurrences of pair into new token idx."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def fit(self, texts: List[str]):
        """Train BPE on texts."""
        # Start with byte encoding
        all_ids = []
        for text in texts:
            all_ids.extend(list(text.encode('utf-8')))
        
        num_merges = self.vocab_size - self.base_vocab_size
        
        for i in range(num_merges):
            stats = self._get_stats(all_ids)
            if not stats:
                break
            
            best = max(stats, key=stats.get)
            idx = self.base_vocab_size + i
            
            all_ids = self._merge(all_ids, best, idx)
            self.merges.append(best)
            self.vocab[idx] = self.vocab[best[0]] + self.vocab[best[1]]
            
            if (i + 1) % 100 == 0:
                print(f"BPE merge {i + 1}/{num_merges}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges."""
        ids = list(text.encode('utf-8'))
        
        for pair, new_idx in zip(self.merges, range(self.base_vocab_size, self.vocab_size)):
            ids = self._merge(ids, pair, new_idx)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode('utf-8', errors='replace')
    
    def save(self, path: str):
        """Save tokenizer to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'merges': self.merges,
                'vocab': {k: v.decode('utf-8', errors='replace') for k, v in self.vocab.items()}
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tok = cls(data['vocab_size'])
        tok.merges = [tuple(m) for m in data['merges']]
        tok.vocab = {int(k): v.encode('utf-8') for k, v in data['vocab'].items()}
        return tok


class DataLoader:
    """Load and preprocess datasets."""
    
    def __init__(self, tokenizer: Optional[CharTokenizer] = None):
        self.tokenizer = tokenizer or CharTokenizer()
        self.train_docs: List[str] = []
        self.val_docs: List[str] = []
    
    def load_file(self, path: str, val_split: float = 0.1, 
                  min_length: int = 1, max_length: int = 1000) -> Tuple[List[str], List[str]]:
        """Load and split data from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Filter by length
        lines = [line for line in lines if min_length <= len(line) <= max_length]
        
        # Shuffle and split
        random.shuffle(lines)
        split_idx = int(len(lines) * (1 - val_split))
        
        self.train_docs = lines[:split_idx]
        self.val_docs = lines[split_idx:] if val_split > 0 else []
        
        # Fit tokenizer
        self.tokenizer.fit(self.train_docs)
        
        print(f"Loaded {len(self.train_docs)} train, {len(self.val_docs)} val documents")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        
        return self.train_docs, self.val_docs
    
    def load_names(self, val_split: float = 0.1) -> Tuple[List[str], List[str]]:
        """Load default names dataset."""
        path = 'input.txt'
        
        if not os.path.exists(path):
            import urllib.request
            url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
            print(f"Downloading dataset from {url}...")
            urllib.request.urlretrieve(url, path)
        
        return self.load_file(path, val_split)
    
    def get_batch(self, batch_size: int, split: str = 'train') -> List[List[int]]:
        """Get a batch of tokenized sequences."""
        docs = self.train_docs if split == 'train' else self.val_docs
        
        batch = []
        for _ in range(batch_size):
            doc = random.choice(docs)
            tokens = self.tokenizer.encode(doc)
            batch.append(tokens)
        
        return batch
    
    def save(self, path: str):
        """Save dataset and tokenizer."""
        import json
        data = {
            'train': self.train_docs,
            'val': self.val_docs,
            'tokenizer_type': 'char',
            'vocab': self.tokenizer.char_to_idx if isinstance(self.tokenizer, CharTokenizer) else None
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'DataLoader':
        """Load dataset and tokenizer."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        loader = cls()
        loader.train_docs = data['train']
        loader.val_docs = data['val']
        
        if data.get('tokenizer_type') == 'char' and data.get('vocab'):
            loader.tokenizer = CharTokenizer()
            loader.tokenizer.char_to_idx = data['vocab']
            loader.tokenizer.idx_to_char = {i: ch for ch, i in data['vocab'].items()}
            loader.tokenizer.bos_token = len(data['vocab'])
        
        return loader


def preprocess_text(text: str, lowercase: bool = True, 
                   remove_punctuation: bool = False) -> str:
    """Basic text preprocessing."""
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
