"""
Data pipeline for large-scale training.
Handles data loading, preprocessing, batching, and streaming.
"""

import random
import json
from typing import Iterator, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import threading
from collections import deque


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    batch_size: int = 32
    seq_length: int = 512
    shuffle: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    buffer_size: int = 1000


class DataLoader:
    """Efficient data loader with prefetching."""
    
    def __init__(self, dataset: List, config: DataConfig = None, tokenizer=None):
        self.dataset = dataset
        self.config = config or DataConfig()
        self.tokenizer = tokenizer
        self._buffer = deque(maxlen=self.config.buffer_size)
        self._worker_threads = []
        self._stop_event = threading.Event()
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))
        
        if self.config.shuffle:
            random.shuffle(indices)
        
        batch = []
        for idx in indices:
            item = self.dataset[idx]
            
            # Tokenize if needed
            if self.tokenizer and isinstance(item, str):
                item = self.tokenizer.encode(item)
            
            batch.append(item)
            
            if len(batch) == self.config.batch_size:
                yield self._collate(batch)
                batch = []
        
        # Yield remaining
        if batch:
            yield self._collate(batch)
    
    def _collate(self, batch: List) -> Dict[str, Any]:
        """Collate batch into tensors."""
        # Find max length
        max_len = max(len(x) for x in batch)
        max_len = min(max_len, self.config.seq_length)
        
        # Pad sequences
        padded = []
        for x in batch:
            if len(x) > max_len:
                x = x[:max_len]
            else:
                x = x + [0] * (max_len - len(x))
            padded.append(x)
        
        # Create input/target pairs
        inputs = [x[:-1] for x in padded]
        targets = [x[1:] for x in padded]
        
        return {
            'input': inputs,
            'target': targets,
            'lengths': [len(x) for x in batch],
        }
    
    def start_prefetch(self):
        """Start prefetching in background."""
        def prefetch_worker():
            for batch in self:
                if self._stop_event.is_set():
                    break
                self._buffer.append(batch)
        
        for _ in range(self.config.num_workers):
            t = threading.Thread(target=prefetch_worker, daemon=True)
            t.start()
            self._worker_threads.append(t)
    
    def stop_prefetch(self):
        """Stop prefetching."""
        self._stop_event.set()
        for t in self._worker_threads:
            t.join(timeout=1)


class DataProcessor:
    """Process and transform data."""
    
    def __init__(self):
        self.transforms: List[Callable] = []
    
    def add_transform(self, transform: Callable):
        """Add a transform."""
        self.transforms.append(transform)
    
    def process(self, text: str) -> str:
        """Apply all transforms."""
        for transform in self.transforms:
            text = transform(text)
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace."""
        return ' '.join(text.split())
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert to lowercase."""
        return text.lower()
    
    @staticmethod
    def truncate(max_length: int) -> Callable:
        """Create truncate transform."""
        def transform(text: str) -> str:
            return text[:max_length]
        return transform


class StreamingDataset:
    """Stream data from files without loading all into memory."""
    
    def __init__(self, file_path: str, tokenizer=None, max_seq_length: int = 512):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __iter__(self) -> Iterator[List[int]]:
        """Stream tokens."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if self.tokenizer:
                    tokens = self.tokenizer.encode(line)
                    if len(tokens) > self.max_seq_length:
                        # Sliding window
                        for i in range(0, len(tokens) - self.max_seq_length, self.max_seq_length // 2):
                            yield tokens[i:i + self.max_seq_length]
                    else:
                        yield tokens
                else:
                    yield list(line.encode('utf-8'))
    
    def batch_iter(self, batch_size: int) -> Iterator[List[List[int]]]:
        """Iterate in batches."""
        batch = []
        for tokens in self:
            batch.append(tokens)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class DataAugmenter:
    """Data augmentation for text."""
    
    @staticmethod
    def random_mask(tokens: List[int], mask_token: int = 0, mask_prob: float = 0.15) -> List[int]:
        """Random token masking (BERT-style)."""
        masked = tokens.copy()
        for i in range(len(masked)):
            if random.random() < mask_prob:
                masked[i] = mask_token
        return masked
    
    @staticmethod
    def random_insert(tokens: List[int], vocab_size: int, insert_prob: float = 0.1) -> List[int]:
        """Random token insertion."""
        result = []
        for token in tokens:
            if random.random() < insert_prob:
                result.append(random.randint(0, vocab_size - 1))
            result.append(token)
        return result
    
    @staticmethod
    def random_delete(tokens: List[int], delete_prob: float = 0.1) -> List[int]:
        """Random token deletion."""
        return [t for t in tokens if random.random() > delete_prob]
    
    @staticmethod
    def random_swap(tokens: List[int], swap_prob: float = 0.1) -> List[int]:
        """Random token swapping."""
        result = tokens.copy()
        for i in range(len(result) - 1):
            if random.random() < swap_prob:
                result[i], result[i+1] = result[i+1], result[i]
        return result


class DataValidator:
    """Validate data quality."""
    
    def __init__(self):
        self.issues: List[Dict] = []
    
    def validate(self, text: str) -> bool:
        """Validate a text sample."""
        issues = []
        
        # Check empty
        if not text or not text.strip():
            issues.append("empty_text")
        
        # Check length
        if len(text) > 100000:
            issues.append("too_long")
        
        # Check encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("encoding_error")
        
        if issues:
            self.issues.append({
                'text_preview': text[:50],
                'issues': issues
            })
            return False
        
        return True
    
    def get_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            'total_issues': len(self.issues),
            'issue_types': self._count_issues(),
            'samples': self.issues[:10]  # First 10 issues
        }
    
    def _count_issues(self) -> Dict[str, int]:
        """Count issue types."""
        counts = {}
        for issue in self.issues:
            for itype in issue['issues']:
                counts[itype] = counts.get(itype, 0) + 1
        return counts


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = [
        "Hello world this is a test",
        "Another example sentence",
        "Machine learning is fascinating",
    ] * 10
    
    # Create tokenizer
    from data import CharTokenizer
    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")
    
    # Create data loader
    config = DataConfig(batch_size=4, seq_length=20)
    loader = DataLoader(data, config, tokenizer)
    
    print("DataLoader test:")
    for i, batch in enumerate(loader):
        print(f"Batch {i}: {len(batch['input'])} samples")
        if i >= 2:
            break
    
    # Test streaming
    print("\nStreaming test:")
    # Create temp file
    with open("test_data.txt", 'w') as f:
        for item in data:
            f.write(item + "\n")
    
    stream = StreamingDataset("test_data.txt", tokenizer, max_seq_length=10)
    for i, tokens in enumerate(stream):
        print(f"Stream {i}: {len(tokens)} tokens")
        if i >= 5:
            break
    
    # Cleanup
    import os
    os.remove("test_data.txt")
    
    # Test augmentation
    print("\nAugmentation test:")
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Original: {tokens}")
    print(f"Masked: {DataAugmenter.random_mask(tokens, mask_token=0)}")
    print(f"Swapped: {DataAugmenter.random_swap(tokens)}")
