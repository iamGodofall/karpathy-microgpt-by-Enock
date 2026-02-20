"""
Feature store for microgpt.
Manage embeddings, features, and vector storage.
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle


@dataclass
class FeatureVector:
    """A feature vector with metadata."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    created_at: float
    version: str = "1.0"


class FeatureStore:
    """Store and manage feature vectors."""

    def __init__(self, store_dir: str = "feature_store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        self.index: Dict[str, FeatureVector] = {}
        self.versions: Dict[str, str] = {}
        self._load_index()

    def _load_index(self):
        """Load feature index."""
        index_file = self.store_dir / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)
                for k, v in data.items():
                    self.index[k] = FeatureVector(**v)

    def _save_index(self):
        """Save feature index."""
        index_file = self.store_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump({k: asdict(v) for k, v in self.index.items()}, f, default=str)

    def add_feature(
        self,
        feature_id: str,
        vector: List[float],
        metadata: Dict[str, Any] = None,
        version: str = "1.0",
    ):
        """Add a feature vector."""
        import time

        feature = FeatureVector(
            id=feature_id,
            vector=vector,
            metadata=metadata or {},
            created_at=time.time(),
            version=version,
        )
        self.index[feature_id] = feature
        self._save_index()

    def get_feature(self, feature_id: str) -> Optional[FeatureVector]:
        """Get a feature by ID."""
        return self.index.get(feature_id)

    def search_similar(self, query: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar features using cosine similarity."""
        if not self.index:
            return []

        similarities = []
        for feature_id, feature in self.index.items():
            sim = self._cosine_similarity(query, feature.vector)
            similarities.append((feature_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    def get_features_by_metadata(self, key: str, value: Any) -> List[FeatureVector]:
        """Search features by metadata."""
        return [f for f in self.index.values() if f.metadata.get(key) == value]

    def delete_feature(self, feature_id: str):
        """Delete a feature."""
        if feature_id in self.index:
            del self.index[feature_id]
            self._save_index()

    def list_features(self) -> List[str]:
        """List all feature IDs."""
        return list(self.index.keys())

    def export_features(self, path: str):
        """Export features to file."""
        with open(path, "wb") as f:
            pickle.dump(self.index, f)
        print(f"Exported {len(self.index)} features to {path}")

    def import_features(self, path: str):
        """Import features from file."""
        with open(path, "rb") as f:
            self.index = pickle.load(f)
        self._save_index()
        print(f"Imported {len(self.index)} features from {path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        if not self.index:
            return {"count": 0}

        vector_lengths = [len(f.vector) for f in self.index.values()]

        return {
            "count": len(self.index),
            "vector_dim": vector_lengths[0] if vector_lengths else 0,
            "avg_vector_length": sum(vector_lengths) / len(vector_lengths),
            "versions": list(set(f.version for f in self.index.values())),
        }


class EmbeddingCache:
    """Cache for text embeddings."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, List[float]] = {}
        self.access_count: Dict[str, int] = {}
        self.max_size = max_size

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        if text in self.cache:
            self.access_count[text] = self.access_count.get(text, 0) + 1
            return self.cache[text]
        return None

    def put(self, text: str, embedding: List[float]):
        """Store embedding in cache."""
        if len(self.cache) >= self.max_size:
            # LRU eviction
            lru = min(self.access_count, key=self.access_count.get)
            del self.cache[lru]
            del self.access_count[lru]

        self.cache[text] = embedding
        self.access_count[text] = 1

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()


class FeaturePipeline:
    """Pipeline for feature extraction and transformation."""

    def __init__(self):
        self.transformations: List[callable] = []

    def add_transform(self, transform: callable):
        """Add a transformation."""
        self.transformations.append(transform)

    def process(self, features: List[float]) -> List[float]:
        """Apply all transformations."""
        result = features
        for transform in self.transformations:
            result = transform(result)
        return result

    @staticmethod
    def normalize(features: List[float]) -> List[float]:
        """L2 normalization."""
        norm = sum(x * x for x in features) ** 0.5
        if norm == 0:
            return features
        return [x / norm for x in features]

    @staticmethod
    def center(features: List[float]) -> List[float]:
        """Center features (zero mean)."""
        mean = sum(features) / len(features)
        return [x - mean for x in features]

    @staticmethod
    def scale(factor: float) -> callable:
        """Create scaling transform."""

        def transform(features: List[float]) -> List[float]:
            return [x * factor for x in features]

        return transform


# Example usage
if __name__ == "__main__":
    store = FeatureStore()

    # Add some features
    for i in range(10):
        store.add_feature(
            f"doc_{i}",
            vector=[random.random() for _ in range(128)],
            metadata={"category": "text", "id": i},
        )

    # Search similar
    query = [random.random() for _ in range(128)]
    similar = store.search_similar(query, top_k=3)
    print("Top 3 similar features:")
    for fid, sim in similar:
        print(f"  {fid}: {sim:.4f}")

    # Get statistics
    stats = store.get_statistics()
    print(f"\nFeature store: {stats['count']} features, dim={stats['vector_dim']}")

    # Test cache
    cache = EmbeddingCache()
    cache.put("hello", [0.1, 0.2, 0.3])
    cached = cache.get("hello")
    print(f"\nCached embedding: {cached}")
