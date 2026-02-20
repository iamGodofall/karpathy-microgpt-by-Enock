"""
Model serving system for microgpt.
High-performance model serving with batching and caching.
"""

import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque


@dataclass
class InferenceRequest:
    """Inference request."""

    id: str
    inputs: Any
    callback: Callable
    timestamp: float
    priority: int = 0


class Batcher:
    """Dynamic batching for inference."""

    def __init__(self, max_batch_size: int = 8, max_latency_ms: float = 10):
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.queue = queue.PriorityQueue()
        self._lock = threading.Lock()

    def add_request(self, request: InferenceRequest):
        """Add request to batch."""
        self.queue.put((request.priority, time.time(), request))

    def get_batch(self) -> List[InferenceRequest]:
        """Get a batch of requests."""
        batch = []
        deadline = time.time() + self.max_latency_ms / 1000

        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                _, _, request = self.queue.get(timeout=0.001)
                batch.append(request)
            except queue.Empty:
                break

        return batch


class ModelServer:
    """High-performance model server."""

    def __init__(self, model, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        self.batcher = Batcher(
            max_batch_size=self.config.get("max_batch_size", 8),
            max_latency_ms=self.config.get("max_latency_ms", 10),
        )
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._running = False
        self._worker_thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start the server."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._inference_loop)
        self._worker_thread.start()
        print("Model server started")

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        print("Model server stopped")

    def _inference_loop(self):
        """Main inference loop."""
        while self._running:
            batch = self.batcher.get_batch()
            if batch:
                self._process_batch(batch)
            else:
                time.sleep(0.001)

    def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of requests."""
        # Check cache first
        cached_results = {}
        uncached_requests = []

        for req in requests:
            cache_key = str(req.inputs)
            if cache_key in self.cache:
                cached_results[req.id] = self.cache[cache_key]
                self.cache_hits += 1
            else:
                uncached_requests.append(req)
                self.cache_misses += 1

        # Run inference for uncached
        if uncached_requests:
            inputs = [r.inputs for r in uncached_requests]
            outputs = self._batch_inference(inputs)

            for req, output in zip(uncached_requests, outputs):
                cache_key = str(req.inputs)
                self.cache[cache_key] = output
                cached_results[req.id] = output

        # Send results
        for req in requests:
            if req.id in cached_results:
                req.callback(cached_results[req.id])

    def _batch_inference(self, inputs: List[Any]) -> List[Any]:
        """Run batch inference."""
        results = []
        for inp in inputs:
            if hasattr(self.model, "forward"):
                result = self.model.forward(inp)
            elif hasattr(self.model, "generate"):
                result = self.model.generate(inp)
            else:
                result = None
            results.append(result)
        return results

    def predict(self, inputs: Any, timeout: float = 30) -> Any:
        """Synchronous prediction."""
        result_container = {}

        def callback(output):
            result_container["result"] = output

        request = InferenceRequest(
            id=str(time.time()), inputs=inputs, callback=callback, timestamp=time.time()
        )

        self.batcher.add_request(request)

        # Wait for result
        start = time.time()
        while "result" not in result_container and time.time() - start < timeout:
            time.sleep(0.001)

        return result_container.get("result")

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "queue_size": self.batcher.queue.qsize(),
        }


class ModelEnsemble:
    """Ensemble of models for improved predictions."""

    def __init__(self, models: List[Any], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, inputs: Any) -> Dict[str, Any]:
        """Get ensemble prediction."""
        predictions = []
        for model in self.models:
            if hasattr(model, "forward"):
                pred = model.forward(inputs)
            elif hasattr(model, "generate"):
                pred = model.generate(inputs)
            else:
                pred = None
            predictions.append(pred)

        # Weighted average (simplified)
        return {
            "predictions": predictions,
            "ensemble": self._aggregate(predictions),
            "uncertainty": self._uncertainty(predictions),
        }

    def _aggregate(self, predictions: List[Any]) -> Any:
        """Aggregate predictions."""
        # Simple averaging for logits
        if predictions and hasattr(predictions[0], "__len__"):
            result = [0.0] * len(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                for i, v in enumerate(pred):
                    result[i] += v.data * weight if hasattr(v, "data") else v * weight
            return result
        return predictions[0] if predictions else None

    def _uncertainty(self, predictions: List[Any]) -> float:
        """Calculate prediction uncertainty."""
        if not predictions or len(predictions) < 2:
            return 0.0

        # Variance as uncertainty measure
        if hasattr(predictions[0], "__len__"):
            variances = []
            for i in range(len(predictions[0])):
                values = [p[i].data if hasattr(p[i], "data") else p[i] for p in predictions]
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / len(values)
                variances.append(var)
            return sum(variances) / len(variances)

        return 0.0


# Example usage
if __name__ == "__main__":
    from model import GPT, GPTConfig

    # Create model
    config = GPTConfig(vocab_size=100, n_embd=32, n_layer=1)
    model = GPT(config)

    # Create server
    server = ModelServer(model, {"max_batch_size": 4})
    server.start()

    # Test predictions
    for i in range(5):
        result = server.predict([1, 2, 3])
        print(f"Prediction {i+1}: {result is not None}")

    # Stats
    print(f"\nServer stats: {server.get_stats()}")

    server.stop()

    # Test ensemble
    print("\nEnsemble test:")
    models = [GPT(config) for _ in range(3)]
    ensemble = ModelEnsemble(models, [0.5, 0.3, 0.2])
    result = ensemble.predict([1, 2, 3])
    print(f"Ensemble prediction: {result['ensemble'] is not None}")
    print(f"Uncertainty: {result['uncertainty']:.4f}")
