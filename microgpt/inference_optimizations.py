"""
Inference optimizations from vLLM, TensorRT-LLM, and other SOTA systems.
Includes PagedAttention, continuous batching, and more.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
from .model import Value, GPT


class PagedAttention:
    """
    PagedAttention from vLLM.
    Reduces memory waste with dynamic key-value cache management.
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 1000):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Free block pool
        self.free_blocks = set(range(num_blocks))

        # Block table: sequence_id -> list of block indices
        self.block_tables: Dict[int, List[int]] = {}

        # KV cache stored per block
        self.kv_cache: Dict[int, Tuple[List[Value], List[Value]]] = {}

    def allocate(self, sequence_id: int, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError("Out of memory: not enough free blocks")

        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block = self.free_blocks.pop()
            allocated.append(block)

        self.block_tables[sequence_id] = allocated
        return allocated

    def free(self, sequence_id: int):
        """Free blocks for a sequence."""
        if sequence_id in self.block_tables:
            for block in self.block_tables[sequence_id]:
                self.free_blocks.add(block)
                if block in self.kv_cache:
                    del self.kv_cache[block]
            del self.block_tables[sequence_id]

    def get_kv_cache(
        self, sequence_id: int, position: int
    ) -> Optional[Tuple[List[Value], List[Value]]]:
        """Get KV cache for a position."""
        if sequence_id not in self.block_tables:
            return None

        block_idx = position // self.block_size
        if block_idx >= len(self.block_tables[sequence_id]):
            return None

        block = self.block_tables[sequence_id][block_idx]
        return self.kv_cache.get(block)

    def set_kv_cache(self, sequence_id: int, position: int, k: List[Value], v: List[Value]):
        """Set KV cache for a position."""
        if sequence_id not in self.block_tables:
            return

        block_idx = position // self.block_size
        if block_idx >= len(self.block_tables[sequence_id]):
            # Need to allocate more blocks
            new_blocks = []
            for _ in range(block_idx - len(self.block_tables[sequence_id]) + 1):
                if self.free_blocks:
                    new_blocks.append(self.free_blocks.pop())

            self.block_tables[sequence_id].extend(new_blocks)

        block = self.block_tables[sequence_id][block_idx]
        self.kv_cache[block] = (k, v)


class ContinuousBatching:
    """
    Continuous batching for high-throughput serving.
    Process multiple requests together with dynamic batching.
    """

    def __init__(self, model: GPT, max_batch_size: int = 32, max_waiting_time: float = 0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_waiting_time = max_waiting_time

        # Request queue
        self.queue: deque = deque()

        # Active sequences
        self.active_sequences: Dict[int, dict] = {}

        # PagedAttention manager
        self.paged_attn = PagedAttention()

        self.request_counter = 0

    def add_request(
        self, prompt_tokens: List[int], max_length: int, temperature: float = 0.7
    ) -> int:
        """Add a new generation request."""
        request_id = self.request_counter
        self.request_counter += 1

        request = {
            "id": request_id,
            "prompt": prompt_tokens,
            "max_length": max_length,
            "temperature": temperature,
            "generated": list(prompt_tokens),
            "status": "waiting",
        }

        self.queue.append(request)
        return request_id

    def step(self) -> List[Tuple[int, int, Optional[int]]]:
        """
        Process one step of continuous batching.
        Returns list of (request_id, is_finished, next_token).
        """
        # Add waiting requests to active batch
        while (
            len(self.active_sequences) < self.max_batch_size
            and self.queue
            and len(self.queue) <= self.max_batch_size - len(self.active_sequences)
        ):
            request = self.queue.popleft()

            # Allocate blocks
            try:
                self.paged_attn.allocate(request["id"], len(request["prompt"]))
                self.active_sequences[request["id"]] = request
                request["status"] = "active"
            except MemoryError:
                # Put back in queue
                self.queue.appendleft(request)
                break

        # Process active sequences
        results = []
        finished = []

        for req_id, request in self.active_sequences.items():
            # Get current position
            pos = len(request["generated"]) - 1

            # Get KV cache
            kv_cache = self.paged_attn.get_kv_cache(req_id, pos)
            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]

            if kv_cache:
                # Use cached values
                pass

            # Forward pass
            token_id = request["generated"][-1]
            logits = self.model.forward(token_id, pos, keys, values)

            # Sample next token
            probs = self._softmax([logit.data / request["temperature"] for logit in logits])

            next_token = random.choices(range(len(probs)), weights=probs)[0]

            # Update KV cache
            # Simplified - would extract actual K, V from model

            request["generated"].append(next_token)

            # Check if finished
            is_finished = (
                len(request["generated"]) >= request["max_length"] or next_token == 0
            )  # EOS token

            if is_finished:
                finished.append(req_id)

            results.append((req_id, is_finished, next_token))

        # Clean up finished sequences
        for req_id in finished:
            self.paged_attn.free(req_id)
            del self.active_sequences[req_id]

        return results

    def _softmax(self, logits: List[float]) -> List[float]:
        """Numerically stable softmax."""
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for logit in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def run(self) -> Dict[int, List[int]]:
        """Run until all requests complete."""
        results = {}

        while self.queue or self.active_sequences:
            step_results = self.step()
            for req_id, is_finished, _ in step_results:
                if is_finished and req_id not in results:
                    results[req_id] = self.active_sequences.get(req_id, {}).get("generated", [])

        return results


class SpeculativeDecodingEngine:
    """
    Advanced speculative decoding with tree-based verification.
    """

    def __init__(self, draft_model: GPT, target_model: GPT, num_draft_tokens: int = 4):
        self.draft_model = draft_model
        self.target_model = target_model
        self.num_draft_tokens = num_draft_tokens

    def generate_tree(self, prompt: List[int]) -> List[List[int]]:
        """
        Generate a tree of draft tokens.
        """
        # Simple chain for now - could be tree-structured
        drafts = []
        current = list(prompt)

        for _ in range(self.num_draft_tokens):
            # Draft model generates next token
            keys = [[] for _ in range(self.draft_model.n_layer)]
            values = [[] for _ in range(self.draft_model.n_layer)]

            for i, tok in enumerate(current[:-1]):
                _ = self.draft_model.forward(tok, i, keys, values)

            logits = self.draft_model.forward(current[-1], len(current) - 1, keys, values)
            probs = self._softmax([logit.data for logit in logits])

            next_token = random.choices(range(len(probs)), weights=probs)[0]

            current.append(next_token)
            drafts.append(list(current))

        return drafts

    def verify_and_accept(
        self, prompt: List[int], drafts: List[List[int]]
    ) -> Tuple[List[int], int]:
        """
        Verify drafts with target model and accept valid prefix.
        """
        # Get target model probabilities for all draft positions
        keys = [[] for _ in range(self.target_model.n_layer)]
        values = [[] for _ in range(self.target_model.n_layer)]

        # Prime with prompt
        for i, tok in enumerate(prompt[:-1]):
            _ = self.target_model.forward(tok, i, keys, values)

        accepted = 0
        for i, draft in enumerate(drafts):
            pos = len(prompt) + i
            logits = self.target_model.forward(draft[-2], pos - 1, keys, values)

            # Check if draft token is acceptable
            target_probs = self._softmax([logit.data for logit in logits])

            draft_token = draft[-1]

            # Acceptance criterion
            if random.random() < target_probs[draft_token]:
                accepted += 1
            else:
                # Reject - resample from adjusted distribution
                adjusted_probs = [
                    max(0, tp - dp)
                    for tp, dp in zip(target_probs, self._get_draft_probs(drafts, i))
                ]
                total = sum(adjusted_probs)
                if total > 0:
                    adjusted_probs = [p / total for p in adjusted_probs]
                    new_token = random.choices(range(len(adjusted_probs)), weights=adjusted_probs)[
                        0
                    ]
                    drafts[i][-1] = new_token
                break

        return drafts[accepted - 1] if accepted > 0 else prompt, accepted

    def _softmax(self, logits: List[float]) -> List[float]:
        """Numerically stable softmax."""
        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]

        total = sum(exps)
        return [e / total for e in exps]

    def _get_draft_probs(self, drafts: List[List[int]], index: int) -> List[float]:
        """Get draft model probabilities."""
        # Simplified
        return [1.0 / self.draft_model.vocab_size] * self.draft_model.vocab_size

    def generate(self, prompt: List[int], max_length: int) -> List[int]:
        """Generate with speculative decoding."""
        result = list(prompt)

        while len(result) < max_length:
            # Generate drafts
            drafts = self.generate_tree(result)

            # Verify and accept
            accepted, num_accepted = self.verify_and_accept(result, drafts)
            result = accepted

            if num_accepted < len(drafts):
                # Some rejected, continue from there
                pass

            # Check for EOS
            if result[-1] == 0:
                break

        return result


class QuantizedCache:
    """
    KV cache quantization for memory efficiency.
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.cache = {}

    def quantize(self, tensor: List[Value]) -> Tuple[List[int], float, float]:
        """Quantize tensor to low precision."""
        # Find range
        values = [v.data for v in tensor]
        min_val = min(values)
        max_val = max(values)

        # Quantize
        scale = (max_val - min_val) / (2**self.bits - 1)
        zero_point = -min_val / scale if scale > 0 else 0

        quantized = [int((v - min_val) / scale) for v in values]

        return quantized, scale, zero_point

    def dequantize(self, quantized: List[int], scale: float, zero_point: float) -> List[Value]:
        """Dequantize to full precision."""
        return [Value((q - zero_point) * scale) for q in quantized]

    def store(self, key: str, k: List[Value], v: List[Value]):
        """Store quantized KV."""
        k_q, k_s, k_z = self.quantize(k)
        v_q, v_s, v_z = self.quantize(v)

        self.cache[key] = {"k": (k_q, k_s, k_z), "v": (v_q, v_s, v_z)}

    def load(self, key: str) -> Optional[Tuple[List[Value], List[Value]]]:
        """Load and dequantize KV."""
        if key not in self.cache:
            return None

        cached = self.cache[key]
        k = self.dequantize(*cached["k"])
        v = self.dequantize(*cached["v"])

        return k, v


class PrefixCaching:
    """
    Cache common prompt prefixes to avoid recomputation.
    """

    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache: Dict[Tuple, dict] = {}
        self.access_count: Dict[Tuple, int] = {}

    def get_cache_key(self, tokens: List[int]) -> Tuple:
        """Create cache key from tokens."""
        return tuple(tokens[-64:])  # Use last 64 tokens as key

    def lookup(self, tokens: List[int]) -> Optional[dict]:
        """Look up cached prefix."""
        key = self.get_cache_key(tokens)

        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]

        return None

    def store(self, tokens: List[int], kv_cache: dict):
        """Store prefix in cache."""
        key = self.get_cache_key(tokens)

        if len(self.cache) >= self.max_cache_size:
            # Evict least frequently used
            lfu_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lfu_key]
            del self.access_count[lfu_key]

        self.cache[key] = kv_cache
        self.access_count[key] = 1


class StreamingLLM:
    """
    StreamingLLM for infinite context length.
    Keep initial tokens (attention sinks) and recent tokens.
    """

    def __init__(self, model: GPT, sink_tokens: int = 4, recent_tokens: int = 1024):
        self.model = model
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens

        # Attention sink positions (always kept)
        self.sink_positions = list(range(sink_tokens))

        # Rolling buffer for recent tokens
        self.recent_buffer = []

    def process_token(self, token: int, position: int) -> List[Value]:
        """Process a single token with streaming attention."""
        # Determine which positions to attend to
        if position < self.sink_tokens:
            # Initial tokens - attend to all previous
            attend_positions = list(range(position + 1))
        else:
            # Attend to sinks + recent window
            attend_positions = self.sink_positions + list(
                range(max(0, position - self.recent_tokens), position)
            )

        # Build KV cache for attended positions
        # Simplified - would use actual cached values
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        # Forward pass
        logits = self.model.forward(token, position, keys, values)

        return logits

    def generate_streaming(self, prompt: List[int], max_new_tokens: int) -> List[int]:
        """Generate with streaming attention."""
        result = list(prompt)

        for i in range(max_new_tokens):
            pos = len(result)
            logits = self.process_token(result[-1], pos)

            # Sample
            probs = [logit.data for logit in logits]
            total = sum(probs)

            probs = [p / total for p in probs]
            next_token = random.choices(range(len(probs)), weights=probs)[0]

            result.append(next_token)

            if next_token == 0:  # EOS
                break

        return result


def create_optimized_inference_engine(model: GPT) -> ContinuousBatching:
    """Create optimized inference engine."""
    return ContinuousBatching(model)
