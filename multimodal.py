"""
Multi-modal capabilities for microgpt.
Vision, audio, and multi-modal understanding.
"""

import random
import math
from typing import List, Tuple, Optional, Dict
from model import Value, GPT


class VisionEncoder:
    """
    Simple vision encoder for image understanding.
    Patch-based encoding similar to ViT.
    """
    
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 n_embd: int = 768, n_layer: int = 12):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.n_embd = n_embd
        self.n_layer = n_layer
        
        # Patch embedding
        self.patch_embed = [[random.gauss(0, 0.02) 
                            for _ in range(patch_size * patch_size * 3)] 
                           for _ in range(n_embd)]
        
        # Position embeddings
        self.pos_embed = [[random.gauss(0, 0.02) 
                          for _ in range(n_embd)] 
                         for _ in range(self.n_patches + 1)]  # +1 for CLS token
        
        # Transformer layers (simplified)
        self.layers = []
        for _ in range(n_layer):
            # Attention weights
            qkv = [[random.gauss(0, 0.02) for _ in range(n_embd)] 
                   for _ in range(n_embd * 3)]
            proj = [[random.gauss(0, 0.02) for _ in range(n_embd)] 
                    for _ in range(n_embd)]
            self.layers.append((qkv, proj))
    
    def patchify(self, image: List[List[List[float]]]) -> List[List[float]]:
        """Convert image to patches."""
        # image: [H][W][C]
        patches = []
        for i in range(0, self.image_size, self.patch_size):
            for j in range(0, self.image_size, self.patch_size):
                patch = []
                for pi in range(self.patch_size):
                    for pj in range(self.patch_size):
                        for c in range(3):
                            if i + pi < len(image) and j + pj < len(image[0]):
                                patch.append(image[i + pi][j + pj][c])
                            else:
                                patch.append(0.0)
                patches.append(patch)
        
        return patches
    
    def encode(self, image: List[List[List[float]]]) -> List[Value]:
        """Encode image to embeddings."""
        # Patchify
        patches = self.patchify(image)
        
        # Embed patches
        embeddings = []
        for patch in patches:
            emb = [sum(w[i] * patch[i] for i in range(len(patch))) 
                   for w in self.patch_embed]
            embeddings.append(emb)
        
        # Add CLS token
        cls_token = [0.0] * self.n_embd
        embeddings.insert(0, cls_token)
        
        # Add position embeddings
        for i, emb in enumerate(embeddings):
            for j in range(self.n_embd):
                emb[j] += self.pos_embed[i][j]
        
        # Convert to Values
        value_embeddings = [[Value(v) for v in emb] for emb in embeddings]
        
        # Transformer forward (simplified)
        for qkv, proj in self.layers:
            # Self-attention
            new_embeddings = []
            for i, x in enumerate(value_embeddings):
                # Compute Q, K, V
                qkv_out = []
                for w in qkv:
                    val = sum(w[j] * x[j].data for j in range(len(x)))
                    qkv_out.append(val)
                
                # Simplified attention
                attn_out = [Value(v) for v in qkv_out[:self.n_embd]]
                new_embeddings.append(attn_out)
            
            value_embeddings = new_embeddings
        
        # Return CLS token representation
        return value_embeddings[0]


class AudioEncoder:
    """
    Audio encoder for speech and sound understanding.
    Uses spectrogram-based encoding.
    """
    
    def __init__(self, n_mels: int = 80, n_fft: int = 400,
                 n_embd: int = 768, n_layer: int = 12):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_embd = n_embd
        self.n_layer = n_layer
        
        # Spectrogram embedding
        self.spec_embed = [[random.gauss(0, 0.02) 
                          for _ in range(n_mels)] 
                         for _ in range(n_embd)]
        
        # Positional encoding for time
        self.time_embed = [[random.gauss(0, 0.02) 
                           for _ in range(n_embd)] 
                          for _ in range(1000)]  # Max 1000 time steps
    
    def spectrogram(self, audio: List[float]) -> List[List[float]]:
        """Compute mel spectrogram (simplified)."""
        # Simplified - real implementation uses STFT
        # Return random features for demonstration
        n_frames = len(audio) // self.n_fft
        return [[random.random() for _ in range(self.n_mels)] 
                for _ in range(min(n_frames, 1000))]
    
    def encode(self, audio: List[float]) -> List[Value]:
        """Encode audio to embeddings."""
        # Compute spectrogram
        spec = self.spectrogram(audio)
        
        # Embed
        embeddings = []
        for frame in spec:
            emb = [sum(w[i] * frame[i] for i in range(len(frame))) 
                   for w in self.spec_embed]
            embeddings.append(emb)
        
        # Add time embeddings
        for i, emb in enumerate(embeddings):
            for j in range(self.n_embd):
                emb[j] += self.time_embed[i][j]
        
        # Convert to Values
        return [Value(sum(e) / len(e)) for e in embeddings]


class MultiModalGPT:
    """
    GPT with multi-modal understanding.
    Combines text, vision, and audio.
    """
    
    def __init__(self, text_model: GPT, vision_encoder: Optional[VisionEncoder] = None,
                 audio_encoder: Optional[AudioEncoder] = None):
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        
        # Projection layers to align modalities
        self.vision_proj = None
        self.audio_proj = None
        
        if vision_encoder:
            self.vision_proj = [[random.gauss(0, 0.02) 
                               for _ in range(vision_encoder.n_embd)] 
                              for _ in range(text_model.n_embd)]
        
        if audio_encoder:
            self.audio_proj = [[random.gauss(0, 0.02) 
                               for _ in range(audio_encoder.n_embd)] 
                              for _ in range(text_model.n_embd)]
        
        # Special tokens
        self.image_token = text_model.vocab_size  # New token for image
        self.audio_token = text_model.vocab_size + 1  # New token for audio
    
    def encode_image(self, image: List[List[List[float]]]) -> List[Value]:
        """Encode and project image."""
        if not self.vision_encoder:
            raise ValueError("Vision encoder not initialized")
        
        vision_features = self.vision_encoder.encode(image)
        
        # Project to text space
        projected = [sum(self.vision_proj[i][j] * vision_features[j].data 
                        for j in range(len(vision_features))) 
                    for i in range(len(self.vision_proj))]
        
        return [Value(v) for v in projected]
    
    def encode_audio(self, audio: List[float]) -> List[Value]:
        """Encode and project audio."""
        if not self.audio_encoder:
            raise ValueError("Audio encoder not initialized")
        
        audio_features = self.audio_encoder.encode(audio)
        
        # Project to text space
        projected = [sum(self.audio_proj[i][j] * audio_features[j].data 
                        for j in range(len(audio_features))) 
                    for i in range(len(self.audio_proj))]
        
        return [Value(v) for v in projected]
    
    def generate_multimodal(self, 
                          text_prompt: Optional[str] = None,
                          image: Optional[List[List[List[float]]]] = None,
                          audio: Optional[List[float]] = None,
                          max_length: int = 50) -> str:
        """
        Generate text conditioned on multiple modalities.
        """
        # Encode modalities
        context = []
        
        if image:
            img_emb = self.encode_image(image)
            context.extend(img_emb)
        
        if audio:
            aud_emb = self.encode_audio(audio)
            context.extend(aud_emb)
        
        # Add text prompt if provided
        if text_prompt:
            # Simplified - would tokenize and process
            pass
        
        # Generate (simplified)
        return "Multi-modal response..."


class ToolUse:
    """
    Tool use / function calling capabilities.
    Allows model to use external tools.
    """
    
    def __init__(self, model: GPT):
        self.model = model
        
        # Available tools
        self.tools = {
            'calculator': self._tool_calculator,
            'search': self._tool_search,
            'weather': self._tool_weather,
            'datetime': self._tool_datetime,
        }
        
        # Tool descriptions for prompting
        self.tool_descriptions = {
            'calculator': "Evaluate mathematical expressions",
            'search': "Search for information on the internet",
            'weather': "Get current weather for a location",
            'datetime': "Get current date and time",
        }
    
    def _tool_calculator(self, expression: str) -> str:
        """Calculator tool."""
        try:
            # Safe evaluation (simplified)
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except:
            return "Error: Invalid expression"
    
    def _tool_search(self, query: str) -> str:
        """Search tool (mock)."""
        return f"Search results for: {query}"
    
    def _tool_weather(self, location: str) -> str:
        """Weather tool (mock)."""
        return f"Weather in {location}: 72°F, sunny"
    
    def _tool_datetime(self, _: str) -> str:
        """DateTime tool."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def parse_tool_call(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Parse tool call from model output.
        Format: <tool>tool_name</tool><args>arguments</args>
        """
        import re
        
        tool_match = re.search(r'<tool>(.*?)</tool>', text)
        args_match = re.search(r'<args>(.*?)</args>', text)
        
        if tool_match and args_match:
            return tool_match.group(1), args_match.group(1)
        
        return None
    
    def execute_tool(self, tool_name: str, args: str) -> str:
        """Execute a tool."""
        if tool_name in self.tools:
            return self.tools[tool_name](args)
        return f"Unknown tool: {tool_name}"
    
    def generate_with_tools(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate with potential tool use.
        """
        # Generate initial response
        # Simplified - would use actual model
        response = f"To answer this, I'll use a tool. <tool>calculator</tool><args>2+2</args>"
        
        # Check for tool calls
        tool_call = self.parse_tool_call(response)
        
        if tool_call:
            tool_name, args = tool_call
            result = self.execute_tool(tool_name, args)
            
            # Continue generation with tool result
            final_response = f"{response}\nTool result: {result}\n"
            return final_response
        
        return response


class RAG:
    """
    Retrieval-Augmented Generation.
    Enhance generation with external knowledge.
    """
    
    def __init__(self, model: GPT, knowledge_base: List[str] = None):
        self.model = model
        self.knowledge_base = knowledge_base or []
        
        # Simple embedding for retrieval
        self.embeddings = []
        self._index_knowledge()
    
    def _index_knowledge(self):
        """Index knowledge base for retrieval."""
        # Simplified - real implementation uses vector DB
        for doc in self.knowledge_base:
            # Create simple hash-based embedding
            embedding = [ord(c) % 256 / 256.0 for c in doc[:100]]
            self.embeddings.append((doc, embedding))
    
    def _similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents."""
        # Create query embedding
        query_emb = [ord(c) % 256 / 256.0 for c in query[:100]]
        
        # Find most similar
        similarities = [
            (doc, self._similarity(query_emb, emb))
            for doc, emb in self.embeddings
        ]
        
        similarities.sort(key=lambda x: -x[1])
        return [doc for doc, _ in similarities[:k]]
    
    def generate_with_rag(self, query: str, max_length: int = 100) -> str:
        """
        Generate with retrieved context.
        """
        # Retrieve relevant documents
        context_docs = self.retrieve(query, k=3)
        
        # Build prompt with context
        context_str = "\n".join(f"Document: {doc}" for doc in context_docs)
        prompt = f"""Context:
{context_str}

Question: {query}
Answer:"""
        
        # Generate (simplified)
        return f"Based on the context: {context_docs[0] if context_docs else 'No relevant info'}"


class MixtureOfDepths:
    """
    Mixture of Depths from DeepMind.
    Dynamically allocate compute based on token importance.
    """
    
    def __init__(self, model: GPT, capacity_factor: float = 0.5):
        self.model = model
        self.capacity_factor = capacity_factor  # Fraction of tokens to process deeply
        
        # Router for depth selection
        self.router = [[random.gauss(0, 0.02) 
                        for _ in range(model.n_embd)] 
                       for _ in range(1)]
    
    def route_tokens(self, tokens: List[List[Value]]) -> Tuple[List[int], List[int]]:
        """
        Decide which tokens get deep vs shallow processing.
        """
        # Compute importance scores
        scores = []
        for token_emb in tokens:
            score = sum(self.router[0][i] * token_emb[i].data 
                       for i in range(len(token_emb)))
            scores.append(score)
        
        # Select top-k for deep processing
        k = int(len(tokens) * self.capacity_factor)
        deep_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        deep_set = set(deep_indices)
        
        shallow_indices = [i for i in range(len(tokens)) if i not in deep_set]
        
        return deep_indices, shallow_indices
    
    def forward(self, tokens: List[List[Value]], 
                keys: List[List], values: List[List]) -> List[List[Value]]:
        """
        Forward pass with dynamic depth.
        """
        deep_indices, shallow_indices = self.route_tokens(tokens)
        
        output = [None] * len(tokens)
        
        # Deep processing for important tokens
        for i in deep_indices:
            # Full transformer layer
            output[i] = self._deep_forward(tokens[i], keys, values, i)
        
        # Shallow processing for others
        for i in shallow_indices:
            # Simplified processing (e.g., just MLP)
            output[i] = self._shallow_forward(tokens[i])
        
        return output
    
    def _deep_forward(self, token: List[Value], keys, values, pos):
        """Full transformer processing."""
        # Use base model's forward
        return self.model.forward(0, pos, keys, values)  # Simplified
    
    def _shallow_forward(self, token: List[Value]):
        """Simplified processing."""
        # Just pass through with minimal computation
        return token


def create_multimodal_model(text_model: GPT) -> MultiModalGPT:
    """Create a multi-modal model."""
    vision = VisionEncoder()
    audio = AudioEncoder()
    return MultiModalGPT(text_model, vision, audio)
