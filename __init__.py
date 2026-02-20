"""
microgpt - A comprehensive, pure-Python GPT ecosystem.

This package provides a complete implementation of GPT (Generative Pre-trained Transformer)
with state-of-the-art features including:

- Core: Pure Python transformer with autograd
- Training: Adam, advanced optimizers, LR scheduling
- Architecture: Multi-layer, multi-head, modern techniques (RoPE, SwiGLU, etc.)
- Efficiency: Quantization, LoRA, gradient checkpointing
- Inference: PagedAttention, speculative decoding, continuous batching
- Safety: RLHF, DPO, Constitutional AI
- Multimodal: Vision, audio, tool use
- Merging: TIES, DARE, Model Soups
- Reasoning: Chain-of-Thought, Tree-of-Thought, ReAct
- Agents: Multi-agent systems, planning

Example:
    >>> from microgpt import GPT, Trainer, CharTokenizer
    >>> model = GPT(vocab_size=256, block_size=16, n_layer=2, n_embd=32)
    >>> trainer = Trainer(model)
    >>> # Train and generate...
"""

__version__ = "2.0.0"
__author__ = "microgpt Team"
__license__ = "MIT"

# Core components
from .model import GPT, Value
from .trainer import Trainer, AdamOptimizer, LRScheduler
from .data import CharTokenizer, load_data
from .config import Config, ModelConfig, TrainingConfig, GenerationConfig
from .checkpoint import CheckpointManager
from .logger import TrainingLogger, Metrics

# Advanced features
from .advanced_features import (
    BeamSearch, ContrastiveSearch, RepetitionPenalty,
    MirostatSampler, TypicalSampler
)
from .quantization import QuantizedGPT, quantize_model
from .export import export_to_torch, export_to_numpy, export_to_onnx
from .distributed import DataParallel, GradientCheckpointing

# Modern architecture
from .modern_architecture import (
    ModernGPT, RoPE, SwiGLU, ALiBi, 
    FlashAttention, GroupedQueryAttention
)

# Advanced training
from .advanced_training import (
    LionOptimizer, SophiaOptimizer, MuonOptimizer,
    ScheduleFreeOptimizer, ChinchillaScaling,
    CurriculumLearning, MixtureOfExperts
)

# Safety and alignment
from .safety_alignment import (
    RLHFTrainer, DPOTrainer, ConstitutionalAI,
    SafetyClassifier, Watermarking
)

# Multimodal
from .multimodal import (
    VisionEncoder, AudioEncoder, MultiModalGPT,
    ToolUse, RAG
)

# Inference optimizations
from .inference_optimizations import (
    PagedAttention, ContinuousBatching,
    SpeculativeDecodingEngine, StreamingLLM
)

# Memory efficiency
from .memory_efficient import (
    LoRA, QLoRA, DoRA, ReLoRA, GaLore,
    GradientCheckpointing
)

# State of the art
from .state_of_the_art import (
    MambaBlock, GriffinBlock, JambaArchitecture,
    DiffTransformer, TitansArchitecture
)

# Model merging
from .model_merging import (
    TIESMerging, DARE, ModelSoups,
    TaskArithmetic, SLERP
)

# Evaluation
from .evaluation import (
    PerplexityMetrics, BLEU, ROUGE,
    DiversityMetrics, ComprehensiveEvaluator
)

# Reasoning
from .reasoning import (
    ChainOfThought, TreeOfThought, ReAct,
    Reflexion, SelfConsistency
)

# Agents
from .agents import Agent, MultiAgentSystem, Planner

# Profiling and compression
from .profiling import Profiler, ModelAnalyzer, SpeedBenchmark
from .compression import (
    MagnitudePruning, KnowledgeDistillation,
    WeightSharing, QuantizationAwareTraining
)

# Model zoo
from .model_zoo import create_model, list_models

# Tokenizers
from .tokenizers import create_tokenizer, list_tokenizers


__all__ = [
    # Core
    'GPT', 'Value', 'Trainer', 'AdamOptimizer', 'LRScheduler',
    'CharTokenizer', 'load_data',
    'Config', 'ModelConfig', 'TrainingConfig', 'GenerationConfig',
    'CheckpointManager', 'TrainingLogger', 'Metrics',
    
    # Advanced features
    'BeamSearch', 'ContrastiveSearch', 'RepetitionPenalty',
    'QuantizedGPT', 'quantize_model',
    
    # Modern architecture
    'ModernGPT', 'RoPE', 'SwiGLU', 'ALiBi',
    
    # Training
    'LionOptimizer', 'SophiaOptimizer', 'MuonOptimizer',
    'CurriculumLearning', 'MixtureOfExperts',
    
    # Safety
    'RLHFTrainer', 'DPOTrainer', 'ConstitutionalAI',
    'SafetyClassifier', 'Watermarking',
    
    # Multimodal
    'VisionEncoder', 'AudioEncoder', 'MultiModalGPT',
    'ToolUse', 'RAG',
    
    # Inference
    'PagedAttention', 'ContinuousBatching',
    'SpeculativeDecodingEngine', 'StreamingLLM',
    
    # Memory
    'LoRA', 'QLoRA', 'DoRA', 'ReLoRA', 'GaLore',
    
    # SOTA
    'MambaBlock', 'GriffinBlock', 'JambaArchitecture',
    'DiffTransformer', 'TitansArchitecture',
    
    # Merging
    'TIESMerging', 'DARE', 'ModelSoups',
    'TaskArithmetic', 'SLERP',
    
    # Evaluation
    'PerplexityMetrics', 'BLEU', 'ROUGE',
    'ComprehensiveEvaluator',
    
    # Reasoning
    'ChainOfThought', 'TreeOfThought', 'ReAct',
    'Reflexion', 'SelfConsistency',
    
    # Agents
    'Agent', 'MultiAgentSystem', 'Planner',
    
    # Utils
    'Profiler', 'ModelAnalyzer', 'SpeedBenchmark',
    'MagnitudePruning', 'KnowledgeDistillation',
    'create_model', 'list_models',
    'create_tokenizer', 'list_tokenizers',
]
