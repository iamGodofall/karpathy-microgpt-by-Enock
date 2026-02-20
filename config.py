"""
Configuration management for microgpt.
Supports YAML and JSON config files with sensible defaults.
"""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_layer: int = 1
    n_embd: int = 16
    n_head: int = 4
    block_size: int = 16
    dropout: float = 0.0
    use_gelu: bool = False  # False = ReLU, True = GELU
    use_layernorm: bool = False  # False = RMSNorm, True = LayerNorm


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_steps: int = 1000
    batch_size: int = 1
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps_adam: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 0.0  # 0 = no clipping
    lr_schedule: str = "linear"  # linear, cosine, constant
    warmup_steps: int = 0
    val_split: float = 0.1
    eval_interval: int = 100
    save_interval: int = 500


@dataclass
class GenerationConfig:
    """Text generation parameters."""
    temperature: float = 0.5
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0  # 1.0 = disabled (nucleus sampling)
    max_length: int = 16
    num_samples: int = 20
    seed: int = 42


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = None
    training: TrainingConfig = None
    generation: GenerationConfig = None
    data_path: str = "input.txt"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def to_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# Default configuration
DEFAULT_CONFIG = Config()
