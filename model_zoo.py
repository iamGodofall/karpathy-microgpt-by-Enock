"""
Pre-trained model configurations and zoo.
Provides ready-to-use model architectures for different use cases.
"""

from typing import Optional
from config import Config, ModelConfig, TrainingConfig, GenerationConfig



class ModelZoo:
    """Collection of pre-configured models."""
    
    @staticmethod
    def tiny() -> Config:
        """Tiny model for testing and debugging."""
        return Config(
            model=ModelConfig(
                n_layer=1,
                n_embd=16,
                n_head=4,
                block_size=16,
                dropout=0.0,
                use_gelu=False,
                use_layernorm=False
            ),
            training=TrainingConfig(
                num_steps=1000,
                learning_rate=0.01,
                lr_schedule='linear'
            ),
            generation=GenerationConfig(
                temperature=0.8,
                max_length=20
            )
        )
    
    @staticmethod
    def small() -> Config:
        """Small model for quick experiments."""
        return Config(
            model=ModelConfig(
                n_layer=2,
                n_embd=32,
                n_head=4,
                block_size=32,
                dropout=0.1,
                use_gelu=True,
                use_layernorm=False
            ),
            training=TrainingConfig(
                num_steps=5000,
                learning_rate=0.005,
                lr_schedule='cosine',
                warmup_steps=200,
                weight_decay=0.01
            ),
            generation=GenerationConfig(
                temperature=0.7,
                top_k=40,
                max_length=50
            )
        )
    
    @staticmethod
    def medium() -> Config:
        """Medium model for serious training."""
        return Config(
            model=ModelConfig(
                n_layer=4,
                n_embd=64,
                n_head=8,
                block_size=64,
                dropout=0.15,
                use_gelu=True,
                use_layernorm=True
            ),
            training=TrainingConfig(
                num_steps=20000,
                learning_rate=0.003,
                lr_schedule='cosine',
                warmup_steps=1000,
                weight_decay=0.01,
                grad_clip=1.0
            ),
            generation=GenerationConfig(
                temperature=0.6,
                top_k=40,
                top_p=0.95,
                max_length=100
            )
        )
    
    @staticmethod
    def large() -> Config:
        """Large model for best quality."""
        return Config(
            model=ModelConfig(
                n_layer=8,
                n_embd=128,
                n_head=8,
                block_size=128,
                dropout=0.2,
                use_gelu=True,
                use_layernorm=True
            ),
            training=TrainingConfig(
                num_steps=50000,
                learning_rate=0.001,
                lr_schedule='cosine',
                warmup_steps=2000,
                weight_decay=0.01,
                grad_clip=1.0
            ),
            generation=GenerationConfig(
                temperature=0.5,
                top_k=50,
                top_p=0.95,
                max_length=200
            )
        )
    
    @staticmethod
    def names_generator() -> Config:
        """Optimized for name generation."""
        return Config(
            model=ModelConfig(
                n_layer=3,
                n_embd=48,
                n_head=6,
                block_size=20,
                dropout=0.1,
                use_gelu=True,
                use_layernorm=False
            ),
            training=TrainingConfig(
                num_steps=10000,
                learning_rate=0.01,
                lr_schedule='cosine',
                warmup_steps=500
            ),
            generation=GenerationConfig(
                temperature=0.8,
                top_k=10,
                max_length=15
            )
        )
    
    @staticmethod
    def code_generator() -> Config:
        """Optimized for code generation."""
        return Config(
            model=ModelConfig(
                n_layer=6,
                n_embd=96,
                n_head=8,
                block_size=256,
                dropout=0.1,
                use_gelu=True,
                use_layernorm=True
            ),
            training=TrainingConfig(
                num_steps=30000,
                learning_rate=0.002,
                lr_schedule='cosine',
                warmup_steps=1000,
                weight_decay=0.01
            ),
            generation=GenerationConfig(
                temperature=0.4,
                top_k=40,
                top_p=0.95,
                max_length=512
            )
        )
    
    @staticmethod
    def chat_model() -> Config:
        """Optimized for conversational AI."""
        return Config(
            model=ModelConfig(
                n_layer=6,
                n_embd=128,
                n_head=8,
                block_size=512,
                dropout=0.1,
                use_gelu=True,
                use_layernorm=True
            ),
            training=TrainingConfig(
                num_steps=50000,
                learning_rate=0.0005,
                lr_schedule='cosine',
                warmup_steps=2000,
                weight_decay=0.01
            ),
            generation=GenerationConfig(
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                max_length=256
            )
        )
    
    @staticmethod
    def list_models():
        """List all available model configurations."""
        models = {
            'tiny': 'Tiny model (1 layer, 16 dims) - for testing',
            'small': 'Small model (2 layers, 32 dims) - quick experiments',
            'medium': 'Medium model (4 layers, 64 dims) - serious training',
            'large': 'Large model (8 layers, 128 dims) - best quality',
            'names': 'Name generator (3 layers, 48 dims) - optimized for names',
            'code': 'Code generator (6 layers, 96 dims) - optimized for code',
            'chat': 'Chat model (6 layers, 128 dims) - conversational AI'
        }
        
        print("Available model configurations:")
        for name, desc in models.items():
            print(f"  {name:10s} - {desc}")
        
        return list(models.keys())


def create_model(config_name: str, save_path: Optional[str] = None):
    """Create a model from zoo configuration."""
    zoo = ModelZoo()
    
    config_map = {
        'tiny': zoo.tiny,
        'small': zoo.small,
        'medium': zoo.medium,
        'large': zoo.large,
        'names': zoo.names_generator,
        'code': zoo.code_generator,
        'chat': zoo.chat_model
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown config: {config_name}. Use one of: {list(config_map.keys())}")
    
    config = config_map[config_name]()
    
    from model import GPT
    
    model = GPT(
        vocab_size=27,  # Will be updated when data is loaded
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
        use_gelu=config.model.use_gelu,
        use_layernorm=config.model.use_layernorm
    )
    
    print(f"Created {config_name} model:")
    print(f"  Parameters: {model.num_params():,}")
    print(f"  Layers: {model.n_layer}")
    print(f"  Embedding dim: {model.n_embd}")
    print(f"  Heads: {model.n_head}")
    
    if save_path:
        config.to_yaml(save_path)
        print(f"Config saved to {save_path}")
    
    return model, config


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Zoo')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--create', type=str, help='Create model config (tiny/small/medium/large/names/code/chat)')
    parser.add_argument('--save', type=str, help='Save config to file')
    
    args = parser.parse_args()
    
    if args.list:
        ModelZoo.list_models()
    elif args.create:
        create_model(args.create, args.save)
    else:
        parser.print_help()
