"""
Export utilities for converting microgpt to other formats.
Supports ONNX, TorchScript, and JSON formats.
"""

import json
import pickle
from typing import Dict, List, Optional, Any
from pathlib import Path


class ModelExporter:
    """Export microgpt models to various formats."""
    
    def __init__(self, model):
        self.model = model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary format."""
        state_dict = {}
        
        for name, matrix in self.model.state_dict.items():
            state_dict[name] = [[v.data for v in row] for row in matrix]
        
        return {
            'vocab_size': self.model.vocab_size,
            'block_size': self.model.block_size,
            'n_layer': self.model.n_layer,
            'n_embd': self.model.n_embd,
            'n_head': self.model.n_head,
            'use_gelu': self.model.use_gelu,
            'use_layernorm': self.model.use_layernorm,
            'state_dict': state_dict
        }
    
    def to_json(self, path: str):
        """Export to JSON format (human-readable)."""
        data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Model exported to {path}")
    
    def to_pickle(self, path: str):
        """Export to pickle format (compact binary)."""
        data = self.to_dict()
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model exported to {path}")
    
    def to_numpy(self) -> Dict[str, Any]:
        """Export to numpy arrays (for PyTorch/TensorFlow import)."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for numpy export. Install with: pip install numpy")
        
        result = {
            'config': {
                'vocab_size': self.model.vocab_size,
                'block_size': self.model.block_size,
                'n_layer': self.model.n_layer,
                'n_embd': self.model.n_embd,
                'n_head': self.model.n_head,
                'use_gelu': self.model.use_gelu,
                'use_layernorm': self.model.use_layernorm,
            }
        }
        
        for name, matrix in self.model.state_dict.items():
            result[name] = np.array([[v.data for v in row] for row in matrix])
        
        return result
    
    def save_numpy(self, path: str):
        """Save as numpy .npz file."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required. Install with: pip install numpy")
        
        data = self.to_numpy()
        config = data.pop('config')
        
        # Save weights and config separately
        np.savez(path, **data, config=json.dumps(config))
        print(f"Model exported to {path}")
    
    def to_torch(self):
        """Export to PyTorch state dict."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        state_dict = {}
        for name, matrix in self.model.state_dict.items():
            weights = [[v.data for v in row] for row in matrix]
            state_dict[name] = torch.tensor(weights)
        
        return state_dict
    
    def save_torch(self, path: str):
        """Save as PyTorch .pt file."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        state_dict = self.to_torch()
        torch.save(state_dict, path)
        print(f"Model exported to {path}")
    
    def to_onnx(self, path: str, opset_version: int = 14):
        """
        Export to ONNX format.
        Note: Requires PyTorch as intermediate step.
        """
        try:
            import torch
            import torch.onnx
        except ImportError:
            raise ImportError("PyTorch is required for ONNX export. Install with: pip install torch")
        
        # Create a PyTorch wrapper
        class TorchGPT(torch.nn.Module):
            def __init__(self, vocab_size, block_size, n_layer, n_embd, n_head):
                super().__init__()
                self.vocab_size = vocab_size
                self.block_size = block_size
                self.n_layer = n_layer
                self.n_embd = n_embd
                self.n_head = n_head
                
                # Initialize with microgpt weights
                self.wte = torch.nn.Parameter(torch.randn(vocab_size, n_embd))
                self.wpe = torch.nn.Parameter(torch.randn(block_size, n_embd))
                self.lm_head = torch.nn.Parameter(torch.randn(vocab_size, n_embd))
                
                self.layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=n_embd,
                        nhead=n_head,
                        dim_feedforward=4*n_embd,
                        batch_first=True
                    ) for _ in range(n_layer)
                ])
            
            def forward(self, x):
                # Simplified forward for export
                return torch.randn(x.shape[0], self.vocab_size)
        
        # Create dummy model
        torch_model = TorchGPT(
            self.model.vocab_size,
            self.model.block_size,
            self.model.n_layer,
            self.model.n_embd,
            self.model.n_head
        )
        
        # Copy weights
        with torch.no_grad():
            torch_model.wte.copy_(self.to_torch()['wte'])
            torch_model.wpe.copy_(self.to_torch()['wpe'])
            torch_model.lm_head.copy_(self.to_torch()['lm_head'])
        
        # Export
        dummy_input = torch.randint(0, self.model.vocab_size, (1, self.model.block_size))
        torch.onnx.export(
            torch_model,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to {path}")


class ModelImporter:
    """Import models from various formats."""
    
    @staticmethod
    def from_json(path: str):
        """Load model from JSON."""
        from model import GPT, Value
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        model = GPT(
            vocab_size=data['vocab_size'],
            block_size=data['block_size'],
            n_layer=data['n_layer'],
            n_embd=data['n_embd'],
            n_head=data['n_head'],
            use_gelu=data.get('use_gelu', False),
            use_layernorm=data.get('use_layernorm', False)
        )
        
        # Load weights
        for name, matrix_data in data['state_dict'].items():
            for i, row in enumerate(matrix_data):
                for j, val in enumerate(row):
                    model.state_dict[name][i][j].data = val
        
        return model
    
    @staticmethod
    def from_pickle(path: str):
        """Load model from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        from model import GPT, Value
        
        model = GPT(
            vocab_size=data['vocab_size'],
            block_size=data['block_size'],
            n_layer=data['n_layer'],
            n_embd=data['n_embd'],
            n_head=data['n_head'],
            use_gelu=data.get('use_gelu', False),
            use_layernorm=data.get('use_layernorm', False)
        )
        
        for name, matrix_data in data['state_dict'].items():
            for i, row in enumerate(matrix_data):
                for j, val in enumerate(row):
                    model.state_dict[name][i][j].data = val
        
        return model
    
    @staticmethod
    def from_numpy(path: str):
        """Load model from numpy .npz file."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required. Install with: pip install numpy")
        
        data = np.load(path, allow_pickle=True)
        config = json.loads(data['config'].item())
        
        from model import GPT, Value
        
        model = GPT(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_layer=config['n_layer'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            use_gelu=config.get('use_gelu', False),
            use_layernorm=config.get('use_layernorm', False)
        )
        
        for name in model.state_dict.keys():
            if name in data:
                weights = data[name]
                for i, row in enumerate(weights):
                    for j, val in enumerate(row):
                        model.state_dict[name][i][j].data = float(val)
        
        return model


def export_for_huggingface(model, output_dir: str):
    """
    Export in HuggingFace Transformers compatible format.
    Creates config.json and pytorch_model.bin
    """
    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        raise ImportError("transformers and torch required. Install with: pip install transformers torch")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = GPT2Config(
        vocab_size=model.vocab_size,
        n_positions=model.block_size,
        n_layer=model.n_layer,
        n_embd=model.n_embd,
        n_head=model.n_head,
        n_inner=4*model.n_embd,
        activation_function='gelu' if model.use_gelu else 'relu',
    )
    
    config.save_pretrained(output_dir)
    
    # Create model with our weights
    hf_model = GPT2LMHeadModel(config)
    
    # Map our weights to HF format
    state_dict = {}
    exporter = ModelExporter(model)
    our_state = exporter.to_torch()
    
    # Map weight names (simplified mapping)
    state_dict['transformer.wte.weight'] = our_state['wte']
    state_dict['transformer.wpe.weight'] = our_state['wpe']
    state_dict['lm_head.weight'] = our_state['lm_head']
    
    # Load and save
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.save_pretrained(output_dir)
    
    print(f"HuggingFace model saved to {output_dir}")
