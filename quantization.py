"""
Model quantization for efficient inference.
Implements INT8 and INT4 quantization with symmetric and asymmetric schemes.
"""

import math
from typing import List, Dict, Tuple, Optional
from model import Value, GPT


class QuantizedTensor:
    """Represents a quantized tensor with scale and zero point."""
    
    def __init__(self, quantized_data: List[List[int]], 
                 scale: float, zero_point: int = 0,
                 bits: int = 8):
        self.data = quantized_data
        self.scale = scale
        self.zero_point = zero_point
        self.bits = bits
    
    def dequantize(self) -> List[List[float]]:
        """Convert back to float."""
        return [[(q - self.zero_point) * self.scale for q in row] 
                for row in self.data]


class QuantizationConfig:
    """Configuration for quantization."""
    
    def __init__(self, bits: int = 8, symmetric: bool = True,
                 per_channel: bool = False, group_size: int = 128):
        self.bits = bits  # 8 or 4
        self.symmetric = symmetric
        self.per_channel = per_channel  # Per-row quantization
        self.group_size = group_size  # For grouped quantization
    
    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1)) if self.symmetric else 0
    
    @property
    def qmax(self) -> int:
        return (2 ** (self.bits - 1)) - 1 if self.symmetric else (2 ** self.bits) - 1


class Quantizer:
    """Quantize model weights for efficient inference."""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.quantized_state: Dict[str, QuantizedTensor] = {}
    
    def quantize_tensor(self, tensor: List[List[Value]]) -> QuantizedTensor:
        """Quantize a weight matrix."""
        # Extract float values
        float_data = [[v.data for v in row] for row in tensor]
        
        if self.config.per_channel:
            # Per-row quantization
            quantized_rows = []
            scales = []
            
            for row in float_data:
                q_row, scale = self._quantize_row(row)
                quantized_rows.append(q_row)
                scales.append(scale)
            
            # Store scales separately for each row
            return QuantizedTensor(quantized_rows, scales, bits=self.config.bits)
        else:
            # Per-tensor quantization
            flat = [v for row in float_data for v in row]
            min_val = min(flat)
            max_val = max(flat)
            
            if self.config.symmetric:
                abs_max = max(abs(min_val), abs(max_val))
                scale = abs_max / self.config.qmax
                zero_point = 0
            else:
                scale = (max_val - min_val) / (self.config.qmax - self.config.qmin)
                zero_point = round(-min_val / scale) if scale != 0 else 0
            
            if scale == 0:
                scale = 1.0
            
            quantized = [[self._quantize_value(v, scale, zero_point) for v in row] 
                        for row in float_data]
            
            return QuantizedTensor(quantized, scale, zero_point, self.config.bits)
    
    def _quantize_value(self, val: float, scale: float, zero_point: int) -> int:
        """Quantize a single value."""
        q = round(val / scale) + zero_point
        return max(self.config.qmin, min(self.config.qmax, q))
    
    def _quantize_row(self, row: List[float]) -> Tuple[List[int], float]:
        """Quantize a single row (for per-channel quantization)."""
        min_val = min(row)
        max_val = max(row)
        
        if self.config.symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            scale = abs_max / self.config.qmax if abs_max > 0 else 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / (self.config.qmax - self.config.qmin)
            zero_point = round(-min_val / scale) if scale != 0 else 0
        
        if scale == 0:
            scale = 1.0
        
        quantized = [self._quantize_value(v, scale, zero_point) for v in row]
        return quantized, scale
    
    def quantize_model(self, model: GPT) -> Dict[str, QuantizedTensor]:
        """Quantize all weights in the model."""
        self.quantized_state = {}
        
        for name, weights in model.state_dict.items():
            self.quantized_state[name] = self.quantize_tensor(weights)
        
        return self.quantized_state
    
    def save_quantized(self, path: str):
        """Save quantized model to file."""
        import json
        
        serializable = {}
        for name, qt in self.quantized_state.items():
            serializable[name] = {
                'data': qt.data,
                'scale': qt.scale,
                'zero_point': qt.zero_point,
                'bits': qt.bits
            }
        
        with open(path, 'w') as f:
            json.dump(serializable, f)
    
    @classmethod
    def load_quantized(cls, path: str) -> Dict[str, QuantizedTensor]:
        """Load quantized model from file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        result = {}
        for name, qt_data in data.items():
            result[name] = QuantizedTensor(
                qt_data['data'],
                qt_data['scale'],
                qt_data['zero_point'],
                qt_data['bits']
            )
        
        return result


class QuantizedGPT:
    """GPT model using quantized weights for inference."""
    
    def __init__(self, quantized_state: Dict[str, QuantizedTensor], 
                 config: Optional[QuantizationConfig] = None):
        self.quantized_state = quantized_state
        self.config = config or QuantizationConfig()
        self.n_layer = self._infer_num_layers()
        
        # Dequantize for computation (could be optimized further)
        self.state_dict = self._dequantize_all()
    
    def _infer_num_layers(self) -> int:
        """Infer number of layers from state dict keys."""
        layers = set()
        for key in self.quantized_state.keys():
            if key.startswith('layer'):
                layer_num = int(key.split('.')[0].replace('layer', ''))
                layers.add(layer_num)
        return max(layers) + 1 if layers else 1
    
    def _dequantize_all(self) -> Dict[str, List[List[Value]]]:
        """Dequantize all weights to Values."""
        from model import Value
        
        dequantized = {}
        for name, qt in self.quantized_state.items():
            float_data = qt.dequantize()
            dequantized[name] = [[Value(v) for v in row] for row in float_data]
        
        return dequantized
    
    def forward(self, token_id: int, pos_id: int,
                keys: List[List], values: List[List]) -> List[Value]:
        """Forward pass with quantized weights."""
        from model import linear, rmsnorm, softmax
        
        # Embeddings
        tok_emb = self.state_dict['wte'][token_id]
        pos_emb = self.state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)
        
        # Transformer layers
        for li in range(self.n_layer):
            # Attention
            x_residual = x
            x = rmsnorm(x)
            
            q = linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = linear(x, self.state_dict[f'layer{li}.attn_wv'])
            
            keys[li].append(k)
            values[li].append(v)
            
            # Multi-head attention
            head_dim = len(q) // 4  # Assuming 4 heads
            x_attn = []
            for h in range(4):
                hs = h * head_dim
                q_h = q[hs:hs+head_dim]
                k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+head_dim] for vi in values[li]]
                
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / 
                              (head_dim ** 0.5) for t in range(len(k_h))]
                attn_weights = softmax(attn_logits)
                
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                           for j in range(head_dim)]
                x_attn.extend(head_out)
            
            x = linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            
            # MLP
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]
        
        # Output
        logits = linear(x, self.state_dict['lm_head'])
        return logits


def quantize_model(model: GPT, bits: int = 8) -> QuantizedGPT:
    """Convenience function to quantize a model."""
    config = QuantizationConfig(bits=bits)
    quantizer = Quantizer(config)
    quantizer.quantize_model(model)
    return QuantizedGPT(quantizer.quantized_state, config)
