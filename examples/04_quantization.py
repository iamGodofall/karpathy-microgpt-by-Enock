"""
Example 4: Model Quantization
Demonstrates INT8 and INT4 quantization for efficient inference.
"""

from model import GPT
from data import DataLoader
from quantization import Quantizer, QuantizationConfig, quantize_model


def main():
    print("="*60)
    print("Example 4: Model Quantization")
    print("="*60)
    
    # Create a model
    print("\n1. Creating model...")
    loader = DataLoader()
    loader.load_names(val_split=0)
    
    model = GPT(
        vocab_size=loader.tokenizer.vocab_size,
        block_size=32,
        n_layer=2,
        n_embd=32,
        n_head=4
    )
    print(f"   Original parameters: {model.num_params():,}")
    
    # Quantize to INT8
    print("\n2. Quantizing to INT8...")
    config_8bit = QuantizationConfig(bits=8, symmetric=True)
    quantizer_8bit = Quantizer(config_8bit)
    quantized_8bit = quantizer_8bit.quantize_model(model)
    
    print(f"   Quantized {len(quantized_8bit)} tensors")
    
    # Check size reduction
    original_size = model.num_params() * 4  # 4 bytes per float
    quantized_size = sum(
        len(row) * 1 for qt in quantized_8bit.values() for row in qt.data
    )  # 1 byte per int8
    
    print(f"   Original size: {original_size / 1024:.1f} KB")
    print(f"   Quantized size: {quantized_size / 1024:.1f} KB")
    print(f"   Compression: {original_size / quantized_size:.1f}x")
    
    # Quantize to INT4
    print("\n3. Quantizing to INT4...")
    config_4bit = QuantizationConfig(bits=4, symmetric=True)
    quantizer_4bit = Quantizer(config_4bit)
    quantized_4bit = quantizer_4bit.quantize_model(model)
    
    quantized_size_4bit = sum(
        len(row) * 0.5 for qt in quantized_4bit.values() for row in qt.data
    )  # 0.5 bytes per int4
    
    print(f"   Quantized size: {quantized_size_4bit / 1024:.1f} KB")
    print(f"   Compression: {original_size / quantized_size_4bit:.1f}x")
    
    # Save quantized model
    print("\n4. Saving quantized model...")
    quantizer_8bit.save_quantized('model_int8.json')
    print("   Saved to model_int8.json")
    
    # Load and test
    print("\n5. Loading quantized model...")
    from quantization import QuantizedGPT
    
    loaded = Quantizer.load_quantized('model_int8.json')
    q_model = QuantizedGPT(loaded, config_8bit)
    print(f"   Loaded quantized model with {q_model.n_layer} layers")
    
    # Generate with quantized model
    print("\n6. Generating with quantized model...")
    tokens = q_model.generate(
        loader.tokenizer.bos_token,
        max_length=15,
        temperature=0.8
    )
    text = loader.tokenizer.decode(tokens)
    print(f"   Generated: {text}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == '__main__':
    main()
