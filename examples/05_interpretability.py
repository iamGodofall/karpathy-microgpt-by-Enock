"""
Example 5: Model Interpretability
Visualize attention patterns and analyze neuron activations.
"""

from model import GPT
from data import DataLoader
from interpretability import (
    AttentionVisualizer,
    NeuronAnalyzer,
    ProbingClassifier,
    analyze_model
)


def main():
    print("="*60)
    print("Example 5: Model Interpretability")
    print("="*60)
    
    # Create model
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
    model.set_training(False)
    
    # Analyze attention
    print("\n2. Analyzing Attention Patterns")
    print("-" * 40)
    
    text = "hello world"
    tokens = loader.tokenizer.encode(text)
    
    attn_viz = AttentionVisualizer(model)
    attentions = attn_viz.capture_attention(tokens)
    
    print(f"   Text: '{text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Captured {len(attentions)} attention layers")
    
    # Analyze neurons
    print("\n3. Analyzing Neuron Activations")
    print("-" * 40)
    
    neuron_analyzer = NeuronAnalyzer(model)
    activations = neuron_analyzer.capture_activations(tokens)
    
    print(f"   Captured activations from {len(activations)} layers")
    for layer_name, acts in activations.items():
        print(f"   {layer_name}: {len(acts)} positions")
    
    # Linear probing
    print("\n4. Linear Probing")
    print("-" * 40)
    
    prober = ProbingClassifier(model)
    
    # Create synthetic data for position probing
    test_texts = ["hello world test"] * 10
    accuracy = prober.probe_positional_information(test_texts, loader.tokenizer)
    
    print(f"   Position prediction accuracy: {accuracy:.2%}")
    
    # Full analysis
    print("\n5. Full Model Analysis")
    print("-" * 40)
    analyze_model(model, "hello world", loader.tokenizer)
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == '__main__':
    main()
