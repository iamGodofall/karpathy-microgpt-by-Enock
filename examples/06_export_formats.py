"""
Example 6: Exporting Models
Demonstrates various export formats for model deployment.
"""

from model import GPT
from data import DataLoader
from export import ModelExporter, ModelImporter


def main():
    print("="*60)
    print("Example 6: Model Export Formats")
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
    print(f"   Parameters: {model.num_params():,}")
    
    # Export to different formats
    print("\n2. Exporting to Different Formats")
    print("-" * 40)
    
    exporter = ModelExporter(model)
    
    # JSON (human-readable)
    print("   Exporting to JSON...")
    exporter.to_json('model_export.json')
    
    # Pickle (binary)
    print("   Exporting to Pickle...")
    exporter.to_pickle('model_export.pkl')
    
    # Try NumPy (if available)
    try:
        print("   Exporting to NumPy...")
        exporter.save_numpy('model_export.npz')
    except ImportError:
        print("   NumPy not available, skipping...")
    
    # Try PyTorch (if available)
    try:
        print("   Exporting to PyTorch...")
        exporter.save_torch('model_export.pt')
    except ImportError:
        print("   PyTorch not available, skipping...")
    
    # Import back
    print("\n3. Importing from JSON...")
    imported_model = ModelImporter.from_json('model_export.json')
    print(f"   Imported model with {imported_model.num_params():,} parameters")
    
    print("\n4. Importing from Pickle...")
    imported_model = ModelImporter.from_pickle('model_export.pkl')
    print(f"   Imported model with {imported_model.num_params():,} parameters")
    
    # Verify equivalence
    print("\n5. Verifying Model Equivalence")
    print("-" * 40)
    
    # Generate with original
    model.set_training(False)
    tokens_orig = model.generate(loader.tokenizer.bos_token, max_length=10)
    text_orig = loader.tokenizer.decode(tokens_orig)
    
    # Generate with imported
    imported_model.set_training(False)
    tokens_imported = imported_model.generate(loader.tokenizer.bos_token, max_length=10)
    text_imported = loader.tokenizer.decode(tokens_imported)
    
    print(f"   Original:   {text_orig}")
    print(f"   Imported:   {text_imported}")
    print(f"   Match: {text_orig == text_imported}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == '__main__':
    main()
