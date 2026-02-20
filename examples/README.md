# microgpt Examples

This directory contains comprehensive examples demonstrating all features of the microgpt ecosystem.

## Quick Start Examples

| Example | Description | Run Command |
|---------|-------------|-------------|
| `01_basic_training.py` | Train your first model | `python examples/01_basic_training.py` |
| `02_advanced_generation.py` | Different sampling methods | `python examples/02_advanced_generation.py` |
| `03_model_zoo.py` | Pre-configured models | `python examples/03_model_zoo.py` |
| `04_quantization.py` | INT8/INT4 quantization | `python examples/04_quantization.py` |
| `05_interpretability.py` | Attention & neuron analysis | `python examples/05_interpretability.py` |
| `06_export_formats.py` | Export to various formats | `python examples/06_export_formats.py` |

## Running All Examples

```bash
# Run all examples
for f in examples/*.py; do python "$f"; done
```

## What Each Example Covers

### 01_basic_training.py
- Loading the names dataset
- Creating a GPT model
- Configuring training hyperparameters
- Training loop with progress tracking
- Generating samples from trained model

### 02_advanced_generation.py
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Beam search decoding
- Contrastive search

### 03_model_zoo.py
- Listing available model configurations
- Creating tiny/small/medium/large models
- Custom configuration
- Comparative benchmarking

### 04_quantization.py
- INT8 quantization
- INT4 quantization
- Size reduction analysis
- Quantized inference

### 05_interpretability.py
- Attention pattern visualization
- Neuron activation analysis
- Linear probing
- Full model analysis

### 06_export_formats.py
- JSON export (human-readable)
- Pickle export (binary)
- NumPy export
- PyTorch export
- Model import verification

## Advanced Usage

### Custom Training Script
```python
from model import GPT
from trainer import Trainer, TrainingConfig
from data import DataLoader

# Load data
loader = DataLoader()
train_docs, val_docs = loader.load_names(val_split=0.1)

# Create model
model = GPT(
    vocab_size=loader.tokenizer.vocab_size,
    block_size=64,
    n_layer=4,
    n_embd=64,
    n_head=8,
    dropout=0.1,
    use_gelu=True
)

# Configure training
config = TrainingConfig(
    num_steps=10000,
    learning_rate=0.003,
    lr_schedule='cosine',
    warmup_steps=500,
    weight_decay=0.01,
    grad_clip=1.0
)

# Train
trainer = Trainer(model, config)
trainer.train(train_docs, val_docs, 
              loader.tokenizer.char_to_idx,
              loader.tokenizer.bos_token)
```

### Custom Generation
```python
from advanced_features import BeamSearchDecoder, ContrastiveSearchDecoder

# Beam search
decoder = BeamSearchDecoder(beam_width=5)
tokens, score = decoder.decode(model, start_token)

# Contrastive search
decoder = ContrastiveSearchDecoder(k=5, alpha=0.6)
tokens = decoder.decode(model, start_token, max_length=50)
```

## Tips

1. **Start Small**: Use `tiny` or `small` models for experimentation
2. **Monitor Memory**: Use `benchmark.py` to check memory usage
3. **Quantize for Speed**: Use INT8 for 4x faster inference
4. **Save Checkpoints**: Use checkpoint manager for long training runs
5. **Visualize**: Use interpretability tools to understand your model

## Troubleshooting

**Out of Memory**: Reduce `n_embd`, `n_layer`, or `block_size`
**Slow Training**: Enable gradient checkpointing or use quantization
**Poor Generation**: Increase model size or train longer
**Overfitting**: Add dropout, increase weight decay, use early stopping
