# microgpt 🚀

A complete, production-ready ecosystem built around Andrej Karpathy's minimal GPT implementation in pure Python. This project transforms the educational microgpt into a fully-featured training and inference framework.

## 🌟 Features

### Core Architecture
- **Pure Python**: No PyTorch, TensorFlow, or JAX dependencies
- **Autograd Engine**: Custom automatic differentiation with backpropagation
- **Transformer Architecture**: Multi-layer, multi-head attention with configurable depth
- **Flexible Design**: Support for different normalizations (RMSNorm/LayerNorm) and activations (ReLU/GELU)

### Training Infrastructure
- **Adam Optimizer**: With gradient clipping and weight decay
- **Learning Rate Scheduling**: Linear, cosine annealing, and warmup
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Checkpointing**: Save/load models in JSON or pickle format
- **Logging**: Structured metrics tracking with JSONL format

### Data & Tokenization
- **Character-level Tokenizer**: Simple and fast
- **BPE Tokenizer**: Byte-pair encoding for better compression
- **Data Pipeline**: Loading, preprocessing, and batching

### Generation & Sampling
- **Temperature Sampling**: Control randomness
- **Top-k Sampling**: Limit to k most likely tokens
- **Nucleus (Top-p) Sampling**: Dynamic vocabulary truncation
- **Interactive Mode**: Chat-like interface

### Tools & Interfaces
- **CLI**: Full command-line interface for train/generate/eval
- **Web UI**: Flask-based web interface
- **Visualization**: Training curves and generation analysis
- **Testing**: Comprehensive unit and integration tests

## 📁 Project Structure

```
microgpt/
├── microgpt.py          # Original minimal implementation
├── model.py             # Enhanced GPT model with all features
├── config.py            # Configuration management
├── trainer.py           # Training loop and optimizer
├── data.py              # Data loading and tokenization
├── checkpoint.py        # Model checkpointing
├── logger.py            # Training metrics logging
├── cli.py               # Command-line interface
├── web_app.py           # Web interface
├── visualize.py         # Visualization tools
├── test_microgpt.py     # Test suite
└── README.md            # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd microgpt

# No dependencies required for core functionality!
# Optional: Install visualization dependencies
pip install pyyaml plotly matplotlib flask
```

### Training

```bash
# Train with default settings
python cli.py train

# Train with custom configuration
python cli.py train --steps 2000 --lr 0.005 --batch-size 4

# Train with config file
python cli.py train --config config.yaml
```

### Generation

```bash
# Generate samples
python cli.py generate --temperature 0.7 --num-samples 10

# Interactive mode
python cli.py generate --interactive

# With advanced sampling
python cli.py generate --temperature 0.8 --top-k 40 --top-p 0.9
```

### Evaluation

```bash
# Evaluate model
python cli.py eval --detailed
```

### Web Interface

```bash
# Start web server
python web_app.py

# Open http://localhost:5000 in your browser
```

## ⚙️ Configuration

Create a `config.yaml` file:

```yaml
model:
  n_layer: 2
  n_embd: 32
  n_head: 4
  block_size: 32
  dropout: 0.1
  use_gelu: true
  use_layernorm: false

training:
  num_steps: 5000
  batch_size: 1
  learning_rate: 0.01
  lr_schedule: cosine
  warmup_steps: 100
  weight_decay: 0.01
  grad_clip: 1.0
  val_split: 0.1
  eval_interval: 100
  save_interval: 500

generation:
  temperature: 0.7
  top_k: 0
  top_p: 1.0
  max_length: 50
  num_samples: 10
  seed: 42

checkpoint_dir: checkpoints
log_dir: logs
```

## 🧪 Testing

```bash
# Run all tests
python test_microgpt.py

# Or use pytest
pytest test_microgpt.py -v
```

## 📊 Visualization

```bash
# Plot training curves
python visualize.py plot logs/experiment.jsonl training_curves.html

# Analyze generated samples
python visualize.py analyze samples.txt

# Compare checkpoints
python visualize.py compare checkpoints/model1.pkl checkpoints/model2.pkl
```

## 🏗️ Architecture Details

### Model Components

| Component | Description |
|-----------|-------------|
| Token Embeddings | Learned embeddings for each token |
| Position Embeddings | Learned positional encodings |
| Multi-Head Attention | Parallel attention heads with scaled dot-product |
| Feed-Forward Network | 2-layer MLP with expansion factor 4x |
| Normalization | RMSNorm or LayerNorm |
| Activation | ReLU or GELU |

### Training Features

- **Adam Optimizer**: Adaptive learning rates with momentum
- **Gradient Clipping**: Prevents exploding gradients
- **Weight Decay**: L2 regularization
- **Learning Rate Scheduling**: Linear decay or cosine annealing
- **Validation**: Automatic train/val split with monitoring
- **Early Stopping**: Stop when validation loss plateaus

## 📈 Performance Tips

1. **Start Small**: Begin with n_layer=1, n_embd=16 for quick experiments
2. **Scale Gradually**: Increase depth and width as needed
3. **Use Dropout**: 0.1-0.2 for regularization
4. **Learning Rate**: Start with 0.01, adjust based on loss curves
5. **Batch Size**: Increase for more stable gradients (if memory allows)

## 🔧 Advanced Usage

### Custom Dataset

```python
from data import DataLoader

loader = DataLoader()
train_docs, val_docs = loader.load_file('my_data.txt', val_split=0.1)
```

### Custom Training Loop

```python
from model import GPT
from trainer import Trainer
from config import Config

config = Config()
model = GPT(vocab_size=100, block_size=32, n_layer=2, n_embd=64)
trainer = Trainer(model, config.training)

# Custom training loop
for step in range(1000):
    loss = trainer.train_step(batch_tokens, step)
```

### Programmatic Generation

```python
tokens = model.generate(
    token_id=tokenizer.bos_token,
    max_length=50,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)
text = tokenizer.decode(tokens)
```

## 📝 Citation

Based on Andrej Karpathy's microgpt:
> "The most atomic way to train and run inference for a GPT in pure, dependency-free Python."

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Multi-GPU support
- [ ] Mixed precision training
- [ ] More tokenizers (WordPiece, SentencePiece)
- [ ] Quantization
- [ ] ONNX export
- [ ] More architectures (RoPE, ALiBi, etc.)
