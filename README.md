# 🧠 microgpt

[![CI](https://github.com/iamGodofall/karpathy-microgpt-by-Enock/actions/workflows/tests.yml/badge.svg)](https://github.com/iamGodofall/karpathy-microgpt-by-Enock/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **The most atomic way to train and run inference for a GPT in pure, dependency-free Python.**

This is a complete, production-ready ecosystem built around Andrej Karpathy's minimal GPT implementation. Everything you need to train, deploy, and scale language models from scratch.

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/iamGodofall/karpathy-microgpt-by-Enock.git
cd karpathy-microgpt-by-Enock
pip install -e .

# Train a model
python cli.py train --steps 1000 --experiment-name my_model

# Generate text
python cli.py generate --temperature 0.8 --num-samples 5

# Start API server
python api_server.py --checkpoint checkpoints/best_model.pkl
```

## 📦 What's Included

### Core Components
| Component | Description |
|-----------|-------------|
| `model.py` | Enhanced GPT with GELU, dropout, LayerNorm, multi-layer support |
| `trainer.py` | Adam optimizer with LR scheduling, gradient clipping, early stopping |
| `data.py` | Data loading with CharTokenizer and BPE tokenization |
| `config.py` | YAML/JSON configuration management |

### Advanced Features
| Feature | File | Description |
|---------|------|-------------|
| Quantization | `quantization.py` | INT8/INT4 quantization for efficient inference |
| Distributed | `distributed.py` | Data/pipeline parallel training |
| Export | `export.py` | Export to JSON, Pickle, NumPy, PyTorch, ONNX |
| Benchmark | `benchmark.py` | Speed, memory, and quality metrics |
| Interpretability | `interpretability.py` | Attention visualization, neuron analysis |

### Interfaces
| Interface | File | Description |
|-----------|------|-------------|
| CLI | `cli.py` | Command-line interface for all operations |
| Web UI | `web_app.py` | Flask-based web interface |
| REST API | `api_server.py` | OpenAI-compatible API server |
| Chat | `chat.py` | Interactive chat with conversation history |

### Pre-configured Models
| Model | Parameters | Use Case |
|-------|-----------|----------|
| tiny | ~1K | Testing, debugging |
| small | ~10K | Quick experiments |
| medium | ~100K | Standard training |
| large | ~1M | Serious training |
| names | ~3K | Name generation |
| code | ~50K | Code generation |
| chat | ~100K | Conversational AI |

## 🎯 Examples

```bash
# Run all examples
python examples/01_basic_training.py
python examples/02_advanced_generation.py
python examples/03_model_zoo.py
python examples/04_quantization.py
python examples/05_interpretability.py
python examples/06_export_formats.py
```

See [examples/README.md](examples/README.md) for detailed explanations.

## 📚 Documentation

- [Complete Guide](docs/GUIDE.md) - Comprehensive usage guide
- [API Reference](docs/GUIDE.md#api-reference) - REST API documentation
- [Examples](examples/) - Working code examples

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           microgpt Ecosystem            │
├─────────────────────────────────────────┤
│  Interfaces: CLI | Web | API | Chat    │
├─────────────────────────────────────────┤
│  Advanced: Quant | Distrib | Export     │
├─────────────────────────────────────────┤
│  Core: Model | Trainer | Data | Config   │
├─────────────────────────────────────────┤
│  Pure Python (no PyTorch/TensorFlow)   │
└─────────────────────────────────────────┘
```

## 🚂 Training

### Basic Training
```python
from model import GPT
from trainer import Trainer, TrainingConfig
from data import DataLoader

loader = DataLoader()
train_docs, val_docs = loader.load_names(val_split=0.1)

model = GPT(
    vocab_size=loader.tokenizer.vocab_size,
    block_size=32,
    n_layer=2,
    n_embd=32,
    n_head=4
)

config = TrainingConfig(
    num_steps=5000,
    learning_rate=0.005,
    lr_schedule='cosine'
)

trainer = Trainer(model, config)
trainer.train(train_docs, val_docs, 
              loader.tokenizer.char_to_idx,
              loader.tokenizer.bos_token)
```

### Advanced Training
```python
from model_zoo import create_model
from pretrain import Pretrainer

# Large-scale pretraining
model, config = create_model('large')
pretrainer = Pretrainer(model, config)
pretrainer.train_large_corpus('data/corpus.txt')
```

## 🎨 Generation

### Sampling Methods
```python
# Temperature sampling
tokens = model.generate(start_token, max_length=50, temperature=0.7)

# Top-k sampling
tokens = model.generate(start_token, max_length=50, top_k=40)

# Nucleus (top-p) sampling
tokens = model.generate(start_token, max_length=50, top_p=0.9)

# Beam search
from advanced_features import BeamSearchDecoder
decoder = BeamSearchDecoder(beam_width=5)
tokens, score = decoder.decode(model, start_token)

# Contrastive search
from advanced_features import ContrastiveSearchDecoder
decoder = ContrastiveSearchDecoder(k=5, alpha=0.6)
tokens = decoder.decode(model, start_token, max_length=50)
```

## 🔧 Deployment

### Docker
```bash
docker build -t microgpt .
docker run -p 5000:5000 microgpt
```

### API Server
```bash
python api_server.py --checkpoint checkpoints/best_model.pkl --port 5000

# Test
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 20}'
```

### Web Interface
```bash
python web_app.py
# Open http://localhost:5000
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest --cov=. tests/
```

## 📊 Benchmarking

```python
from benchmark import run_full_benchmark
from data import DataLoader

loader = DataLoader()
run_full_benchmark(model, loader.tokenizer)
```

## 🎓 Learning Resources

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's makemore](https://github.com/karpathy/makemore)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🌟 Acknowledgments

- Original concept by [Andrej Karpathy](https://karpathy.ai/)
- Built for educational and research purposes

---

**[⬆ Back to Top](#-microgpt)**
