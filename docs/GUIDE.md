# microgpt Complete Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Training](#training)
4. [Generation](#generation)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Deployment](#deployment)

## Quick Start

### Installation
```bash
git clone https://github.com/iamGodofall/karpathy-microgpt-by-Enock.git
cd karpathy-microgpt-by-Enock
pip install -e .
```

### Train Your First Model
```bash
# Using CLI
python cli.py train --steps 1000 --experiment-name my_first_model

# Using Python API
from model_zoo import create_model
model, config = create_model('small')
```

### Generate Text
```bash
python cli.py generate --temperature 0.8 --num-samples 5
```

## Architecture Overview

### Model Components
- **Embeddings**: Token + positional embeddings
- **Transformer Layers**: Multi-head self-attention + MLP
- **Normalization**: RMSNorm or LayerNorm
- **Activations**: ReLU or GELU

### Key Parameters
| Parameter | Description | Typical Values |
|-----------|-------------|--------------|
| n_layer | Number of transformer layers | 1-8 |
| n_embd | Embedding dimension | 16-128 |
| n_head | Attention heads | 4-8 |
| block_size | Context window | 16-512 |

## Training

### Basic Training
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
    block_size=32,
    n_layer=2,
    n_embd=32,
    n_head=4
)

# Configure training
config = TrainingConfig(
    num_steps=5000,
    learning_rate=0.005,
    lr_schedule='cosine'
)

# Train
trainer = Trainer(model, config)
trainer.train(train_docs, val_docs, 
              loader.tokenizer.char_to_idx,
              loader.tokenizer.bos_token)
```

### Advanced Training Features

#### 1. Gradient Accumulation
```python
from advanced_features import GradientAccumulator

accumulator = GradientAccumulator(accumulation_steps=4)
```

#### 2. Mixed Precision
```python
from advanced_features import MixedPrecisionTrainer

mp_trainer = MixedPrecisionTrainer(loss_scale=2.0**16)
```

#### 3. Distributed Training
```python
from distributed import DataParallelTrainer, DistributedConfig

config = DistributedConfig(world_size=4, rank=0)
trainer = DataParallelTrainer(model, config)
```

## Generation

### Sampling Methods

#### Temperature Sampling
```python
tokens = model.generate(
    start_token,
    max_length=50,
    temperature=0.7  # Lower = more focused, higher = more random
)
```

#### Top-k Sampling
```python
tokens = model.generate(
    start_token,
    max_length=50,
    temperature=0.7,
    top_k=40  # Only sample from top 40 tokens
)
```

#### Nucleus (Top-p) Sampling
```python
tokens = model.generate(
    start_token,
    max_length=50,
    top_p=0.9  # Sample from smallest set with cumulative prob >= 0.9
)
```

#### Beam Search
```python
from advanced_features import BeamSearchDecoder

decoder = BeamSearchDecoder(beam_width=5)
sequence, score = decoder.decode(model, start_token)
```

#### Contrastive Search
```python
from advanced_features import ContrastiveSearchDecoder

decoder = ContrastiveSearchDecoder(k=5, alpha=0.6)
tokens = decoder.decode(model, start_token, max_length=50)
```

## Advanced Features

### Quantization
```python
from quantization import Quantizer, QuantizationConfig

config = QuantizationConfig(bits=8, symmetric=True)
quantizer = Quantizer(config)
quantized_model = quantizer.quantize_model(model)
quantizer.save_quantized('model_int8.json')
```

### Export Formats
```python
from export import ModelExporter

exporter = ModelExporter(model)
exporter.to_json('model.json')
exporter.to_pickle('model.pkl')
exporter.save_numpy('model.npz')
# exporter.save_torch('model.pt')  # Requires PyTorch
# exporter.to_onnx('model.onnx')   # Requires PyTorch
```

### Benchmarking
```python
from benchmark import run_full_benchmark

run_full_benchmark(model, tokenizer)
```

### Interpretability
```python
from interpretability import analyze_model

analyze_model(model, "Hello world", tokenizer)
```

## API Reference

### REST API

Start the server:
```bash
python api_server.py --checkpoint checkpoints/best_model.pkl --port 5000
```

Endpoints:
- `POST /generate` - Generate text
- `POST /chat` - Chat completion (OpenAI-compatible)
- `POST /embeddings` - Get embeddings
- `GET /health` - Health check

Example:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 20}'
```

### Web Interface
```bash
python web_app.py
```
Then open http://localhost:5000 in your browser.

### Chat Interface
```bash
python chat.py --checkpoint checkpoints/best_model.pkl --role default
```

## Deployment

### Docker
```bash
docker build -t microgpt .
docker run -p 5000:5000 microgpt
```

### Environment Variables
- `MICROGPT_CHECKPOINT` - Default checkpoint path
- `MICROGPT_PORT` - Server port
- `MICROGPT_HOST` - Server host

### Production Tips
1. Use quantized models for faster inference
2. Enable gradient checkpointing for large models
3. Use beam search for quality, top-p for creativity
4. Monitor memory usage with benchmark tools
