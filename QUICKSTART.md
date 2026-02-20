# microgpt Quickstart Guide

Get started with the most comprehensive pure-Python GPT ecosystem in minutes!

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/iamGodofall/karpathy-microgpt-by-Enock.git
cd karpathy-microgpt-by-Enock

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## 📖 Basic Usage

### 1. Train Your First Model

```bash
# Quick start with defaults
python main.py train

# With custom config
python main.py train --config config.yaml

# With CLI overrides
python main.py train --epochs 1000 --lr 0.001
```

### 2. Generate Text

```bash
# Use best checkpoint
python main.py generate --num-samples 10

# With custom settings
python main.py generate --temperature 0.8 --top-k 40 --max-length 100
```

### 3. Interactive Chat

```bash
# Start chat session
python main.py chat

# With specific checkpoint
python main.py chat --checkpoint checkpoints/best_model.pkl
```

### 4. Evaluate Model

```bash
# Run full evaluation
python main.py eval --test-file test.txt
```

### 5. Explore Model Zoo

```bash
# List available models
python main.py zoo

# Get model details
python main.py zoo --info gpt-small
```

## 🧪 Python API

### Basic Training

```python
from microgpt import GPT, Trainer, CharTokenizer, load_data

# Load data
docs = load_data('input.txt')

# Create tokenizer
tokenizer = CharTokenizer()
tokenizer.fit(docs)

# Create model
model = GPT(
    vocab_size=tokenizer.vocab_size,
    block_size=16,
    n_layer=2,
    n_embd=32,
    n_head=4
)

# Train
trainer = Trainer(model)
trainer.train(docs, tokenizer.char_to_idx, tokenizer.bos_token)
```

### Advanced Generation

```python
from microgpt import GPT, GenerationConfig

# Load model
model = GPT(...)

# Configure generation
config = GenerationConfig(
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    max_length=100
)

# Generate
tokens = model.generate(
    token_id=tokenizer.bos_token,
    max_length=config.max_length,
    temperature=config.temperature,
    top_k=config.top_k,
    top_p=config.top_p
)

text = tokenizer.decode(tokens)
print(text)
```

### Using Model Zoo

```python
from model_zoo import create_model

# Create pre-configured model
model = create_model('gpt-small')  # or 'gpt-medium', 'gpt-large'

# Train or use directly
```

### Quantization

```python
from quantization import QuantizedGPT, quantize_model

# Quantize existing model
qmodel = quantize_model(model, bits=8)

# Or create new quantized model
qmodel = QuantizedGPT(
    vocab_size=256,
    block_size=16,
    n_layer=2,
    n_embd=32,
    bits=8
)
```

### LoRA Fine-tuning

```python
from memory_efficient import LoRA

# Create LoRA adapters
lora_adapters = {
    name: LoRA(len(matrix[0]), len(matrix), rank=8)
    for name, matrix in model.state_dict.items()
    if 'wte' not in name and 'wpe' not in name
}

# Train only LoRA parameters
```

### Model Merging

```python
from model_merging import merge_with_ties, merge_with_dare, create_model_soup

# TIES merging
merged = merge_with_ties(base_model, [model1, model2, model3])

# DARE merging
merged = merge_with_dare(base_model, [model1, model2], drop_rate=0.5)

# Model soup
merged = create_model_soup([model1, model2, model3])
```

### Safety Alignment

```python
from safety_alignment import RLHFTrainer, DPOTrainer

# RLHF
rlhf = RLHFTrainer(policy_model, reward_model)
rlhf.ppo_step(...)

# DPO
dpo = DPOTrainer(model)
loss = dpo.dpo_loss(preferred_seq, rejected_seq)
```

### Advanced Reasoning

```python
from reasoning import ChainOfThought, TreeOfThought, ReAct

# Chain-of-Thought
cot = ChainOfThought(model)
result = cot.generate_with_cot("What is 23 * 47?")

# Tree-of-Thought
tot = TreeOfThought(model)
result = tot.search("Solve this puzzle")

# ReAct
tools = {'search': search_func, 'calculator': calc_func}
react = ReAct(model, tools)
result = react.run("What is the weather in Paris?")
```

### Multi-Agent Systems

```python
from agents import create_agent_system

# Create multi-agent system
system = create_agent_system(model, num_agents=3)

# Delegate task
result = system.delegate("Analyze this data", agent_name="Agent-1")

# Collaborative solving
result = system.collaborate("Complex task", 
                            agent_names=["Agent-1", "Agent-2"])
```

### Profiling

```python
from profiling import profile_model

# Full analysis
profile_model(model, operation="all")

# Just analysis
profile_model(model, operation="analyze")

# Speed benchmark
profile_model(model, operation="speed")
```

## 🎯 Common Workflows

### Workflow 1: Train from Scratch

```bash
# 1. Prepare data
echo "Your training data here" > input.txt

# 2. Train
python main.py train --epochs 1000

# 3. Generate
python main.py generate --num-samples 5
```

### Workflow 2: Fine-tune with LoRA

```python
from microgpt import GPT, load_data
from memory_efficient import LoRA
from data import CharTokenizer

# Load base model
base_model = GPT(vocab_size=256, block_size=16, n_layer=4, n_embd=64)

# Load fine-tuning data
docs = load_data('fine_tune_data.txt')
tokenizer = CharTokenizer()
tokenizer.fit(docs)

# Add LoRA
lora_adapters = {}
for name, matrix in base_model.state_dict.items():
    if 'wte' not in name and 'wpe' not in name:
        lora_adapters[name] = LoRA(len(matrix[0]), len(matrix), rank=8)

# Train only LoRA parameters
# (Implementation details in memory_efficient.py)
```

### Workflow 3: Merge Models

```python
from model_merging import merge_with_ties, TIESMerging

# Load models
base = GPT(...)
model1 = GPT(...)  # Fine-tuned on task 1
model2 = GPT(...)  # Fine-tuned on task 2

# Merge
merged = TIESMerging.merge(base, [model1, model2])

# Save
from checkpoint import CheckpointManager
cm = CheckpointManager()
cm.save_pickle(merged.state_dict, config, 0, 0.0)
```

### Workflow 4: Deploy API

```bash
# Start API server
python api_server.py

# Or with custom settings
python api_server.py --host 0.0.0.0 --port 8080
```

Then use:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 50}'
```

### Workflow 5: Web Interface

```bash
# Start web app
python web_app.py

# Open browser at http://localhost:5000
```

## 🔧 Configuration

### YAML Config Example

```yaml
# config.yaml
model:
  n_layer: 4
  n_embd: 64
  n_head: 4
  block_size: 32
  dropout: 0.1
  use_gelu: true
  use_layernorm: false

training:
  num_steps: 5000
  batch_size: 1
  learning_rate: 0.001
  lr_schedule: cosine
  warmup_steps: 100
  weight_decay: 0.01
  grad_clip: 1.0

generation:
  temperature: 0.7
  top_k: 40
  top_p: 0.9
  max_length: 100

data_path: "input.txt"
checkpoint_dir: "checkpoints"
log_dir: "logs"
```

## 📊 Monitoring Training

Training automatically logs to:
- Console: Real-time loss and perplexity
- File: `logs/run_YYYYMMDD_HHMMSS.jsonl`
- Checkpoints: `checkpoints/checkpoint_step_N.pkl`

View logs:
```bash
# Follow training
tail -f logs/latest.jsonl

# Plot training curves
python -c "from visualize import plot_training_curves; plot_training_curves('logs/latest.jsonl')"
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_model.py

# With coverage
python -m pytest --cov=microgpt tests/
```

## 🐳 Docker

```bash
# Build
docker build -t microgpt .

# Run training
docker run -v $(pwd)/data:/app/data microgpt python main.py train

# Run API
docker run -p 5000:5000 microgpt python api_server.py
```

## 📚 Next Steps

1. **Explore Examples**: Check `examples/` directory
2. **Read Full Guide**: See `docs/GUIDE.md`
3. **Check Architecture**: See `ECOSYSTEM.md`
4. **Review API**: All modules have docstrings

## 💡 Tips

- **Small models train faster**: Start with `n_layer=2, n_embd=32`
- **Use LoRA for fine-tuning**: Much faster than full fine-tuning
- **Enable gradient checkpointing**: For large models
- **Use quantization**: For deployment (4x memory reduction)
- **Profile first**: Use `profiling.py` to find bottlenecks

## 🆘 Troubleshooting

**Out of memory?**
- Reduce `n_embd` or `n_layer`
- Enable gradient checkpointing
- Use LoRA instead of full training
- Try quantization

**Training too slow?**
- Reduce model size
- Use linear LR schedule instead of cosine
- Profile with `profiling.py`

**Poor generation quality?**
- Train longer
- Increase model size
- Use better data
- Try different sampling (temperature, top-k, top-p)

## 🎓 Learning Resources

- Original microgpt: `microgpt.py` (read this first!)
- Architecture deep-dive: `docs/GUIDE.md`
- API reference: Module docstrings
- Examples: `examples/` directory

---

**Happy training! 🚀**
