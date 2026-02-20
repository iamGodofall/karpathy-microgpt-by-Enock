# OpenClaw Architecture Integration for microgpt

## Overview

This integration brings OpenClaw's sophisticated AI assistant architecture to microgpt, combining:
- **microgpt**: Pure Python GPT implementation (Karpathy's educational code)
- **OpenClaw**: Production-grade session management, auth systems, and tool frameworks

## Architecture Comparison

| Aspect | Original microgpt | With OpenClaw Integration |
|--------|------------------|---------------------------|
| Sessions | None | Full session management with compaction |
| Auth | None | Multi-profile auth with rotation |
| Context Window | Fixed | Auto-compacting with summaries |
| Fallback | None | Model fallback chain |
| Tools | None | Extensible tool registry |
| Storage | None | Persistent session storage |

## Key Components

### 1. OpenClaw Adapter (`openclaw_adapter.py`)

Core patterns adapted from OpenClaw:

```python
# Session Management
session = adapter.create_session("user_123", max_context=2048)
session.add_message("user", "Hello!")
session.add_message("assistant", "Hi there!")

# Auth Profiles
profile = AuthProfile("openai_1", "openai", api_key="sk-...")
adapter.auth_store.add_profile(profile)

# Model Fallback
fallback = ModelFallback(
    models=["gpt-4", "gpt-3.5-turbo"],
    fallback_chain=["claude-3", "local-model"]
)
```

### 2. Integrated Model (`microgpt_openclaw_integration.py`)

Full GPT with OpenClaw features:

```python
from microgpt_openclaw_integration import MicroGPTWithOpenClaw, MicroGPTConfig

# Configure
config = MicroGPTConfig(
    n_layer=4, n_embd=128, n_head=4,
    max_context_tokens=2048,
    enable_fallback=True
)

# Create model
model = MicroGPTWithOpenClaw(config)

# Chat with session management
result = model.chat("Hello!", session_id="user_123")
print(result['response'])
print(f"Session tokens: {result['tokens_used']}")

# Train
model.train(num_steps=1000)

# Generate with advanced sampling
text = model.generate(
    prompt="Once upon a time",
    temperature=0.8,
    top_k=40,
    top_p=0.9
)
```

## OpenClaw Features Implemented

### Session Management
- **Automatic compaction**: Summarizes old messages when context grows
- **Persistent storage**: Sessions saved to `.microgpt/sessions/`
- **Context window tracking**: Monitors token usage
- **Multi-session support**: Handle multiple concurrent conversations

### Auth System
- **Profile management**: Multiple API keys per provider
- **Cooldown handling**: Automatic rotation on failures
- **Provider fallback**: Switch between providers on errors
- **Secure storage**: JSON-based profile storage

### Model Features
- **Top-k sampling**: Limit to k most likely tokens
- **Top-p (nucleus) sampling**: Dynamic vocabulary filtering
- **Temperature control**: Adjust randomness
- **Fallback chains**: Automatic model switching

## File Structure

```
microgpt/
├── microgpt.py                      # Original pure Python GPT
├── openclaw_adapter.py              # OpenClaw patterns
├── microgpt_openclaw_integration.py # Full integration
├── config.py                        # Configuration system
├── checkpoint.py                    # Model checkpointing
├── logger.py                        # Logging infrastructure
├── trainer.py                       # Training loop
├── data.py                          # Data loading
├── cli.py                           # Command-line interface
├── web_app.py                       # Flask web interface
├── api_server.py                    # REST API
├── agents.py                        # Agent system
├── model_zoo.py                     # Pre-configured models
├── tokenizers.py                    # BPE tokenizer
├── quantization.py                  # Model compression
├── export.py                        # Format export
├── examples/
│   ├── 01_basic_training.py
│   ├── 02_advanced_generation.py
│   ├── 03_model_zoo.py
│   ├── 04_quantization.py
│   ├── 05_interpretability.py
│   ├── 06_export_formats.py
│   └── 07_openclaw_integration.py  # New!
└── tests/
    ├── test_model.py
    ├── test_training.py
    └── test_advanced.py
```

## Usage Examples

### Basic Chat
```python
from microgpt_openclaw_integration import MicroGPTWithOpenClaw

model = MicroGPTWithOpenClaw()
result = model.chat("What's the weather like?", session_id="user_1")
print(result['response'])
```

### With System Prompt
```python
result = model.chat(
    "Explain quantum computing",
    session_id="user_1",
    system_prompt="You are a helpful physics expert."
)
```

### Session Management
```python
# Get session info
info = model.adapter.get_session_info("user_1")
print(f"Messages: {info['message_count']}")
print(f"Tokens: {info['estimated_tokens']}")

# Manual compaction
model.adapter.compact_session("user_1")
```

### Training
```python
config = MicroGPTConfig(num_steps=5000, learning_rate=0.001)
model = MicroGPTWithOpenClaw(config)
model.train()

# Save checkpoint
model.save_checkpoint("my_model.json")

# Load later
model.load_checkpoint("my_model.json")
```

## CLI Usage

```bash
# Train
python microgpt_openclaw_integration.py --train --steps 1000 --save model.json

# Chat
python microgpt_openclaw_integration.py --chat "Hello" --session user_123

# Generate
python microgpt_openclaw_integration.py --generate --prompt "Once upon" --temperature 0.8

# Stats
python microgpt_openclaw_integration.py --stats
```

## Design Patterns from OpenClaw

1. **Session-based conversations**: Track context across interactions
2. **Automatic compaction**: Manage context window automatically
3. **Auth profile rotation**: Handle API failures gracefully
4. **Event hooks**: Extensible architecture
5. **Persistent state**: Survive restarts
6. **Tool registry**: Pluggable capabilities

## Benefits

- **Educational**: Still pure Python, easy to understand
- **Production-ready**: Session management and error handling
- **Extensible**: Tool system and hooks
- **Persistent**: State survives restarts
- **Robust**: Fallback and retry mechanisms

## Next Steps

1. Implement actual tool calling (browser, code execution)
2. Add more sophisticated summarization
3. Implement multi-modal support
4. Add distributed training
5. Create web-based chat interface
6. Add voice/vision capabilities

## Credits

- **microgpt**: Andrej Karpathy's educational GPT implementation
- **OpenClaw**: Peter Steinberger's AI assistant framework
- **Integration**: Adapted for pure Python ecosystem
