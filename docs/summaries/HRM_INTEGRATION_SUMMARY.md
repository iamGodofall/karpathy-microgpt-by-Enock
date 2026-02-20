# Hierarchical Reasoning Model (HRM) Integration

This document describes the integration of the Hierarchical Reasoning Model (HRM) architecture into the microgpt ecosystem.

## Overview

**Paper**: "Hierarchical Reasoning Model" (Wang et al., 2025)  
**arXiv**: https://arxiv.org/abs/2506.21734

HRM is a novel recurrent architecture that achieves significant computational depth while maintaining training stability and efficiency. It executes sequential reasoning tasks in a single forward pass without explicit supervision of intermediate steps.

## Architecture

### Core Components

```
┌─────────────────────────────────────────┐
│         Hierarchical Reasoning Model     │
├─────────────────────────────────────────┤
│  High-Level Module (H)                  │
│  ├── Slow, abstract planning            │
│  └── H_cycles iterations                │
│                                         │
│  Low-Level Module (L)                   │
│  ├── Fast, detailed computation         │
│  └── L_cycles per H cycle               │
│                                         │
│  Adaptive Computation Time (ACT)         │
│  ├── Q-learning based halting           │
│  └── Dynamic step count                 │
└─────────────────────────────────────────┘
```

### Key Features

1. **Dual-Timescale Processing**
   - High-level module: Coarse-grained planning (slow)
   - Low-level module: Fine-grained execution (fast)
   - Interleaved computation: H guides L, L updates H

2. **Adaptive Computation Time (ACT)**
   - Q-learning for halting decisions
   - Exploration during training
   - Dynamic computation based on complexity

3. **No Chain-of-Thought Required**
   - Implicit reasoning through recurrence
   - Single forward pass
   - No intermediate supervision needed

## Files

| File | Description |
|------|-------------|
| `hrm_adapter.py` | Core HRM implementation in pure Python |
| `microgpt_hrm_integration.py` | Integration with microgpt ecosystem |
| `examples/08_hrm_integration.py` | Usage examples and demos |

## Usage

### Basic Training

```python
from microgpt_hrm_integration import HybridGPTWithHRM, HRMIntegratedConfig, HRMTrainer

# Configuration
config = HRMIntegratedConfig(
    vocab_size=100,
    hidden_size=64,
    H_layers=2,      # High-level planning layers
    L_layers=2,      # Low-level execution layers
    H_cycles=3,      # Planning iterations
    L_cycles=3,      # Execution iterations per planning step
    halt_max_steps=8,  # Max computation steps
)

# Create model
model = HybridGPTWithHRM(config)

# Train
trainer = HRMTrainer(model, config)
history = trainer.train(dataset, num_steps=1000)
```

### Generation

```python
# Generate with HRM
result = model.generate(
    prompt="hello world",
    tokenizer=tokenizer,
    max_length=50,
    use_hrm=True
)
```

### ACT Behavior

```python
# Forward pass returns computation info
result = model.forward_hrm(tokens)
print(f"Steps used: {result['steps']}")
print(f"Halted early: {result['halted']}")
```

## Comparison with Standard GPT

| Feature | Standard GPT | HRM |
|---------|-----------|-----|
| Architecture | Flat layers | Hierarchical (H + L) |
| Computation | Fixed depth | Adaptive (ACT) |
| Reasoning | Requires CoT | Implicit in recurrence |
| Parameters | ~same | ~same |
| Training stability | Good | Excellent |

## Implementation Details

### Pure Python

- No PyTorch/TensorFlow dependencies
- Compatible with microgpt's `Value` class
- Scalar-level autograd
- Compatible with all microgpt features

### HRM Block

```python
class HRMBlock:
    def __init__(self, hidden_size, num_heads):
        # Multi-head self-attention
        self.self_attn = Attention(...)
        # SwiGLU MLP
        self.mlp = SwiGLU(...)
        # RMSNorm
        self.norm = RMSNorm(...)
```

### ACT Mechanism

```python
def should_halt(q_halt, q_continue, step):
    # Q-learning decision
    if q_halt > q_continue:
        return True
    # Max steps check
    if step >= max_steps:
        return True
    # Exploration
    if random() < exploration_prob:
        return True
    return False
```

## Performance

### Small-Sample Learning

HRM achieves strong performance with minimal data:
- 1,000 examples sufficient for complex tasks
- No pre-training required
- Fast convergence

### Tasks

- **Sudoku**: Solves 9×9 extreme puzzles
- **Mazes**: Optimal path finding in 30×30 grids
- **ARC**: Abstraction and Reasoning Corpus
- **Pattern Learning**: a^n b^n, bracket matching

## Integration with microgpt Ecosystem

### Compatible Features

- ✅ Checkpointing (save/load)
- ✅ Quantization
- ✅ Export formats
- ✅ Web interface
- ✅ API server
- ✅ Visualization

### Example: HRM + Web Interface

```python
from microgpt_hrm_integration import HybridGPTWithHRM
from web_app import create_app

model = HybridGPTWithHRM(config)
app = create_app(model)
```

## Citation

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
  title={Hierarchical Reasoning Model}, 
  author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
  year={2025},
  eprint={2506.21734},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.21734}, 
}
```

## Next Steps

1. **Benchmarking**: Compare with standard microgpt on reasoning tasks
2. **Scaling**: Test larger HRM configurations
3. **Multi-task**: Train on diverse reasoning problems
4. **Visualization**: Plot attention patterns across H/L cycles

## References

- Original paper: https://arxiv.org/abs/2506.21734
- microgpt: https://github.com/karpathy/microgpt
- OpenClaw: Session management patterns
