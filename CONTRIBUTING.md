# Contributing to microgpt

Thank you for your interest in contributing to microgpt! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

- **Report bugs** - Open an issue with bug details
- **Suggest features** - Propose new capabilities
- **Write code** - Implement features or fix bugs
- **Improve docs** - Enhance documentation and examples
- **Share examples** - Add use cases and tutorials

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/karpathy-microgpt-by-Enock.git
   cd karpathy-microgpt-by-Enock
   ```

3. **Set up development environment**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make changes and test**
   ```bash
   python -m pytest tests/
   python integration_test.py
   ```

6. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: description of changes"
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**

## 📋 Development Guidelines

### Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions/classes
- Keep functions focused and small
- Add comments for complex logic

```python
def example_function(param: int) -> str:
    """
    Brief description.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
    """
    return str(param)
```

### Testing

- Write tests for new features
- Maintain >80% code coverage
- Run full test suite before submitting
- Include integration tests for major features

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest --cov=microgpt tests/

# Integration tests
python integration_test.py
```

### Documentation

- Update README.md if adding major features
- Add to docs/GUIDE.md for detailed explanations
- Include examples in examples/ directory
- Update QUICKSTART.md for user-facing changes

### Commit Messages

Use conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```
feat: Add speculative decoding for faster inference

- Implement draft model for token prediction
- Add verification mechanism
- 2-3x speedup on generation
```

## 🏗️ Project Structure

```
microgpt/
├── Core
│   ├── model.py           # GPT architecture
│   ├── trainer.py         # Training loop
│   ├── data.py            # Data loading
│   └── config.py          # Configuration
├── Architecture
│   ├── modern_architecture.py    # RoPE, SwiGLU, etc.
│   └── state_of_the_art.py       # Mamba, Griffin, etc.
├── Training
│   ├── advanced_training.py      # Lion, Sophia, etc.
│   └── memory_efficient.py       # LoRA, QLoRA, etc.
├── Inference
│   ├── inference_optimizations.py # PagedAttention, etc.
│   └── advanced_features.py       # Beam search, etc.
├── Safety
│   └── safety_alignment.py       # RLHF, DPO, etc.
├── Multimodal
│   └── multimodal.py             # Vision, audio, tools
├── Merging
│   └── model_merging.py          # TIES, DARE, etc.
├── Evaluation
│   └── evaluation.py             # Metrics, benchmarks
├── Reasoning
│   └── reasoning.py              # CoT, ToT, ReAct
├── Agents
│   └── agents.py                 # Multi-agent systems
├── Utils
│   ├── profiling.py              # Performance analysis
│   ├── compression.py            # Pruning, distillation
│   ├── checkpoint.py             # Save/load
│   └── logger.py                 # Metrics
├── Interfaces
│   ├── main.py                   # CLI entry point
│   ├── cli.py                    # CLI commands
│   ├── web_app.py                # Web UI
│   ├── api_server.py             # REST API
│   └── chat.py                   # Interactive chat
├── Examples
│   └── examples/                 # Usage examples
└── Tests
    └── tests/                    # Test suite
```

## 🔬 Adding New Features

### New Architecture Component

1. Create module in appropriate file
2. Follow existing patterns (see `modern_architecture.py`)
3. Add to `__init__.py` exports
4. Write tests in `tests/`
5. Add example in `examples/`
6. Update documentation

Example:
```python
# In modern_architecture.py
class NewAttention:
    """Your new attention mechanism."""
    
    def __init__(self, dim: int):
        self.dim = dim
        # Initialize parameters
    
    def forward(self, x: List[Value]) -> List[Value]:
        # Implementation
        return x
```

### New Training Method

1. Add to `advanced_training.py` or `memory_efficient.py`
2. Implement as class with clear interface
3. Add configuration options to `config.py`
4. Write tests and examples

### New Tokenizer

1. Add to `tokenizers.py`
2. Inherit from base `Tokenizer` class
3. Implement `encode()` and `decode()`
4. Add to `create_tokenizer()` factory

## 🧪 Testing Guidelines

### Unit Tests

Test individual components:
```python
def test_new_feature():
    from microgpt import NewFeature
    
    feature = NewFeature(param=10)
    result = feature.process([1, 2, 3])
    
    assert len(result) == 3
    assert result[0] > 0
```

### Integration Tests

Add to `integration_test.py`:
```python
def test_new_feature_integration():
    """Test new feature end-to-end."""
    print("\nTEST: New Feature")
    
    # Setup
    model = GPT(...)
    
    # Test
    result = new_feature(model)
    
    # Verify
    assert result is not None
    print("✅ New feature test PASSED")
    return True
```

## 📊 Performance Considerations

- Profile before optimizing
- Use `profiling.py` for analysis
- Document performance characteristics
- Consider memory vs speed tradeoffs
- Add benchmarks for new features

## 🌍 Compatibility

- Support Python 3.8+
- Maintain pure Python (no heavy dependencies)
- Keep core dependencies minimal
- Optional dependencies for advanced features
- Document version requirements

## 📝 Documentation Standards

### Docstrings

Use Google-style docstrings:
```python
def function(arg1: int, arg2: str) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of first argument
        arg2: Description of second argument
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input provided
        
    Example:
        >>> function(1, "test")
        True
    """
    return True
```

### README Updates

When adding features, update:
- Feature list in README.md
- Code examples if relevant
- Installation if new dependencies
- Quickstart if user-facing

## 🎨 Design Principles

1. **Simplicity First** - Keep core implementation simple
2. **Modularity** - Components should be composable
3. **Extensibility** - Easy to add new features
4. **Education** - Code should be readable and educational
5. **Performance** - Optimize after correctness
6. **Testing** - Everything should be testable

## 🐛 Reporting Bugs

Include in issue:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages
- Minimal code example

## 💡 Feature Requests

Include in issue:
- Use case description
- Proposed API/interface
- Example usage
- Potential implementation approach
- Willingness to contribute

## 🔒 Security

- Report security issues privately
- Don't expose vulnerabilities in public issues
- Follow responsible disclosure
- Security fixes get priority

## 🏅 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## 📞 Getting Help

- Open an issue for questions
- Join discussions
- Check existing documentation
- Review examples

## 🎯 Priority Areas

High priority for contributions:
1. Performance optimizations
2. Additional architecture variants
3. More comprehensive tests
4. Better documentation
5. Real-world examples
6. Bug fixes

## 🚀 Release Process

1. Update version in `__init__.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create git tag
5. Build and upload to PyPI
6. Create GitHub release

---

Thank you for contributing to microgpt! 🎉
