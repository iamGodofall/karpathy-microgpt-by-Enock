"""
Comprehensive integration test for the entire microgpt ecosystem.
Tests all major components end-to-end.
"""

import sys
import random
import tempfile
import os


def test_core_pipeline():
    """Test core training and generation pipeline."""
    print("\n" + "=" * 60)
    print("TEST 1: Core Pipeline")
    print("=" * 60)

    from model import GPT
    from trainer import Trainer, TrainingConfig
    from data import CharTokenizer
    from checkpoint import CheckpointManager

    # Create minimal data
    docs = ["hello world", "test sentence", "another example"]

    # Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(docs)
    print(f"✓ Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Model
    model = GPT(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=1, n_embd=16, n_head=2)
    print(f"✓ Model: {model.num_params()} parameters")

    # Train for a few steps
    config = TrainingConfig(num_steps=10, learning_rate=0.01)
    trainer = Trainer(model, config)

    for i in range(5):
        tokens = (
            [tokenizer.bos_token]
            + [tokenizer.char_to_idx[c] for c in docs[i % len(docs)]]
            + [tokenizer.bos_token]
        )
        loss = trainer.train_step(tokens, i)
        print(f"  Step {i+1}: loss={loss:.4f}")

    # Generate
    generated = model.generate(tokenizer.bos_token, max_length=10)
    text = tokenizer.decode(generated)
    print(f"✓ Generated: {text[:50]}...")

    # Checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir)
        path = cm.save_pickle(model.state_dict, config, 5, loss)
        print(f"✓ Checkpoint saved: {path}")

        loaded = cm.load_pickle(os.path.basename(path))
        print(f"✓ Checkpoint loaded: step={loaded['step']}")

    print("✅ Core pipeline test PASSED")
    return True


def test_advanced_features():
    """Test advanced generation features."""
    print("\n" + "=" * 60)
    print("TEST 2: Advanced Features")
    print("=" * 60)

    from model import GPT
    from advanced_features import BeamSearch, RepetitionPenalty
    from data import CharTokenizer

    model = GPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = {chr(i): i - 97 for i in range(97, 123)}
    tokenizer.idx_to_char = {v: k for k, v in tokenizer.char_to_idx.items()}
    tokenizer.vocab_size = 100
    tokenizer.bos_token = 99

    # Test beam search
    beam_search = BeamSearch(model, beam_width=3)
    result = beam_search.search(tokenizer.bos_token, max_length=5)
    print(f"✓ Beam search: generated {len(result)} tokens")

    # Test repetition penalty
    penalty = RepetitionPenalty(penalty=1.2)
    logits = [model.Value(random.random()) for _ in range(100)]
    penalized = penalty.apply(logits, [1, 2, 3])
    print(f"✓ Repetition penalty applied")

    print("✅ Advanced features test PASSED")
    return True


def test_modern_architecture():
    """Test modern architecture components."""
    print("\n" + "=" * 60)
    print("TEST 3: Modern Architecture")
    print("=" * 60)

    from modern_architecture import RoPE, SwiGLU, ModernGPT

    # Test RoPE
    rope = RoPE(dim=16)
    x = [[rope.Value(random.random()) for _ in range(16)]]
    rotated = rope.apply(x, position=5)
    print(f"✓ RoPE: rotated {len(rotated)} positions")

    # Test SwiGLU
    swiglu = SwiGLU()
    x = [swiglu.Value(random.random()) for _ in range(16)]
    out = swiglu.forward(x)
    print(f"✓ SwiGLU: output dim={len(out)}")

    # Test ModernGPT
    model = ModernGPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)
    print(f"✓ ModernGPT: {model.num_params()} parameters")

    print("✅ Modern architecture test PASSED")
    return True


def test_memory_efficiency():
    """Test memory-efficient training methods."""
    print("\n" + "=" * 60)
    print("TEST 4: Memory Efficiency")
    print("=" * 60)

    from model import GPT
    from memory_efficient import LoRA, GradientCheckpointing

    model = GPT(vocab_size=100, block_size=8, n_layer=2, n_embd=16, n_head=2)

    # Test LoRA
    lora = LoRA(in_dim=16, out_dim=16, rank=4)
    x = [lora.Value(random.random()) for _ in range(16)]
    base_out = [lora.Value(random.random()) for _ in range(16)]
    lora_out = lora.forward(x, base_out)
    print(f"✓ LoRA: rank=4, output dim={len(lora_out)}")

    # Test gradient checkpointing
    gc = GradientCheckpointing(model, checkpoint_every=1)
    print(f"✓ Gradient checkpointing: enabled")

    print("✅ Memory efficiency test PASSED")
    return True


def test_inference_optimizations():
    """Test inference optimizations."""
    print("\n" + "=" * 60)
    print("TEST 5: Inference Optimizations")
    print("=" * 60)

    from model import GPT
    from inference_optimizations import PagedAttention, QuantizedCache

    model = GPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)

    # Test PagedAttention
    paged = PagedAttention(block_size=4, num_blocks=10)
    blocks = paged.allocate(0, 8)
    print(f"✓ PagedAttention: allocated {len(blocks)} blocks")

    # Test quantized cache
    cache = QuantizedCache(bits=8)
    x = [cache.Value(random.random()) for _ in range(16)]
    quantized, scale, zp = cache.quantize(x)
    dequantized = cache.dequantize(quantized, scale, zp)
    print(f"✓ QuantizedCache: {len(quantized)} values @ 8-bit")

    print("✅ Inference optimizations test PASSED")
    return True


def test_safety_alignment():
    """Test safety and alignment features."""
    print("\n" + "=" * 60)
    print("TEST 6: Safety & Alignment")
    print("=" * 60)

    from model import GPT
    from safety_alignment import SafetyClassifier, Watermarking

    model = GPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)

    # Test safety classifier
    classifier = SafetyClassifier(model, num_labels=3)
    tokens = [1, 2, 3, 4, 5]
    scores = classifier.classify(tokens)
    print(f"✓ SafetyClassifier: {len(scores)} labels")

    # Test watermarking
    wm = Watermarking(hash_key="test_key")
    logits = [wm.Value(random.random()) for _ in range(100)]
    watermarked = wm.apply_watermark(logits, previous_token=0)
    print(f"✓ Watermarking: applied to {len(watermarked)} logits")

    z_score = wm.detect_watermark([1, 2, 3, 4, 5])
    print(f"✓ Watermark detection: z-score={z_score:.2f}")

    print("✅ Safety & alignment test PASSED")
    return True


def test_model_merging():
    """Test model merging techniques."""
    print("\n" + "=" * 60)
    print("TEST 7: Model Merging")
    print("=" * 60)

    from model import GPT
    from model_merging import TaskArithmetic, ModelSoups

    # Create base and task models
    base = GPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)
    task1 = GPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)
    task2 = GPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)

    # Modify task models slightly
    for name in task1.state_dict:
        for i in range(len(task1.state_dict[name])):
            for j in range(len(task1.state_dict[name][i])):
                task1.state_dict[name][i][j].data += 0.01
                task2.state_dict[name][i][j].data += 0.02

    # Test task arithmetic
    merged = TaskArithmetic.merge_models(base, [task1, task2], [0.5, 0.5])
    print(f"✓ Task Arithmetic: merged {len(merged.state_dict)} layers")

    # Test model soup
    soup = ModelSoups.average_models([task1, task2])
    print(f"✓ Model Soup: averaged 2 models")

    print("✅ Model merging test PASSED")
    return True


def test_reasoning():
    """Test reasoning capabilities."""
    print("\n" + "=" * 60)
    print("TEST 8: Reasoning")
    print("=" * 60)

    from model import GPT
    from reasoning import ChainOfThought, ReAct

    model = GPT(vocab_size=100, block_size=16, n_layer=1, n_embd=16, n_head=2)

    # Test CoT
    cot = ChainOfThought(model)
    result = cot.generate_with_cot("What is 2+2?", max_steps=3)
    print(f"✓ Chain-of-Thought: generated reasoning")

    # Test ReAct
    tools = {"calculator": lambda x: "4"}
    react = ReAct(model, tools)
    result = react.run("Calculate 2+2", max_steps=3)
    print(f"✓ ReAct: completed task")

    print("✅ Reasoning test PASSED")
    return True


def test_agents():
    """Test agent system."""
    print("\n" + "=" * 60)
    print("TEST 9: Agents")
    print("=" * 60)

    from model import GPT
    from agents import Agent, MultiAgentSystem, ToolLibrary

    model = GPT(vocab_size=100, block_size=8, n_layer=1, n_embd=16, n_head=2)

    # Test single agent
    agent = Agent(model, name="TestAgent")
    tools = ToolLibrary()
    for name, tool in tools.tools.items():
        agent.register_tool(name, tool)
    print(f"✓ Agent: registered {len(tools.tools)} tools")

    # Test multi-agent
    system = MultiAgentSystem()
    for i in range(3):
        a = Agent(model, name=f"Agent-{i}")
        system.add_agent(a)
    print(f"✓ Multi-Agent: created {len(system.agents)} agents")

    print("✅ Agents test PASSED")
    return True


def test_evaluation():
    """Test evaluation metrics."""
    print("\n" + "=" * 60)
    print("TEST 10: Evaluation")
    print("=" * 60)

    from evaluation import BLEU, ROUGE, DiversityMetrics

    # Test BLEU
    ref = "the cat sat on the mat"
    hyp = "the cat is on the mat"
    bleu = BLEU.compute(ref, hyp)
    print(f"✓ BLEU: {bleu:.4f}")

    # Test ROUGE
    rouge1 = ROUGE.rouge_n(ref, hyp, n=1)
    rougeL = ROUGE.rouge_l(ref, hyp)
    print(f"✓ ROUGE-1: {rouge1:.4f}, ROUGE-L: {rougeL:.4f}")

    # Test diversity
    texts = ["hello world", "hello there", "hi everyone"]
    distinct2 = DiversityMetrics.distinct_n(texts, n=2)
    print(f"✓ Distinct-2: {distinct2:.4f}")

    print("✅ Evaluation test PASSED")
    return True


def test_compression():
    """Test model compression."""
    print("\n" + "=" * 60)
    print("TEST 11: Compression")
    print("=" * 60)

    from model import GPT
    from compression import MagnitudePruning, WeightSharing

    model = GPT(vocab_size=100, block_size=8, n_layer=2, n_embd=16, n_head=2)
    original_params = model.num_params()

    # Test pruning
    pruner = MagnitudePruning(sparsity=0.3)
    pruned = pruner.prune(model)
    nonzero = pruner.count_nonzero(pruned)
    print(
        f"✓ Pruning: {original_params} -> {nonzero} params ({100*(1-nonzero/original_params):.1f}% sparsity)"
    )

    # Test weight sharing
    shared = WeightSharing.share_across_layers(model, share_every=2)
    print(f"✓ Weight sharing: applied")

    print("✅ Compression test PASSED")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("MICROGPT COMPREHENSIVE INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        test_core_pipeline,
        test_advanced_features,
        test_modern_architecture,
        test_memory_efficiency,
        test_inference_optimizations,
        test_safety_alignment,
        test_model_merging,
        test_reasoning,
        test_agents,
        test_evaluation,
        test_compression,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
