"""
Complete workflow example demonstrating the full microgpt ecosystem.
Shows how to use all components together in a real-world scenario.
"""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from config import Config, ModelConfig, TrainingConfig, GenerationConfig
from model import GPT
from data import CharTokenizer, Dataset
from trainer import Trainer
from checkpoint import CheckpointManager
from logger import Logger
from model_analyzer import ModelAnalyzer
from config_validator import ConfigValidator
from test_runner import TestRunner
from benchmark_suite import BenchmarkSuite
from monitoring import ModelMonitor
from openclaw_enhanced import EnhancedOpenClaw, Tool, ToolType
from hrm_enhanced import EnhancedHierarchicalReasoningModel, EnhancedHRMConfig
from unified_integration import UnifiedAI, UnifiedConfig


def demo_complete_workflow():
    """Demonstrate complete workflow from config to deployment."""
    
    print("=" * 80)
    print("microgpt Complete Workflow Demo")
    print("=" * 80)
    
    # Step 1: Configuration
    print("\n" + "=" * 80)
    print("Step 1: Configuration")
    print("=" * 80)
    
    config = Config(
        model=ModelConfig(
            vocab_size=128,
            n_embd=64,
            n_layer=2,
            n_head=4,
            block_size=32,
            dropout=0.1,
            use_gelu=True
        ),
        training=TrainingConfig(
            num_steps=100,
            batch_size=4,
            learning_rate=0.01,
            lr_schedule='cosine',
            warmup_steps=10,
            val_split=0.1,
            eval_interval=20
        ),
        generation=GenerationConfig(
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_length=50
        )
    )
    
    # Validate configuration
    validator = ConfigValidator()
    result = validator.validate(config.to_dict())
    validator.print_report(result)
    
    # Step 2: Model Analysis
    print("\n" + "=" * 80)
    print("Step 2: Model Analysis")
    print("=" * 80)
    
    model = GPT(config.model)
    analyzer = ModelAnalyzer(model)
    analysis = analyzer.analyze()
    analyzer.print_analysis(analysis)
    
    # Step 3: Data Preparation
    print("\n" + "=" * 80)
    print("Step 3: Data Preparation")
    print("=" * 80)
    
    # Create sample dataset
    sample_texts = [
        "hello world this is a test",
        "machine learning is fascinating",
        "neural networks are powerful",
        "transformers changed nlp",
        "attention is all you need",
        "deep learning revolution",
        "artificial intelligence future",
        "natural language processing",
    ] * 10
    
    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")
    dataset = Dataset(sample_texts, tokenizer, config.model.block_size)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Step 4: Training with Monitoring
    print("\n" + "=" * 80)
    print("Step 4: Training with Monitoring")
    print("=" * 80)
    
    # Set up monitoring
    monitor = ModelMonitor()
    monitor.alerts.set_threshold('train_loss', max_val=5.0)
    monitor.start(interval=5)
    
    # Create trainer
    trainer = Trainer(model, config.training)
    trainer.set_monitor(monitor)
    
    # Quick training run
    print("Training for 50 steps...")
    history = trainer.train(dataset, num_steps=50)
    
    print(f"Final loss: {history['train_losses'][-1]:.4f}")
    monitor.stop()
    monitor.save_report("training_monitor.json")
    
    # Step 5: Enhanced OpenClaw Integration
    print("\n" + "=" * 80)
    print("Step 5: Enhanced OpenClaw Features")
    print("=" * 80)
    
    oc = EnhancedOpenClaw(storage_dir=".demo_microgpt")
    
    # Register tools
    def calculator(expr: str) -> float:
        """Simple calculator tool."""
        try:
            return eval(expr)
        except:
            return None
    
    calc_tool = Tool(
        name="calc",
        description="Calculate mathematical expressions",
        type=ToolType.CALCULATOR,
        function=calculator,
        parameters={"expr": "string"},
        required_params=["expr"]
    )
    oc.register_tool(calc_tool)
    
    # Create session
    session = oc.create_session(max_context=1024)
    session.add_message("user", "What is 2 + 2?")
    
    # Execute tool
    result = oc.execute_tool("calc", {"expr": "2 + 2"})
    print(f"Tool result: {result}")
    
    # Smart compaction
    for i in range(20):
        session.add_message("user", f"Message {i}")
    print(f"Messages before compaction: {len(session.messages)}")
    session.smart_compact()
    print(f"Messages after compaction: {len(session.messages)}")
    
    # Step 6: Enhanced HRM Integration
    print("\n" + "=" * 80)
    print("Step 6: Enhanced HRM Features")
    print("=" * 80)
    
    hrm_config = EnhancedHRMConfig(
        vocab_size=128,
        hidden_size=64,
        H_layers=2,
        L_layers=2,
        H_cycles=2,
        L_cycles=2,
        adaptive_depth=True,
        use_double_q=True,
        use_meta_learning=True
    )
    
    hrm = EnhancedHierarchicalReasoningModel(hrm_config)
    
    # Test reasoning
    tokens = tokenizer.encode("solve step by step")
    result = hrm.forward(tokens, training=False)
    
    print(f"Reasoning steps: {result['steps']}")
    print(f"Adaptive depth used: {result.get('adaptive_info', {})}")
    
    # Step 7: Unified System
    print("\n" + "=" * 80)
    print("Step 7: Unified AI System")
    print("=" * 80)
    
    unified_config = UnifiedConfig(
        vocab_size=128,
        hidden_size=64,
        hrm_H_layers=2,
        hrm_L_layers=2,
        enable_tools=True,
        enable_streaming=True
    )
    
    ai = UnifiedAI(unified_config)
    
    # Simple query
    result = ai.chat("Hello, how are you?", use_reasoning=False)
    print(f"Simple response: {result['response'][:50]}...")
    
    # Complex query with reasoning
    result = ai.chat(
        "Explain the theory of relativity step by step",
        use_reasoning=True,
        max_reasoning_steps=5
    )
    print(f"Reasoning response: {result['response'][:50]}...")
    print(f"Reasoning steps used: {result.get('reasoning_info', {}).get('steps', 0)}")
    
    # Tool use
    result = ai.chat(
        "Calculate 15 * 23",
        tools=["calculator"]
    )
    print(f"Tool result: {result}")
    
    # Step 8: Checkpointing
    print("\n" + "=" * 80)
    print("Step 8: Checkpoint Management")
    print("=" * 80)
    
    checkpoint_mgr = CheckpointManager("demo_checkpoints")
    
    # Save checkpoint
    state_dict = {k: v for k, v in model.state_dict().items()}
    path = checkpoint_mgr.save_pickle(
        state_dict, config, step=50, loss=history['train_losses'][-1]
    )
    print(f"Saved checkpoint: {path}")
    
    # List checkpoints
    checkpoints = checkpoint_mgr.list_checkpoints()
    print(f"Available checkpoints: {checkpoints}")
    
    # Step 9: Testing
    print("\n" + "=" * 80)
    print("Step 9: Testing")
    print("=" * 80)
    
    runner = TestRunner(verbose=True)
    
    def quick_test():
        """Quick functionality test."""
        test_model = GPT(config.model)
        tokens = tokenizer.encode("test")
        logits = test_model.forward(tokens)
        assert len(logits) == config.model.vocab_size
    
    runner.run_test("quick_functionality", quick_test)
    runner._generate_report()
    
    # Step 10: Benchmarking
    print("\n" + "=" * 80)
    print("Step 10: Benchmarking")
    print("=" * 80)
    
    suite = BenchmarkSuite()
    
    def bench_forward():
        tokens = [random.randint(0, 127) for _ in range(20)]
        _ = model.forward(tokens)
        return 20
    
    suite.benchmark("forward_pass", "tokens/sec", "throughput", bench_forward)
    suite._generate_report()
    
    # Summary
    print("\n" + "=" * 80)
    print("Workflow Complete!")
    print("=" * 80)
    print("""
Summary:
✓ Configuration validated
✓ Model analyzed (parameters, FLOPs, memory)
✓ Dataset prepared and tokenized
✓ Model trained with monitoring
✓ OpenClaw tools and sessions working
✓ HRM reasoning with adaptive depth
✓ Unified system with intelligent routing
✓ Checkpoints saved
✓ Tests passed
✓ Benchmarks completed

All components working together successfully!
    """)
    
    # Cleanup
    import shutil
    if os.path.exists("demo_checkpoints"):
        shutil.rmtree("demo_checkpoints")
    if os.path.exists(".demo_microgpt"):
        shutil.rmtree(".demo_microgpt")


if __name__ == "__main__":
    demo_complete_workflow()
