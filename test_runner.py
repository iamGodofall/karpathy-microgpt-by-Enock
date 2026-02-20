"""
Comprehensive test runner for microgpt ecosystem.
Runs all tests with reporting and coverage analysis.
"""

import sys
import time
import traceback
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    error: str = ""
    output: str = ""


class TestRunner:
    """Run and report on all tests."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.start_time = None
    
    def run_test(self, name: str, test_func: Callable) -> TestResult:
        """Run a single test and record result."""
        if self.verbose:
            print(f"  Running {name}...", end=" ")
        
        start = time.time()
        try:
            test_func()
            duration = time.time() - start
            result = TestResult(name, True, duration)
            if self.verbose:
                print(f"✓ ({duration:.3f}s)")
        except Exception as e:
            duration = time.time() - start
            result = TestResult(name, False, duration, error=str(e), output=traceback.format_exc())
            if self.verbose:
                print(f"✗ ({duration:.3f}s)")
                print(f"    Error: {e}")
        
        self.results.append(result)
        return result
    
    def run_all(self) -> Dict[str, Any]:
        """Run all tests and generate report."""
        self.start_time = time.time()
        
        print("=" * 70)
        print("microgpt Ecosystem Test Suite")
        print("=" * 70)
        
        # Import and run test modules
        self._run_core_tests()
        self._run_integration_tests()
        self._run_enhanced_tests()
        
        # Generate report
        return self._generate_report()
    
    def _run_core_tests(self):
        """Run core functionality tests."""
        print("\n--- Core Tests ---")
        
        # Test 1: Value class
        def test_value():
            from microgpt import Value
            a = Value(2.0)
            b = Value(3.0)
            c = a * b
            assert c.data == 6.0
            c.backward()
            assert a.grad == 3.0
            assert b.grad == 2.0
        
        self.run_test("Value autograd", test_value)
        
        # Test 2: Softmax
        def test_softmax():
            from microgpt import Value, softmax
            logits = [Value(1.0), Value(2.0), Value(3.0)]
            probs = softmax(logits)
            assert abs(sum(p.data for p in probs) - 1.0) < 0.001
        
        self.run_test("Softmax", test_softmax)
        
        # Test 3: RMSNorm
        def test_rmsnorm():
            from microgpt import rmsnorm, Value
            x = [Value(1.0), Value(2.0), Value(3.0)]
            y = rmsnorm(x)
            # RMSNorm should normalize
            assert all(yi.data != 0 for yi in y)
        
        self.run_test("RMSNorm", test_rmsnorm)
    
    def _run_integration_tests(self):
        """Run integration tests."""
        print("\n--- Integration Tests ---")
        
        # Test 4: Model creation
        def test_model_creation():
            from model import GPT, GPTConfig
            config = GPTConfig(vocab_size=100, n_embd=32, n_layer=2)
            model = GPT(config)
            assert model is not None
        
        self.run_test("Model creation", test_model_creation)
        
        # Test 5: Training step
        def test_training_step():
            from model import GPT, GPTConfig
            from data import CharTokenizer
            
            config = GPTConfig(vocab_size=128, n_embd=32, n_layer=1, block_size=8)
            model = GPT(config)
            tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
            
            tokens = tokenizer.encode("hello")
            targets = tokenizer.encode("world")
            
            loss, info = model.train_step(tokens, targets)
            assert loss.data > 0
        
        self.run_test("Training step", test_training_step)
        
        # Test 6: Generation
        def test_generation():
            from model import GPT, GPTConfig
            from data import CharTokenizer
            
            config = GPTConfig(vocab_size=128, n_embd=32, n_layer=1)
            model = GPT(config)
            tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
            
            result = model.generate("h", tokenizer, max_length=5)
            assert len(result) > 0
        
        self.run_test("Generation", test_generation)
    
    def _run_enhanced_tests(self):
        """Run enhanced integration tests."""
        print("\n--- Enhanced Integration Tests ---")
        
        # Test 7: OpenClaw session
        def test_openclaw_session():
            from openclaw_enhanced import EnhancedOpenClaw
            
            oc = EnhancedOpenClaw(storage_dir=".test_microgpt")
            session = oc.create_session(max_context=1024)
            session.add_message("user", "hello")
            assert len(session.messages) == 1
        
        self.run_test("OpenClaw session", test_openclaw_session)
        
        # Test 8: HRM forward pass
        def test_hrm_forward():
            from hrm_enhanced import EnhancedHierarchicalReasoningModel, EnhancedHRMConfig
            
            config = EnhancedHRMConfig(
                vocab_size=100,
                hidden_size=32,
                H_layers=1,
                L_layers=1,
                H_cycles=2,
                L_cycles=2
            )
            hrm = EnhancedHierarchicalReasoningModel(config)
            
            tokens = [1, 2, 3]
            result = hrm.forward(tokens, training=False)
            
            assert "logits" in result
            assert "steps" in result
        
        self.run_test("HRM forward", test_hrm_forward)
        
        # Test 9: Unified AI
        def test_unified():
            from unified_integration import UnifiedAI, UnifiedConfig
            
            config = UnifiedConfig(
                vocab_size=100,
                hidden_size=32,
                hrm_H_layers=1,
                hrm_L_layers=1
            )
            ai = UnifiedAI(config)
            
            result = ai.chat("hello", use_reasoning=False)
            assert "response" in result
        
        self.run_test("Unified AI", test_unified)
        
        # Test 10: Tool execution
        def test_tools():
            from openclaw_enhanced import EnhancedOpenClaw, Tool, ToolType
            
            oc = EnhancedOpenClaw(storage_dir=".test_microgpt")
            
            def calc(expr):
                return eval(expr)
            
            tool = Tool(
                name="calc",
                description="Calculate",
                type=ToolType.CALCULATOR,
                function=calc,
                parameters={"expr": "string"},
                required_params=["expr"]
            )
            oc.register_tool(tool)
            
            result = oc.execute_tool("calc", {"expr": "2+2"})
            assert result["success"]
            assert result["result"] == 4
        
        self.run_test("Tool execution", test_tools)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate test report."""
        total_time = time.time() - self.start_time
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Success rate: {100*passed/len(self.results):.1f}%")
        print(f"Total time: {total_time:.3f}s")
        print("=" * 70)
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(self.results),
            "duration": total_time,
            "results": self.results,
        }


def run_all_tests():
    """Main entry point."""
    runner = TestRunner(verbose=True)
    return runner.run_all()


if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result["failed"] == 0 else 1)
