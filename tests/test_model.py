"""
Comprehensive test suite for microgpt model components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from model import Value, GPT, softmax, rmsnorm, layernorm


class TestValue(unittest.TestCase):
    """Test the Value class and autograd."""
    
    def test_value_creation(self):
        v = Value(5.0)
        self.assertEqual(v.data, 5.0)
        self.assertEqual(v.grad, 0)
    
    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)
    
    def test_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
        
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)
    
    def test_chain_rule(self):
        a = Value(2.0)
        b = a * 3
        c = b + 1
        d = c * c
        
        d.backward()
        
        # d = (3a + 1)^2
        # dd/da = 2(3a + 1) * 3 = 2 * 7 * 3 = 42
        self.assertAlmostEqual(a.grad, 42.0, places=5)
    
    def test_relu(self):
        a = Value(-2.0)
        b = a.relu()
        self.assertEqual(b.data, 0.0)
        
        c = Value(3.0)
        d = c.relu()
        self.assertEqual(d.data, 3.0)
    
    def test_gelu(self):
        a = Value(0.0)
        b = a.gelu()
        # GELU(0) ≈ 0
        self.assertAlmostEqual(b.data, 0.0, places=1)


class TestGPT(unittest.TestCase):
    """Test GPT model."""
    
    def setUp(self):
        self.model = GPT(
            vocab_size=10,
            block_size=8,
            n_layer=1,
            n_embd=16,
            n_head=4
        )
    
    def test_model_creation(self):
        self.assertEqual(self.model.vocab_size, 10)
        self.assertEqual(self.model.block_size, 8)
        self.assertEqual(self.model.n_layer, 1)
        self.assertEqual(self.model.n_embd, 16)
        self.assertEqual(self.model.n_head, 4)
    
    def test_parameters(self):
        params = self.model.parameters()
        self.assertGreater(len(params), 0)
        self.assertEqual(self.model.num_params(), len(params))
    
    def test_forward_pass(self):
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]
        
        logits = self.model.forward(0, 0, keys, values)
        self.assertEqual(len(logits), self.model.vocab_size)
    
    def test_generation(self):
        tokens = self.model.generate(
            0,  # start token
            max_length=5,
            temperature=1.0
        )
        self.assertEqual(len(tokens), 5)
        self.assertTrue(all(0 <= t < self.model.vocab_size for t in tokens))
    
    def test_training_mode(self):
        self.model.set_training(True)
        self.assertTrue(self.model.drop.training)
        
        self.model.set_training(False)
        self.assertFalse(self.model.drop.training)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_softmax(self):
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        probs = softmax(logits)
        
        # Check probabilities sum to 1
        total = sum(p.data for p in probs)
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check ordering preserved
        self.assertGreater(probs[2].data, probs[1].data)
        self.assertGreater(probs[1].data, probs[0].data)
    
    def test_rmsnorm(self):
        x = [Value(1.0), Value(2.0), Value(3.0)]
        normalized = rmsnorm(x)
        
        # Check RMS is approximately 1
        rms = sum(n.data ** 2 for n in normalized) / len(normalized)
        self.assertAlmostEqual(rms, 1.0, places=5)
    
    def test_layernorm(self):
        x = [Value(1.0), Value(2.0), Value(3.0)]
        normalized = layernorm(x)
        
        # Check mean is 0
        mean = sum(n.data for n in normalized) / len(normalized)
        self.assertAlmostEqual(mean, 0.0, places=5)


class TestMultiLayer(unittest.TestCase):
    """Test multi-layer models."""
    
    def test_two_layer_model(self):
        model = GPT(
            vocab_size=10,
            block_size=8,
            n_layer=2,
            n_embd=16,
            n_head=4
        )
        
        keys = [[] for _ in range(model.n_layer)]
        values = [[] for _ in range(model.n_layer)]
        
        logits = model.forward(0, 0, keys, values)
        self.assertEqual(len(logits), model.vocab_size)


class TestDropout(unittest.TestCase):
    """Test dropout functionality."""
    
    def test_dropout_training(self):
        from model import Dropout
        
        drop = Dropout(p=0.5)
        drop.training = True
        
        x = [Value(1.0), Value(1.0), Value(1.0), Value(1.0)]
        y = drop(x)
        
        # Some should be 0, others scaled
        zero_count = sum(1 for v in y if v.data == 0)
        self.assertGreater(zero_count, 0)
        self.assertLess(zero_count, 4)
    
    def test_dropout_eval(self):
        from model import Dropout
        
        drop = Dropout(p=0.5)
        drop.training = False
        
        x = [Value(1.0), Value(1.0), Value(1.0), Value(1.0)]
        y = drop(x)
        
        # All should be unchanged
        self.assertTrue(all(v.data == 1.0 for v in y))


if __name__ == '__main__':
    unittest.main()
