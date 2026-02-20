"""
Unit and integration tests for microgpt.
"""

import unittest
import random
import math
from model import Value, GPT, linear, softmax, rmsnorm, layernorm, Dropout
from config import Config, ModelConfig, TrainingConfig, GenerationConfig
from data import CharTokenizer, BPETokenizer, DataLoader

from trainer import AdamOptimizer, Trainer, LRScheduler


class TestValue(unittest.TestCase):
    """Test the Value class and autograd."""

    def setUp(self):
        random.seed(42)

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

    def test_power(self):
        a = Value(2.0)
        b = a**3
        self.assertEqual(b.data, 8.0)

        b.backward()
        self.assertEqual(a.grad, 12.0)  # 3 * 2^2

    def test_chain_rule(self):
        # Test f(x) = (x + 2) * 3
        x = Value(1.0)
        y = (x + 2) * 3
        self.assertEqual(y.data, 9.0)

        y.backward()
        self.assertEqual(x.grad, 3.0)

    def test_exp_log(self):
        a = Value(2.0)
        b = a.exp()
        self.assertAlmostEqual(b.data, math.exp(2.0), places=5)

        b.backward()
        self.assertAlmostEqual(a.grad, math.exp(2.0), places=5)

    def test_relu(self):
        a = Value(-1.0)
        b = a.relu()
        self.assertEqual(b.data, 0.0)

        a2 = Value(1.0)
        b2 = a2.relu()
        self.assertEqual(b2.data, 1.0)

        b2.backward()
        self.assertEqual(a2.grad, 1.0)


class TestLayers(unittest.TestCase):
    """Test layer functions."""

    def test_linear(self):
        # 2 inputs, 3 outputs
        x = [Value(1.0), Value(2.0)]
        w = [[Value(0.1), Value(0.2)], [Value(0.3), Value(0.4)], [Value(0.5), Value(0.6)]]

        out = linear(x, w)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0].data, 0.5)  # 1*0.1 + 2*0.2
        self.assertAlmostEqual(out[1].data, 1.1)  # 1*0.3 + 2*0.4
        self.assertAlmostEqual(out[2].data, 1.7)  # 1*0.5 + 2*0.6

    def test_softmax(self):
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        probs = softmax(logits)

        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(p.data for p in probs), 1.0, places=5)
        self.assertTrue(probs[2].data > probs[1].data > probs[0].data)

    def test_rmsnorm(self):
        x = [Value(1.0), Value(2.0), Value(3.0)]
        out = rmsnorm(x)

        # Check that RMS is approximately 1
        ms = sum(o.data**2 for o in out) / len(out)
        self.assertAlmostEqual(ms, 1.0, places=5)

    def test_layernorm(self):
        x = [Value(1.0), Value(2.0), Value(3.0)]
        out = layernorm(x)

        # Check mean is 0 and variance is 1
        mean = sum(o.data for o in out) / len(out)
        self.assertAlmostEqual(mean, 0.0, places=5)

        var = sum((o.data - mean) ** 2 for o in out) / len(out)
        self.assertAlmostEqual(var, 1.0, places=3)

    def test_dropout_training(self):
        drop = Dropout(p=0.5)
        drop.training = True

        x = [Value(1.0) for _ in range(100)]
        out = drop(x)

        # Check that some values are zero and others are scaled
        zeros = sum(1 for v in out if v.data == 0)
        scaled = sum(1 for v in out if abs(v.data - 2.0) < 0.001)

        self.assertGreater(zeros, 0)
        self.assertGreater(scaled, 0)
        self.assertEqual(zeros + scaled, 100)

    def test_dropout_eval(self):
        drop = Dropout(p=0.5)
        drop.training = False

        x = [Value(1.0) for _ in range(100)]
        out = drop(x)

        # All values should be unchanged
        for v in out:
            self.assertEqual(v.data, 1.0)


class TestGPT(unittest.TestCase):
    """Test the GPT model."""

    def setUp(self):
        random.seed(42)
        self.model = GPT(vocab_size=10, block_size=8, n_layer=1, n_embd=16, n_head=2, dropout=0.0)

    def test_initialization(self):
        self.assertEqual(self.model.vocab_size, 10)
        self.assertEqual(self.model.block_size, 8)
        self.assertEqual(self.model.n_layer, 1)
        self.assertEqual(self.model.n_embd, 16)
        self.assertEqual(self.model.n_head, 2)
        self.assertEqual(self.model.head_dim, 8)

    def test_parameters(self):
        params = self.model.parameters()
        self.assertGreater(len(params), 0)
        self.assertEqual(len(params), self.model.num_params())

    def test_forward(self):
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        logits = self.model.forward(0, 0, keys, values)
        self.assertEqual(len(logits), self.model.vocab_size)

        # Check that keys and values were populated
        self.assertEqual(len(keys[0]), 1)
        self.assertEqual(len(values[0]), 1)

    def test_generate(self):
        tokens = self.model.generate(token_id=0, max_length=5, temperature=1.0)

        self.assertEqual(len(tokens), 5)
        self.assertTrue(all(0 <= t < self.model.vocab_size for t in tokens))

    def test_training_mode(self):
        self.model.set_training(True)
        self.assertTrue(self.model.drop.training)

        self.model.set_training(False)
        self.assertFalse(self.model.drop.training)


class TestTokenizer(unittest.TestCase):
    """Test tokenizers."""

    def test_char_tokenizer(self):
        texts = ["hello", "world", "hello world"]
        tokenizer = CharTokenizer()
        tokenizer.fit(texts)

        # Check vocabulary
        self.assertIn("h", tokenizer.char_to_idx)
        self.assertIn("e", tokenizer.char_to_idx)
        self.assertIn("l", tokenizer.char_to_idx)
        self.assertIn("o", tokenizer.char_to_idx)
        self.assertIn("w", tokenizer.char_to_idx)
        self.assertIn("r", tokenizer.char_to_idx)
        self.assertIn("d", tokenizer.char_to_idx)
        self.assertIn(" ", tokenizer.char_to_idx)

        # Test encode/decode
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "hello")

        # Check BOS token
        self.assertEqual(tokenizer.vocab_size, len(tokenizer.char_to_idx) + 1)

    def test_bpe_tokenizer(self):
        texts = ["hello world", "hello there", "world hello"]
        tokenizer = BPETokenizer(vocab_size=260)
        tokenizer.fit(texts)

        # Test encode/decode
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "hello")


class TestOptimizer(unittest.TestCase):
    """Test optimizer and scheduler."""

    def test_adam_optimizer(self):
        params = [Value(1.0), Value(2.0), Value(3.0)]
        config = TrainingConfig(learning_rate=0.01)
        optimizer = AdamOptimizer(params, config)

        # Simulate gradients
        for p in params:
            p.grad = 0.1

        optimizer.step(0)

        # Check that parameters were updated
        self.assertNotEqual(params[0].data, 1.0)

    def test_lr_scheduler_linear(self):
        schedule = LRScheduler.linear(0.01, 100)

        lr_start = schedule(0)
        lr_mid = schedule(50)
        lr_end = schedule(100)

        self.assertAlmostEqual(lr_start, 0.01, places=5)
        self.assertAlmostEqual(lr_mid, 0.005, places=5)
        self.assertAlmostEqual(lr_end, 0.0, places=5)

    def test_lr_scheduler_cosine(self):
        schedule = LRScheduler.cosine(0.01, 100, warmup_steps=10)

        lr_warmup = schedule(5)
        lr_mid = schedule(50)
        lr_end = schedule(100)

        self.assertLess(lr_warmup, 0.01)
        self.assertGreater(lr_mid, lr_end)
        self.assertAlmostEqual(lr_end, 0.0, places=5)


class TestDataLoader(unittest.TestCase):
    """Test data loading."""

    def test_load_names(self):
        loader = DataLoader()
        train, val = loader.load_names(val_split=0.1)

        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertEqual(len(train) + len(val), len(loader.train_docs) + len(loader.val_docs))

        # Check tokenizer was fitted
        self.assertGreater(loader.tokenizer.vocab_size, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_training_loop(self):
        """Test a minimal training loop."""
        random.seed(42)

        # Small model
        model = GPT(vocab_size=10, block_size=4, n_layer=1, n_embd=8, n_head=2)
        config = TrainingConfig(num_steps=10, learning_rate=0.01)

        # Fake data
        docs = ["abc", "def", "ghi"]
        tokenizer = CharTokenizer()
        tokenizer.fit(docs)

        trainer = Trainer(model, config)

        # Train for a few steps
        initial_params = [p.data for p in model.parameters()]

        for step in range(5):
            doc = docs[step % len(docs)]
            tokens = (
                [tokenizer.bos_token]
                + [tokenizer.char_to_idx[ch] for ch in doc]
                + [tokenizer.bos_token]
            )
            loss = trainer.train_step(tokens, step)

        # Check that parameters changed
        final_params = [p.data for p in model.parameters()]
        self.assertNotEqual(initial_params, final_params)

    def test_generation_consistency(self):
        """Test that generation is deterministic with same seed."""
        random.seed(42)

        model = GPT(vocab_size=10, block_size=16, n_layer=1, n_embd=8, n_head=2)
        model.set_training(False)

        # Generate with seed
        random.seed(123)
        tokens1 = model.generate(0, max_length=5, temperature=1.0)

        random.seed(123)
        tokens2 = model.generate(0, max_length=5, temperature=1.0)

        self.assertEqual(tokens1, tokens2)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestValue))
    suite.addTests(loader.loadTestsFromTestCase(TestLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestGPT))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
