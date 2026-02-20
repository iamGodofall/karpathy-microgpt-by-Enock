"""
Test training components.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from microgpt.model import GPT, Value
from microgpt.trainer import AdamOptimizer, TrainingConfig, Trainer, LRScheduler, EarlyStopping


class TestAdamOptimizer(unittest.TestCase):
    """Test Adam optimizer."""

    def setUp(self):
        self.params = [Value(1.0), Value(2.0), Value(3.0)]
        self.config = TrainingConfig()
        self.optimizer = AdamOptimizer(self.params, self.config)

    def test_initialization(self):
        self.assertEqual(len(self.optimizer.m), len(self.params))
        self.assertEqual(len(self.optimizer.v), len(self.params))
        self.assertEqual(self.optimizer.t, 0)

    def test_zero_grad(self):
        # Set some gradients
        for p in self.params:
            p.grad = 1.0

        self.optimizer.zero_grad()

        for p in self.params:
            self.assertEqual(p.grad, 0)

    def test_step(self):
        # Set gradients
        for p in self.params:
            p.grad = 0.1

        initial_data = [p.data for p in self.params]

        self.optimizer.step(0)

        # Parameters should have changed
        for i, p in enumerate(self.params):
            self.assertNotEqual(p.data, initial_data[i])

    def test_gradient_clipping(self):
        config = TrainingConfig(grad_clip=1.0)
        optimizer = AdamOptimizer(self.params, config)

        # Set large gradients
        for p in self.params:
            p.grad = 100.0

        optimizer._clip_gradients()

        # Check that gradients are clipped
        total_norm = sum(p.grad**2 for p in self.params) ** 0.5
        self.assertLessEqual(total_norm, 1.0 + 1e-6)


class TestLRScheduler(unittest.TestCase):
    """Test learning rate schedulers."""

    def test_linear_schedule(self):
        schedule = LRScheduler.linear(0.01, 100)

        lr_start = schedule(0)
        lr_mid = schedule(50)
        lr_end = schedule(99)

        self.assertAlmostEqual(lr_start, 0.01, places=5)
        self.assertAlmostEqual(lr_mid, 0.005, places=5)
        self.assertAlmostEqual(lr_end, 0.0001, places=5)

    def test_cosine_schedule(self):
        schedule = LRScheduler.cosine(0.01, 100, min_lr=0.001)

        lr_start = schedule(0)
        lr_mid = schedule(50)
        lr_end = schedule(99)

        self.assertAlmostEqual(lr_start, 0.01, places=5)
        self.assertLess(lr_mid, 0.01)
        self.assertGreater(lr_mid, 0.001)
        self.assertAlmostEqual(lr_end, 0.001, places=5)

    def test_constant_schedule(self):
        schedule = LRScheduler.constant(0.01)

        self.assertEqual(schedule(0), 0.01)
        self.assertEqual(schedule(100), 0.01)
        self.assertEqual(schedule(1000), 0.01)


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping."""

    def test_early_stop_trigger(self):
        stopper = EarlyStopping(patience=3, min_delta=0.01)

        # Improving
        self.assertFalse(stopper(1.0))
        self.assertFalse(stopper(0.9))
        self.assertFalse(stopper(0.8))

        # Plateau - need patience+1 calls to trigger (counter starts at 0)
        self.assertFalse(stopper(0.79))  # counter=1
        self.assertFalse(stopper(0.79))  # counter=2
        self.assertTrue(stopper(0.79))  # counter=3, meets patience=3

    def test_improvement_reset(self):
        stopper = EarlyStopping(patience=3)

        stopper(1.0)
        stopper(1.0)  # No improvement, counter=1
        stopper(0.5)  # Improvement, counter reset to 0
        stopper(0.5)  # No improvement, counter=1
        stopper(0.5)  # No improvement, counter=2

        self.assertFalse(stopper.should_stop)  # counter=2 < patience=3


class TestTrainer(unittest.TestCase):
    """Test training loop."""

    def setUp(self):
        self.model = GPT(vocab_size=10, block_size=8, n_layer=1, n_embd=16, n_head=4)
        self.config = TrainingConfig(num_steps=10)
        self.trainer = Trainer(self.model, self.config)

    def test_compute_loss(self):
        tokens = [0, 1, 2, 3, 4]
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        loss = self.trainer.compute_loss(tokens, keys, values)

        self.assertIsInstance(loss, Value)
        self.assertGreater(loss.data, 0)

    def test_train_step(self):
        tokens = [0, 1, 2, 3, 4]

        loss1 = self.trainer.train_step(tokens, 0)
        loss2 = self.trainer.train_step(tokens, 1)

        # Loss should generally decrease (though not guaranteed in one step)
        self.assertIsInstance(loss1, float)
        self.assertIsInstance(loss2, float)


if __name__ == "__main__":
    unittest.main()
