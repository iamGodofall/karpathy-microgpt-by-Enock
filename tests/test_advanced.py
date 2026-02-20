"""
Test advanced features.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from microgpt.model import GPT
from microgpt.advanced_features import (
    GradientAccumulator,
    BeamSearchDecoder,
    RepetitionPenaltyLogitsProcessor,
)


class TestGradientAccumulator(unittest.TestCase):
    """Test gradient accumulation."""

    def test_accumulation(self):
        acc = GradientAccumulator(accumulation_steps=4)

        # Initially step_count=0, so 0 % 4 == 0, should_step returns True
        self.assertTrue(acc.should_step())

        # After first step with empty params, step_count=1
        acc.step([])
        self.assertFalse(acc.should_step())

        for _ in range(2):
            acc.step([])
            self.assertFalse(acc.should_step())

        # After 4 steps total, should_step returns True
        acc.step([])
        self.assertTrue(acc.should_step())


class TestBeamSearch(unittest.TestCase):
    """Test beam search decoding."""

    def setUp(self):
        self.model = GPT(vocab_size=10, block_size=8, n_layer=1, n_embd=16, n_head=4)
        self.model.set_training(False)

    def test_beam_search(self):
        decoder = BeamSearchDecoder(beam_width=3, max_length=5)

        tokens, score = decoder.decode(self.model, 0)

        # Returns tokens (actual length depends on generation)
        # Just verify we get tokens and a score
        self.assertGreater(len(tokens), 0)
        self.assertIsInstance(score, float)

    def test_different_beam_widths(self):
        for width in [1, 3, 5]:
            decoder = BeamSearchDecoder(beam_width=width, max_length=3)
            tokens, score = decoder.decode(self.model, 0)
            # Verify we get tokens (actual length depends on generation)
            self.assertGreater(len(tokens), 0)


class TestRepetitionPenalty(unittest.TestCase):
    """Test repetition penalty."""

    def test_penalty_application(self):
        processor = RepetitionPenaltyLogitsProcessor(penalty=1.2)

        # Create dummy logits as Value objects
        from microgpt.model import Value

        logits = [Value(1.0), Value(2.0), Value(3.0), Value(4.0), Value(5.0)]
        input_ids = [2, 4]  # These should be penalized

        penalized = processor(logits, input_ids)

        # Penalized logits should be lower (compare data values)
        self.assertLess(penalized[2].data, logits[2].data)
        self.assertLess(penalized[4].data, logits[4].data)

        # Non-penalized should be unchanged
        self.assertEqual(penalized[0].data, logits[0].data)
        self.assertEqual(penalized[1].data, logits[1].data)


class TestQuantization(unittest.TestCase):
    """Test quantization features."""

    def test_quantization_config(self):
        from microgpt.quantization import QuantizationConfig

        config = QuantizationConfig(bits=8, symmetric=True)
        self.assertEqual(config.bits, 8)
        self.assertTrue(config.symmetric)

    def test_quantize_tensor(self):
        from microgpt.quantization import Quantizer, QuantizationConfig

        config = QuantizationConfig(bits=8)
        quantizer = Quantizer(config)

        # Create dummy data
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        quantized = quantizer.quantize_tensor(data)

        # Should have quantized data and scale attributes
        self.assertTrue(hasattr(quantized, "data"))
        self.assertTrue(hasattr(quantized, "scale"))
        self.assertTrue(hasattr(quantized, "zero_point"))


class TestCheckpoint(unittest.TestCase):
    """Test checkpoint management."""

    def test_checkpoint_manager(self):
        from microgpt.checkpoint import CheckpointManager

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Create dummy checkpoint
            state_dict = {"w": [[1.0, 2.0], [3.0, 4.0]]}
            config = {"n_layer": 1}

            # Save
            path = manager.save_json(state_dict, config, 100, 1.5)
            self.assertTrue(os.path.exists(path))

            # Load
            loaded = manager.load_json(os.path.basename(path))
            self.assertEqual(loaded["step"], 100)
            self.assertEqual(loaded["loss"], 1.5)


class TestExport(unittest.TestCase):
    """Test model export."""

    def setUp(self):
        self.model = GPT(vocab_size=10, block_size=8, n_layer=1, n_embd=16, n_head=4)

    def test_json_export(self):
        from microgpt.export import ModelExporter

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(self.model)
            path = os.path.join(tmpdir, "model.json")
            exporter.to_json(path)

            self.assertTrue(os.path.exists(path))

    def test_pickle_export(self):
        from microgpt.export import ModelExporter

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(self.model)
            path = os.path.join(tmpdir, "model.pkl")
            exporter.to_pickle(path)

            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
