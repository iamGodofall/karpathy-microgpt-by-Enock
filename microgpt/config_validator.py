"""
Configuration validator for microgpt ecosystem.
Validates configs, suggests improvements, checks compatibility.
"""

import json
import yaml
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class ConfigValidator:
    """Validate microgpt configurations."""

    # Valid configuration keys and their types
    MODEL_KEYS = {
        "vocab_size": int,
        "n_embd": int,
        "n_layer": int,
        "n_head": int,
        "block_size": int,
        "dropout": float,
        "use_gelu": bool,
        "use_layernorm": bool,
    }

    TRAINING_KEYS = {
        "num_steps": int,
        "batch_size": int,
        "learning_rate": float,
        "beta1": float,
        "beta2": float,
        "eps_adam": float,
        "weight_decay": float,
        "grad_clip": float,
        "lr_schedule": str,
        "warmup_steps": int,
        "val_split": float,
        "eval_interval": int,
        "save_interval": int,
    }

    GENERATION_KEYS = {
        "temperature": float,
        "top_k": int,
        "top_p": float,
        "max_length": int,
        "num_samples": int,
        "seed": int,
    }

    VALID_LR_SCHEDULES = ["linear", "cosine", "constant", "exponential"]

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary."""
        self.errors = []
        self.warnings = []
        self.suggestions = []

        # Check required sections
        if "model" not in config:
            self.errors.append("Missing 'model' section")
        if "training" not in config:
            self.errors.append("Missing 'training' section")

        # Validate model config
        if "model" in config:
            self._validate_model(config["model"])

        # Validate training config
        if "training" in config:
            self._validate_training(config["training"])

        # Validate generation config
        if "generation" in config:
            self._validate_generation(config["generation"])

        # Cross-validation
        self._cross_validate(config)

        # Generate suggestions
        self._generate_suggestions(config)

        return ValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            suggestions=self.suggestions,
        )

    def _validate_model(self, model: Dict[str, Any]):
        """Validate model configuration."""
        # Check required keys
        required = ["vocab_size", "n_embd", "n_layer", "n_head", "block_size"]
        for key in required:
            if key not in model:
                self.errors.append(f"Model missing required key: {key}")

        # Check types
        for key, expected_type in self.MODEL_KEYS.items():
            if key in model and not isinstance(model[key], expected_type):
                self.errors.append(
                    f"Model.{key} should be {expected_type.__name__}, got {type(model[key]).__name__}"
                )

        # Check values
        if "vocab_size" in model:
            if model["vocab_size"] < 2:
                self.errors.append("vocab_size must be at least 2")
            if model["vocab_size"] > 100000:
                self.warnings.append("Very large vocab_size may cause memory issues")

        if "n_embd" in model:
            if model["n_embd"] % model.get("n_head", 1) != 0:
                self.errors.append("n_embd must be divisible by n_head")
            if model["n_embd"] > 4096:
                self.warnings.append("Large n_embd may be slow in pure Python")

        if "n_layer" in model:
            if model["n_layer"] < 1:
                self.errors.append("n_layer must be at least 1")
            if model["n_layer"] > 12:
                self.warnings.append("Many layers may be slow without GPU")

        if "n_head" in model:
            if model["n_head"] < 1:
                self.errors.append("n_head must be at least 1")

        if "block_size" in model:
            if model["block_size"] < 1:
                self.errors.append("block_size must be at least 1")
            if model["block_size"] > 2048:
                self.warnings.append("Large block_size increases memory usage")

        if "dropout" in model:
            if not 0 <= model["dropout"] <= 1:
                self.errors.append("dropout must be in [0, 1]")
            if model["dropout"] > 0.5:
                self.warnings.append("High dropout may hurt performance")

    def _validate_training(self, training: Dict[str, Any]):
        """Validate training configuration."""
        # Check types
        for key, expected_type in self.TRAINING_KEYS.items():
            if key in training and not isinstance(training[key], expected_type):
                self.errors.append(f"Training.{key} should be {expected_type.__name__}")

        # Check values
        if "num_steps" in training:
            if training["num_steps"] < 1:
                self.errors.append("num_steps must be at least 1")

        if "batch_size" in training:
            if training["batch_size"] < 1:
                self.errors.append("batch_size must be at least 1")
            if training["batch_size"] > 1000:
                self.warnings.append("Large batch_size needs more memory")

        if "learning_rate" in training:
            if training["learning_rate"] <= 0:
                self.errors.append("learning_rate must be positive")
            if training["learning_rate"] > 0.1:
                self.warnings.append("High learning_rate may cause instability")

        if "lr_schedule" in training:
            if training["lr_schedule"] not in self.VALID_LR_SCHEDULES:
                self.errors.append(f"lr_schedule must be one of {self.VALID_LR_SCHEDULES}")

        if "val_split" in training:
            if not 0 < training["val_split"] < 1:
                self.errors.append("val_split must be in (0, 1)")

        if "warmup_steps" in training:
            if training["warmup_steps"] < 0:
                self.errors.append("warmup_steps must be non-negative")
            if training["warmup_steps"] > training.get("num_steps", 1000) * 0.5:
                self.warnings.append("Warmup is more than 50% of training")

    def _validate_generation(self, generation: Dict[str, Any]):
        """Validate generation configuration."""
        # Check types
        for key, expected_type in self.GENERATION_KEYS.items():
            if key in generation and not isinstance(generation[key], expected_type):
                self.errors.append(f"Generation.{key} should be {expected_type.__name__}")

        # Check values
        if "temperature" in generation:
            if generation["temperature"] <= 0:
                self.errors.append("temperature must be positive")
            if generation["temperature"] > 2.0:
                self.warnings.append("High temperature may produce random output")

        if "top_k" in generation:
            if generation["top_k"] < 0:
                self.errors.append("top_k must be non-negative")

        if "top_p" in generation:
            if not 0 < generation["top_p"] <= 1:
                self.errors.append("top_p must be in (0, 1]")

        if "max_length" in generation:
            if generation["max_length"] < 1:
                self.errors.append("max_length must be at least 1")
            if generation["max_length"] > 10000:
                self.warnings.append("Very long generation may be slow")

    def _cross_validate(self, config: Dict[str, Any]):
        """Cross-validate configuration sections."""
        # Check model/training compatibility
        if "model" in config and "training" in config:
            model = config["model"]
            training = config["training"]

            # Batch size vs model size
            if training.get("batch_size", 1) > 1 and model.get("n_embd", 0) > 512:
                self.warnings.append("Large batch_size + large model may OOM")

            # Block size vs training steps
            if model.get("block_size", 16) > 512 and training.get("num_steps", 1000) > 10000:
                self.suggestions.append("Consider reducing block_size or steps for faster training")

    def _generate_suggestions(self, config: Dict[str, Any]):
        """Generate optimization suggestions."""
        if "model" in config:
            model = config["model"]

            # Suggest better defaults
            if model.get("n_embd", 0) < 64:
                self.suggestions.append("Consider n_embd >= 64 for better performance")

            if model.get("n_layer", 0) == 1:
                self.suggestions.append("Try n_layer=2+ for better capacity")

            if not model.get("use_gelu", False):
                self.suggestions.append("Consider GELU for better convergence")

        if "training" in config:
            training = config["training"]

            if training.get("lr_schedule") == "constant":
                self.suggestions.append("Try cosine or linear schedule for better convergence")

            if training.get("grad_clip", 0) == 0:
                self.suggestions.append("Consider gradient clipping for stability")

    def validate_file(self, path: str) -> ValidationResult:
        """Validate configuration from file."""
        path = Path(path)

        if not path.exists():
            return ValidationResult(
                valid=False, errors=[f"File not found: {path}"], warnings=[], suggestions=[]
            )

        try:
            with open(path) as f:
                if path.suffix == ".json":
                    config = json.load(f)
                elif path.suffix in [".yaml", ".yml"]:
                    config = yaml.safe_load(f)
                else:
                    return ValidationResult(
                        valid=False,
                        errors=[f"Unsupported file format: {path.suffix}"],
                        warnings=[],
                        suggestions=[],
                    )
        except Exception as e:
            return ValidationResult(
                valid=False, errors=[f"Failed to parse file: {e}"], warnings=[], suggestions=[]
            )

        return self.validate(config)

    def print_report(self, result: ValidationResult):
        """Print validation report."""
        print("=" * 70)
        print("Configuration Validation Report")
        print("=" * 70)

        status = "✓ VALID" if result.valid else "✗ INVALID"
        print(f"Status: {status}")
        print()

        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  ✗ {error}")
            print()

        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  ⚠ {warning}")
            print()

        if result.suggestions:
            print("Suggestions:")
            for suggestion in result.suggestions:
                print(f"  💡 {suggestion}")
            print()

        print("=" * 70)


# Example usage
if __name__ == "__main__":
    validator = ConfigValidator()

    # Test with default config
    from config import DEFAULT_CONFIG
    import dataclasses

    config_dict = dataclasses.asdict(DEFAULT_CONFIG)
    result = validator.validate(config_dict)
    validator.print_report(result)

    # Test with file
    print("\n")
    result = validator.validate_file("config.yaml")
    validator.print_report(result)
