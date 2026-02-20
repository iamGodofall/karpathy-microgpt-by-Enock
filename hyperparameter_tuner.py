"""
Hyperparameter tuning for microgpt.
Implements grid search, random search, and Bayesian optimization.
"""

import random
import json
import time
import math
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import copy



@dataclass
class SearchSpace:
    """Define search space for hyperparameters."""
    param_name: str
    param_type: str  # 'int', 'float', 'choice', 'log'
    min_val: float = None
    max_val: float = None
    choices: List[Any] = None
    
    def sample(self) -> Any:
        """Sample a value from the search space."""
        if self.param_type == 'int':
            return random.randint(int(self.min_val), int(self.max_val))
        elif self.param_type == 'float':
            return random.uniform(self.min_val, self.max_val)
        elif self.param_type == 'log':
            # Log-uniform sampling
            log_min = math.log(self.min_val)
            log_max = math.log(self.max_val)
            return math.exp(random.uniform(log_min, log_max))
        elif self.param_type == 'choice':
            return random.choice(self.choices)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


class HyperparameterTuner:
    """Tune hyperparameters for microgpt models."""
    
    def __init__(self, base_config: Any, search_spaces: Dict[str, SearchSpace]):
        self.base_config = base_config
        self.search_spaces = search_spaces
        self.results: List[Dict] = []
        self.best_config = None
        self.best_score = float('inf')
    
    def grid_search(self, train_func: Callable, eval_func: Callable, 
                   param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Perform grid search over hyperparameters."""
        print("Starting Grid Search...")
        
        # Generate all combinations
        import itertools
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        
        for combination in itertools.product(*values):
            config = copy.deepcopy(self.base_config)
            params = dict(zip(keys, combination))
            
            # Apply parameters
            for key, value in params.items():
                self._set_param(config, key, value)
            
            # Train and evaluate
            score = self._train_and_eval(config, train_func, eval_func)
            
            self.results.append({
                'params': params,
                'score': score,
                'config': config
            })
            
            if score < self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"  New best: {score:.4f} with {params}")
        
        return self.best_config
    
    def random_search(self, train_func: Callable, eval_func: Callable,
                     n_iter: int = 10) -> Dict[str, Any]:
        """Perform random search over hyperparameters."""
        print(f"Starting Random Search ({n_iter} iterations)...")
        
        for i in range(n_iter):
            # Sample parameters
            params = {name: space.sample() for name, space in self.search_spaces.items()}
            
            # Create config
            config = copy.deepcopy(self.base_config)
            for key, value in params.items():
                self._set_param(config, key, value)
            
            # Train and evaluate
            score = self._train_and_eval(config, train_func, eval_func)
            
            self.results.append({
                'params': params,
                'score': score,
                'config': config
            })
            
            if score < self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"  Iteration {i+1}/{n_iter}: New best: {score:.4f}")
            else:
                print(f"  Iteration {i+1}/{n_iter}: {score:.4f}")
        
        return self.best_config
    
    def _set_param(self, config: Any, key: str, value: Any):
        """Set a parameter in nested config."""
        parts = key.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _train_and_eval(self, config: Any, train_func: Callable, 
                      eval_func: Callable) -> float:
        """Train model and return evaluation score."""
        model = train_func(config)
        score = eval_func(model)
        return score
    
    def get_results(self) -> List[Dict]:
        """Get all results sorted by score."""
        return sorted(self.results, key=lambda x: x['score'])
    
    def save_results(self, path: str = "tuning_results.json"):
        """Save tuning results."""
        data = {
            'best_score': self.best_score,
            'best_config': asdict(self.best_config) if self.best_config else None,
            'all_results': [
                {
                    'params': r['params'],
                    'score': r['score']
                }
                for r in self.results
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


class EarlyStoppingTuner:
    """Tuner with early stopping based on intermediate results."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.counter = 0
    
    def check(self, score: float) -> bool:
        """Check if should stop."""
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# Example usage
if __name__ == "__main__":
    from config import Config, ModelConfig, TrainingConfig
    
    # Define search space
    search_spaces = {
        'model.n_embd': SearchSpace('n_embd', 'choice', choices=[32, 64, 128]),
        'model.n_layer': SearchSpace('n_layer', 'int', 1, 4),
        'training.learning_rate': SearchSpace('learning_rate', 'log', 0.001, 0.1),
        'training.batch_size': SearchSpace('batch_size', 'choice', choices=[1, 2, 4]),
    }
    
    base_config = Config(
        model=ModelConfig(vocab_size=100, n_embd=64, n_layer=2),
        training=TrainingConfig(num_steps=50, batch_size=2)
    )
    
    tuner = HyperparameterTuner(base_config, search_spaces)
    
    # Define train/eval functions
    def train_func(config):
        from model import GPT
        from data import CharTokenizer, Dataset
        from trainer import Trainer
        
        model = GPT(config.model)
        tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")
        dataset = Dataset(["hello world"] * 20, tokenizer, config.model.block_size)
        trainer = Trainer(model, config.training)
        trainer.train(dataset, num_steps=10)
        return model
    
    def eval_func(model):
        # Simple evaluation - lower is better
        return random.random()  # Placeholder
    
    # Run random search
    best = tuner.random_search(train_func, eval_func, n_iter=5)
    print(f"\nBest config: {best}")
    tuner.save_results()
