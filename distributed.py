"""
Distributed training support for microgpt.
Simulates data parallelism and model parallelism concepts.
"""

import random
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from model import GPT, Value


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 1  # Number of processes/GPUs
    rank: int = 0        # Current process rank
    local_rank: int = 0  # Local GPU index
    backend: str = 'gloo'  # 'gloo', 'nccl' (simulated)


class DataParallelTrainer:
    """
    Data Parallel training wrapper.
    Distributes data across multiple workers and averages gradients.
    """
    
    def __init__(self, model: GPT, config: DistributedConfig):
        self.model = model
        self.config = config
        self.local_gradients: Dict[str, List[List[float]]] = {}
    
    def all_reduce_gradients(self, params: List[Value]):
        """
        Simulate all-reduce: average gradients across all workers.
        In real distributed training, this uses NCCL/Gloo.
        """
        if self.config.world_size == 1:
            return
        
        # Simulate: average gradients
        for i, p in enumerate(params):
            # In real implementation, this would be:
            # dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            p.grad /= self.config.world_size
    
    def train_step(self, batch_docs: List[str], tokenizer, step: int,
                   optimizer_step_fn: Callable) -> float:
        """
        Training step with data parallelism.
        Each worker processes a subset of the batch.
        """
        # Split batch across workers
        chunk_size = len(batch_docs) // self.config.world_size
        start_idx = self.config.rank * chunk_size
        end_idx = start_idx + chunk_size if self.config.rank < self.config.world_size - 1 else len(batch_docs)
        
        local_batch = batch_docs[start_idx:end_idx]
        
        # Compute local gradients
        total_loss = 0.0
        
        for doc in local_batch:
            tokens = [tokenizer.bos_token] + [
                tokenizer.char_to_idx.get(ch, tokenizer.bos_token) 
                for ch in doc if ch in tokenizer.char_to_idx
            ] + [tokenizer.bos_token]
            
            if len(tokens) < 2:
                continue
            
            # Forward/backward for this sample
            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]
            
            # Compute loss (simplified - would need proper loss computation)
            # This is a placeholder for the actual training logic
            total_loss += 0.0  # Would compute actual loss here
        
        # All-reduce gradients
        self.all_reduce_gradients(self.model.parameters())
        
        return total_loss / len(local_batch) if local_batch else 0.0


class GradientCheckpointing:
    """
    Gradient checkpointing to reduce memory usage.
    Trade computation for memory by recomputing activations in backward pass.
    """
    
    def __init__(self, model: GPT, checkpoint_every: int = 1):
        self.model = model
        self.checkpoint_every = checkpoint_every
        self.checkpoints: List[Dict] = []
    
    def forward_with_checkpoints(self, tokens: List[int]) -> Value:
        """
        Forward pass with gradient checkpointing.
        Only store activations at checkpoint boundaries.
        """
        # Store input as checkpoint
        self.checkpoints = [{'tokens': tokens, 'pos': 0}]
        
        # Forward through layers, storing checkpoints periodically
        for i in range(0, len(tokens) - 1, self.checkpoint_every):
            # Compute up to checkpoint
            # In real implementation, would store intermediate activations
            pass
        
        # Final output
        return Value(0.0)  # Placeholder
    
    def backward_from_checkpoints(self, loss: Value):
        """
        Backward pass recomputing activations from checkpoints.
        """
        # Recompute forward for each segment during backward
        # This saves memory at the cost of extra computation
        loss.backward()


class PipelineParallelTrainer:
    """
    Pipeline parallelism for large models.
    Split model layers across multiple devices.
    """
    
    def __init__(self, model: GPT, config: DistributedConfig, 
                 layers_per_device: Optional[List[int]] = None):
        self.model = model
        self.config = config
        
        # Assign layers to devices
        if layers_per_device is None:
            # Even distribution
            layers_per_device = [model.n_layer // config.world_size] * config.world_size
        
        self.layer_assignments = []
        layer_idx = 0
        for device_idx, num_layers in enumerate(layers_per_device):
            for _ in range(num_layers):
                self.layer_assignments.append(device_idx)
                layer_idx += 1
        
        self.local_layers = [
            i for i, d in enumerate(self.layer_assignments) 
            if d == config.rank
        ]
    
    def forward_pipeline(self, x: List[Value], micro_batch_size: int = 1):
        """
        Pipeline parallel forward pass with micro-batching.
        Uses pipeline bubbles for efficiency.
        """
        # In real implementation, this would:
        # 1. Split batch into micro-batches
        # 2. Pipeline them through different devices
        # 3. Handle communication between stages
        
        # For now, just forward through local layers
        for layer_idx in self.local_layers:
            # Would apply layer transformation
            pass
        
        return x


class FederatedLearning:
    """
    Federated learning simulation.
    Train on local data, share only gradients/weights.
    """
    
    def __init__(self, model: GPT, num_clients: int = 3):
        self.global_model = model
        self.num_clients = num_clients
        self.client_models = [
            GPT(
                vocab_size=model.vocab_size,
                block_size=model.block_size,
                n_layer=model.n_layer,
                n_embd=model.n_embd,
                n_head=model.n_head
            ) for _ in range(num_clients)
        ]
        
        # Copy global model to clients
        for client in self.client_models:
            self._copy_weights(self.global_model, client)
    
    def _copy_weights(self, source: GPT, target: GPT):
        """Copy weights from source to target model."""
        for name in source.state_dict.keys():
            for i, row in enumerate(source.state_dict[name]):
                for j, val in enumerate(row):
                    target.state_dict[name][i][j].data = val.data
    
    def client_update(self, client_id: int, local_data: List[str], 
                      tokenizer, num_steps: int = 10) -> Dict[str, List[List[float]]]:
        """
        Train client model on local data.
        Returns gradient updates.
        """
        client_model = self.client_models[client_id]
        
        # Train locally
        for step in range(num_steps):
            doc = random.choice(local_data)
            tokens = [tokenizer.bos_token] + [
                tokenizer.char_to_idx.get(ch, tokenizer.bos_token)
                for ch in doc if ch in tokenizer.char_to_idx
            ] + [tokenizer.bos_token]
            
            # Training step (simplified)
            # Would compute loss and backprop
        
        # Compute weight updates
        updates = {}
        for name in client_model.state_dict.keys():
            updates[name] = [
                [client_model.state_dict[name][i][j].data - 
                 self.global_model.state_dict[name][i][j].data
                 for j in range(len(row))]
                for i, row in enumerate(client_model.state_dict[name])
            ]
        
        return updates
    
    def federated_average(self, client_updates: List[Dict]):
        """
        Federated averaging (FedAvg).
        Average client updates and apply to global model.
        """
        # Average updates
        averaged = {}
        for name in self.global_model.state_dict.keys():
            averaged[name] = [
                [sum(updates[name][i][j] for updates in client_updates) / len(client_updates)
                 for j in range(len(row))]
                for i, row in enumerate(self.global_model.state_dict[name])
            ]
        
        # Apply to global model
        for name in self.global_model.state_dict.keys():
            for i, row in enumerate(self.global_model.state_dict[name]):
                for j in range(len(row)):
                    self.global_model.state_dict[name][i][j].data += averaged[name][i][j]
        
        # Sync to clients
        for client in self.client_models:
            self._copy_weights(self.global_model, client)


class AsyncTraining:
    """
    Asynchronous training with stale gradients.
    For large-scale distributed training.
    """
    
    def __init__(self, model: GPT, staleness_threshold: int = 5):
        self.model = model
        self.staleness_threshold = staleness_threshold
        self.gradient_queue: List[Tuple[int, Dict]] = []  # (step, gradients)
        self.current_step = 0
    
    def apply_gradient(self, gradients: Dict[str, List[List[float]]], 
                       gradient_step: int):
        """
        Apply gradient with staleness check.
        """
        staleness = self.current_step - gradient_step
        
        if staleness > self.staleness_threshold:
            # Gradient too stale, discard or reduce weight
            weight = 0.5  # Reduce weight for stale gradients
        else:
            weight = 1.0
        
        # Apply weighted gradient
        for name in self.model.state_dict.keys():
            for i, row in enumerate(self.model.state_dict[name]):
                for j in range(len(row)):
                    self.model.state_dict[name][i][j].data -= 0.01 * weight * gradients[name][i][j]
        
        self.current_step += 1


def simulate_distributed_training(model: GPT, num_workers: int = 4, 
                                  num_steps: int = 100):
    """
    Simulate distributed training with multiple workers.
    """
    print(f"Simulating distributed training with {num_workers} workers...")
    
    configs = [
        DistributedConfig(world_size=num_workers, rank=i, local_rank=i)
        for i in range(num_workers)
    ]
    
    trainers = [DataParallelTrainer(model, config) for config in configs]
    
    # Simulate training
    for step in range(num_steps):
        # Each worker processes different data
        for trainer in trainers:
            # Would train on local batch
            pass
        
        # Simulate synchronization barrier
        if step % 10 == 0:
            print(f"Step {step}: Synchronizing gradients across {num_workers} workers...")
    
    print("Distributed training simulation complete!")
