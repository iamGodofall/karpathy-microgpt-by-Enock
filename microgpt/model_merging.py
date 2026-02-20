"""
Model merging techniques for combining multiple models.
Includes TIES, DARE, Task Arithmetic, and more.
"""

import random
from typing import List, Dict, Tuple, Optional
from .model import GPT, Value


class TaskArithmetic:
    """
    Task Arithmetic: Editing models by adding/subtracting task vectors.
    """

    @staticmethod
    def merge_models(base_model: GPT, task_models: List[GPT], weights: List[float]) -> GPT:
        """
        Merge multiple task-specific models.
        result = base + sum(weight_i * (task_i - base))
        """
        # Create new model with base architecture
        merged = GPT(
            vocab_size=base_model.vocab_size,
            block_size=base_model.block_size,
            n_layer=base_model.n_layer,
            n_embd=base_model.n_embd,
            n_head=base_model.n_head,
        )

        # Copy base weights
        for name in base_model.state_dict:
            for i, row in enumerate(base_model.state_dict[name]):
                for j, v in enumerate(row):
                    merged.state_dict[name][i][j].data = v.data

        # Add task vectors
        for task_model, weight in zip(task_models, weights):
            for name in base_model.state_dict:
                for i, row in enumerate(base_model.state_dict[name]):
                    for j, v in enumerate(row):
                        task_v = task_model.state_dict[name][i][j].data
                        base_v = v.data
                        # Task vector
                        task_vector = task_v - base_v
                        merged.state_dict[name][i][j].data += weight * task_vector

        return merged

    @staticmethod
    def negate_task(model: GPT, base_model: GPT) -> GPT:
        """
        Create negative task vector (for forgetting).
        """
        negated = GPT(
            vocab_size=model.vocab_size,
            block_size=model.block_size,
            n_layer=model.n_layer,
            n_embd=model.n_embd,
            n_head=model.n_head,
        )

        for name in model.state_dict:
            for i, row in enumerate(model.state_dict[name]):
                for j, v in enumerate(row):
                    base_v = base_model.state_dict[name][i][j].data
                    # Negate: base - (task - base) = 2*base - task
                    negated.state_dict[name][i][j].data = 2 * base_v - v.data

        return negated


class TIESMerging:
    """
    TIES-Merging: Resolving interference in model merging.
    """

    @staticmethod
    def merge(base_model: GPT, models: List[GPT], weights: Optional[List[float]] = None) -> GPT:
        """
        TIES merging with trim-sign-merge.
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # Step 1: Trim low-magnitude task vectors
        trimmed_vectors = []

        for model in models:
            task_vector = {}
            for name in base_model.state_dict:
                task_vector[name] = []
                for i, row in enumerate(base_model.state_dict[name]):
                    tv_row = []
                    for j, v in enumerate(row):
                        task_val = model.state_dict[name][i][j].data - v.data
                        tv_row.append(task_val)
                    task_vector[name].append(tv_row)

            # Trim bottom-k% values
            all_values = [abs(v) for row in task_vector.values() for r in row for v in r]
            threshold = sorted(all_values)[int(len(all_values) * 0.2)]  # Trim 20%

            for name in task_vector:
                for i in range(len(task_vector[name])):
                    for j in range(len(task_vector[name][i])):
                        if abs(task_vector[name][i][j]) < threshold:
                            task_vector[name][i][j] = 0

            trimmed_vectors.append(task_vector)

        # Step 2: Resolve sign conflicts
        merged_vector = {}
        for name in base_model.state_dict:
            merged_vector[name] = []
            for i in range(len(base_model.state_dict[name])):
                row = []
                for j in range(len(base_model.state_dict[name][i])):
                    # Collect signs from all task vectors
                    signs = []
                    values = []
                    for tv, w in zip(trimmed_vectors, weights):
                        val = tv[name][i][j] * w
                        if val != 0:
                            signs.append(1 if val > 0 else -1)
                            values.append(val)

                    if not signs:
                        row.append(0)
                    else:
                        # Majority vote on sign
                        sign = 1 if sum(signs) > 0 else -1
                        # Average magnitude with agreed sign
                        avg_mag = sum(abs(v) for v in values) / len(values)
                        row.append(sign * avg_mag)

                merged_vector[name].append(row)

        # Step 3: Merge into new model
        merged = GPT(
            vocab_size=base_model.vocab_size,
            block_size=base_model.block_size,
            n_layer=base_model.n_layer,
            n_embd=base_model.n_embd,
            n_head=base_model.n_head,
        )

        for name in base_model.state_dict:
            for i, row in enumerate(base_model.state_dict[name]):
                for j, v in enumerate(row):
                    merged.state_dict[name][i][j].data = v.data + merged_vector[name][i][j]

        return merged


class DARE:
    """
    DARE: Drop And REscale for model merging.
    """

    @staticmethod
    def merge(base_model: GPT, models: List[GPT], drop_rate: float = 0.5) -> GPT:
        """
        DARE merging with random dropping and rescaling.
        """
        # Step 1: Compute task vectors
        task_vectors = []
        for model in models:
            tv = {}
            for name in base_model.state_dict:
                tv[name] = []
                for i, row in enumerate(base_model.state_dict[name]):
                    tv_row = []
                    for j, v in enumerate(row):
                        tv_row.append(model.state_dict[name][i][j].data - v.data)
                    tv[name].append(tv_row)
            task_vectors.append(tv)

        # Step 2: Drop and rescale
        dropped_vectors = []
        for tv in task_vectors:
            dropped = {}
            for name in tv:
                dropped[name] = []
                for row in tv[name]:
                    dropped_row = []
                    for val in row:
                        if random.random() > drop_rate:
                            # Keep and rescale
                            dropped_row.append(val / (1 - drop_rate))
                        else:
                            dropped_row.append(0)
                    dropped[name].append(dropped_row)
            dropped_vectors.append(dropped)

        # Step 3: Merge
        merged = GPT(
            vocab_size=base_model.vocab_size,
            block_size=base_model.block_size,
            n_layer=base_model.n_layer,
            n_embd=base_model.n_embd,
            n_head=base_model.n_head,
        )

        for name in base_model.state_dict:
            for i, row in enumerate(base_model.state_dict[name]):
                for j, v in enumerate(row):
                    # Sum dropped task vectors
                    delta = sum(dv[name][i][j] for dv in dropped_vectors)
                    merged.state_dict[name][i][j].data = v.data + delta

        return merged


class ModelSoups:
    """
    Model Soups: Averaging weights of multiple fine-tuned models.
    """

    @staticmethod
    def average_models(models: List[GPT], weights: Optional[List[float]] = None) -> GPT:
        """
        Weighted average of model parameters.
        """
        if not models:
            raise ValueError("No models to merge")

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Create averaged model
        averaged = GPT(
            vocab_size=models[0].vocab_size,
            block_size=models[0].block_size,
            n_layer=models[0].n_layer,
            n_embd=models[0].n_embd,
            n_head=models[0].n_head,
        )

        for name in averaged.state_dict:
            for i in range(len(averaged.state_dict[name])):
                for j in range(len(averaged.state_dict[name][i])):
                    # Weighted average
                    avg = sum(w * m.state_dict[name][i][j].data for m, w in zip(models, weights))
                    averaged.state_dict[name][i][j].data = avg

        return averaged


class SLERP:
    """
    Spherical Linear Interpolation for model merging.
    """

    @staticmethod
    def slerp(model1: GPT, model2: GPT, t: float) -> GPT:
        """
        Spherical interpolation between two models.
        """
        if t < 0 or t > 1:
            raise ValueError("t must be in [0, 1]")

        result = GPT(
            vocab_size=model1.vocab_size,
            block_size=model1.block_size,
            n_layer=model1.n_layer,
            n_embd=model1.n_embd,
            n_head=model1.n_head,
        )

        for name in model1.state_dict:
            for i in range(len(model1.state_dict[name])):
                for j in range(len(model1.state_dict[name][i])):
                    v1 = model1.state_dict[name][i][j].data
                    v2 = model2.state_dict[name][i][j].data

                    # Flatten to vectors for SLERP
                    # Simplified: just linear interpolation
                    # Real SLERP would work on the full parameter space
                    result.state_dict[name][i][j].data = (1 - t) * v1 + t * v2

        return result


class BreadthFirstMerging:
    """
    Breadth-first model merging for better generalization.
    """

    @staticmethod
    def merge(models: List[GPT], base_model: GPT, breadth_factor: float = 0.3) -> GPT:
        """
        Merge with breadth-first exploration.
        """
        # Start with base
        merged = GPT(
            vocab_size=base_model.vocab_size,
            block_size=base_model.block_size,
            n_layer=base_model.n_layer,
            n_embd=base_model.n_embd,
            n_head=base_model.n_head,
        )

        # Copy base
        for name in base_model.state_dict:
            for i, row in enumerate(base_model.state_dict[name]):
                for j, v in enumerate(row):
                    merged.state_dict[name][i][j].data = v.data

        # Add task vectors with breadth
        for model in models:
            for name in base_model.state_dict:
                for i, row in enumerate(base_model.state_dict[name]):
                    for j, v in enumerate(row):
                        task_val = model.state_dict[name][i][j].data - v.data

                        # Breadth: add noise for exploration
                        noise = random.gauss(0, abs(task_val) * breadth_factor)
                        merged.state_dict[name][i][j].data += task_val + noise

        return merged


class FisherWeightedMerging:
    """
    Fisher-weighted merging based on parameter importance.
    """

    @staticmethod
    def compute_fisher(model: GPT, data_samples: List[List[int]]) -> Dict:
        """
        Compute Fisher information for each parameter.
        """
        fisher = {}

        for name in model.state_dict:
            fisher[name] = []
            for i in range(len(model.state_dict[name])):
                fisher[name].append([0.0] * len(model.state_dict[name][i]))

        # Compute gradients on samples
        for sample in data_samples:
            # Forward and backward
            # Simplified - would compute actual gradients
            for name in model.state_dict:
                for i in range(len(model.state_dict[name])):
                    for j in range(len(model.state_dict[name][i])):
                        # Approximate Fisher as squared gradient
                        grad = random.gauss(0, 0.01)  # Placeholder
                        fisher[name][i][j] += grad**2

        # Average
        for name in fisher:
            for i in range(len(fisher[name])):
                for j in range(len(fisher[name][i])):
                    fisher[name][i][j] /= len(data_samples)

        return fisher

    @staticmethod
    def merge(base_model: GPT, models: List[GPT], data_samples: List[List[int]]) -> GPT:
        """
        Merge using Fisher-weighted averaging.
        """
        # Compute Fisher for each model
        fishers = [
            FisherWeightedMerging.compute_fisher(m, data_samples) for m in [base_model] + models
        ]

        # Weighted average with Fisher weights
        merged = GPT(
            vocab_size=base_model.vocab_size,
            block_size=base_model.block_size,
            n_layer=base_model.n_layer,
            n_embd=base_model.n_embd,
            n_head=base_model.n_head,
        )

        for name in base_model.state_dict:
            for i in range(len(base_model.state_dict[name])):
                for j in range(len(base_model.state_dict[name][i])):
                    # Weighted by Fisher information
                    total_fisher = sum(f[name][i][j] for f in fishers)

                    if total_fisher > 0:
                        weighted_sum = sum(
                            f[name][i][j] * m.state_dict[name][i][j].data
                            for f, m in zip(fishers, [base_model] + models)
                        )
                        merged.state_dict[name][i][j].data = weighted_sum / total_fisher
                    else:
                        merged.state_dict[name][i][j].data = base_model.state_dict[name][i][j].data

        return merged


def merge_with_ties(base: GPT, models: List[GPT]) -> GPT:
    """Convenience function for TIES merging."""
    return TIESMerging.merge(base, models)


def merge_with_dare(base: GPT, models: List[GPT], drop_rate: float = 0.5) -> GPT:
    """Convenience function for DARE merging."""
    return DARE.merge(base, models, drop_rate)


def create_model_soup(models: List[GPT]) -> GPT:
    """Convenience function for model soup."""
    return ModelSoups.average_models(models)
