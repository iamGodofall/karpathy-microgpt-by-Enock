"""
Perpetual Probability System
Inspired by No Man's Sky procedural generation and Mandelbrot fractals.
Generates infinite, evolving AI models with emergent behaviors.
"""

import random
import math
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set


class ProbabilityField:
    """A field of probabilities that evolves over time."""

    def __init__(
        self,
        dimensions: int = 128,
        coherence: float = 0.5,
        mutation_rate: float = 0.01,
        seed: int = 42,
    ):
        self.dimensions = dimensions
        self.coherence = coherence
        self.mutation_rate = mutation_rate
        self.seed = seed

        random.seed(seed)
        self.field = [random.random() for _ in range(dimensions)]
        self.history = [self.field.copy()]
        self.time_step = 0

    def evolve(self, steps: int = 1):
        """Evolve the probability field."""
        for _ in range(steps):
            new_field = []
            for i, val in enumerate(self.field):
                # Neighbor influence (like cellular automata)
                left = self.field[(i - 1) % self.dimensions]
                right = self.field[(i + 1) % self.dimensions]

                # Mandelbrot-inspired update: z = z² + c
                c = (left + right) / 2 - 0.5
                new_val = self.coherence * (val * val + c)
                new_val += (1 - self.coherence) * random.random()

                # Apply mutation
                if random.random() < self.mutation_rate:
                    new_val = random.random()

                new_field.append(max(0.0, min(1.0, new_val)))

            self.field = new_field
            self.history.append(self.field.copy())
            self.time_step += 1

    def sample(self, n: int = 1) -> List[float]:
        """Sample from the probability field."""
        return [random.choice(self.field) for _ in range(n)]

    def get_entropy(self) -> float:
        """Calculate entropy of the field."""
        from math import log2

        entropy = 0.0
        for p in self.field:
            if p > 0:
                entropy -= p * log2(p)
        return entropy / self.dimensions

    def visualize(self, width: int = 64) -> str:
        """Create ASCII visualization."""
        chars = " ▁▂▃▄▅▆▇█"
        result = []
        for i in range(0, len(self.field), width):
            row = self.field[i : i + width]
            line = "".join(chars[int(v * (len(chars) - 1))] for v in row)
            result.append(line)
        return "\n".join(result)


class ProceduralRule:
    """A rule that generates structure procedurally."""

    def __init__(
        self, rule_id: str, condition, transformation, probability: float = 1.0, depth: int = 0
    ):
        self.rule_id = rule_id
        self.condition = condition
        self.transformation = transformation
        self.probability = probability
        self.depth = depth

    def apply(self, state: Any) -> Optional[Any]:
        """Apply rule if condition matches."""
        if random.random() < self.probability and self.condition(state):
            return self.transformation(state)
        return None


class ProceduralGenerator:
    """No Man's Sky style procedural generator."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.rules: List[ProceduralRule] = []
        self.generated_objects: List[Dict] = []
        self.universe_state = {}

    def add_rule(self, rule: ProceduralRule):
        """Add a generation rule."""
        self.rules.append(rule)

    def generate(self, initial_state: Any, steps: int = 10) -> List[Any]:
        """Generate sequence of states."""
        states = [initial_state]
        current = initial_state

        for step in range(steps):
            # Try all rules
            applicable = [r for r in self.rules if r.condition(current)]

            if not applicable:
                break

            # Select and apply rule
            rule = random.choice(applicable)
            new_state = rule.apply(current)

            if new_state is not None:
                states.append(new_state)
                current = new_state

        return states

    def generate_universe(self, num_systems: int = 100) -> Dict[str, Any]:
        """Generate a universe like No Man's Sky."""
        universe = {"seed": self.seed, "systems": [], "galaxies": [], "entities": []}

        for i in range(num_systems):
            system = self._generate_star_system(i)
            universe["systems"].append(system)

        return universe

    def _generate_star_system(self, index: int) -> Dict:
        """Generate a star system."""
        random.seed(self.seed + index)

        return {
            "id": f"system_{index}",
            "star_type": random.choice(["G", "K", "M", "O", "B"]),
            "planets": [self._generate_planet(i) for i in range(random.randint(1, 8))],
            "coordinates": {
                "x": random.gauss(0, 1000),
                "y": random.gauss(0, 1000),
                "z": random.gauss(0, 1000),
            },
        }

    def _generate_planet(self, index: int) -> Dict:
        """Generate a planet."""
        resources = []
        for _ in range(3):
            resources.append(random.choice(["iron", "copper", "gold", "uranium"]))

        return {
            "id": f"planet_{index}",
            "type": random.choice(["terran", "gas_giant", "ice", "lava", "ocean"]),
            "life_probability": random.random(),
            "resources": resources,
            "atmosphere": random.choice(["none", "thin", "breathable", "toxic"]),
        }


class PerpetualModel:
    """A model that perpetually evolves and generates new variations."""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.generation = 0
        self.variants: List[Dict] = []
        self.prob_field = ProbabilityField(seed=base_config.get("seed", 42))
        self.procedural = ProceduralGenerator(seed=base_config.get("seed", 42))

        # Evolution history
        self.fitness_history: List[float] = []
        self.complexity_history: List[float] = []

    def evolve_generation(self, fitness_func=None):
        """Evolve to next generation."""
        self.generation += 1

        # Evolve probability field
        self.prob_field.evolve(steps=self.generation)

        # Generate new variant
        variant = self._generate_variant()
        self.variants.append(variant)

        # Evaluate if fitness function provided
        if fitness_func:
            fitness = fitness_func(variant)
            self.fitness_history.append(fitness)

        complexity = self._calculate_complexity(variant)
        self.complexity_history.append(complexity)

        return variant

    def _generate_variant(self) -> Dict[str, Any]:
        """Generate a new model variant."""
        # Use probability field to sample parameters
        params = self.prob_field.sample(10)

        variant = {
            "generation": self.generation,
            "seed": self.base_config.get("seed", 42) + self.generation,
            "n_layer": int(params[0] * 8) + 1,
            "n_embd": int(params[1] * 256) + 16,
            "n_head": int(params[2] * 8) + 1,
            "learning_rate": params[3] * 0.1,
            "dropout": params[4] * 0.5,
            "fractal_depth": int(params[5] * 10) + 1,
            "coherence": params[6],
            "mutation_rate": params[7] * 0.1,
            "specialization": random.choice(["language", "vision", "multimodal", "reasoning"]),
        }

        return variant

    def _calculate_complexity(self, variant: Dict) -> float:
        """Calculate complexity score."""
        # Based on architecture size and fractal depth
        layers = variant.get("n_layer", 1)
        embed = variant.get("n_embd", 16)
        depth = variant.get("fractal_depth", 1)

        return layers * embed * depth * math.log1p(layers * embed)

    def get_best_variant(self) -> Optional[Dict]:
        """Get best variant by fitness."""
        if not self.variants or not self.fitness_history:
            return None

        best_idx = min(range(len(self.fitness_history)), key=lambda i: self.fitness_history[i])
        return self.variants[best_idx]

    def generate_report(self) -> str:
        """Generate evolution report."""
        lines = [
            "# Perpetual Model Evolution Report",
            f"Total Generations: {self.generation}",
            f"Total Variants: {len(self.variants)}",
            f"Current Entropy: {self.prob_field.get_entropy():.4f}",
            "",
        ]

        if self.fitness_history:
            lines.append(f"Best Fitness: {min(self.fitness_history):.4f}")
            avg_fit = sum(self.fitness_history) / len(self.fitness_history)
            lines.append(f"Avg Fitness: {avg_fit:.4f}")
            lines.append("")

        if self.complexity_history:
            lines.append(f"Max Complexity: {max(self.complexity_history):.2f}")
            avg_comp = sum(self.complexity_history) / len(self.complexity_history)
            lines.append(f"Avg Complexity: {avg_comp:.2f}")
            lines.append("")

        # Show recent variants
        lines.append("## Recent Variants")
        for v in self.variants[-5:]:
            lines.append(
                f"- Gen {v['generation']}: {v['specialization']}, "
                f"layers={v['n_layer']}, embed={v['n_embd']}"
            )

        return "\n".join(lines)


class InfiniteUniverse:
    """Infinite universe of evolving models."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.models: Dict[str, PerpetualModel] = {}
        self.universe = ProceduralGenerator(seed).generate_universe(1000)
        self.active_regions: Set[str] = set()

    def spawn_model(self, region_id: str, config: Dict) -> PerpetualModel:
        """Spawn a new model in a region."""
        model = PerpetualModel(config)
        self.models[region_id] = model
        self.active_regions.add(region_id)
        return model

    def evolve_all(self, steps: int = 1):
        """Evolve all models."""
        for region_id, model in self.models.items():
            for _ in range(steps):
                model.evolve_generation()
            print(f"Region {region_id}: Generation {model.generation}")

    def discover_region(self, coordinates: Tuple[float, float, float]) -> str:
        """Discover a new region (like No Man's Sky)."""
        # Hash coordinates to get region ID
        x, y, z = coordinates
        coord_str = f"{x:.2f},{y:.2f},{z:.2f}"
        region_id = hashlib.md5(coord_str.encode()).hexdigest()[:8]

        if region_id not in self.models:
            # Create new model for this region
            config = {"seed": self.seed + len(self.models), "coordinates": coordinates}
            self.spawn_model(region_id, config)
            print(f"Discovered new region: {region_id}")

        return region_id

    def get_universe_stats(self) -> Dict[str, Any]:
        """Get universe statistics."""
        total_regions = len(self.models)
        active = len(self.active_regions)
        total_gen = sum(m.generation for m in self.models.values())

        avg_complexity = 0.0
        if self.models:
            complexities = []
            for m in self.models.values():
                if m.complexity_history:
                    complexities.append(sum(m.complexity_history) / len(m.complexity_history))
            if complexities:
                avg_complexity = sum(complexities) / len(complexities)

        return {
            "total_regions": total_regions,
            "active_regions": active,
            "total_generations": total_gen,
            "avg_complexity": avg_complexity,
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Perpetual Probability System")
    print("No Man's Sky × Mandelbrot × Infinite Evolution")
    print("=" * 70)

    # Create perpetual model
    config = {"seed": 42, "base_lr": 0.01}
    model = PerpetualModel(config)

    print("\nEvolving 10 generations...")
    for i in range(10):
        variant = model.evolve_generation(fitness_func=lambda v: random.random())
        print(
            f"  Gen {i+1}: {variant['specialization']}, "
            f"complexity={model.complexity_history[-1]:.2f}"
        )

    print(f"\nBest variant: {model.get_best_variant()}")
    print(f"\n{model.generate_report()}")

    # Probability field visualization
    print("\nProbability Field Evolution:")
    print(model.prob_field.visualize(64))

    # Infinite universe
    print("\n" + "=" * 70)
    print("Infinite Universe")
    print("=" * 70)

    universe = InfiniteUniverse(seed=42)

    # Discover regions
    for i in range(5):
        coords = (random.gauss(0, 100), random.gauss(0, 100), random.gauss(0, 100))
        region = universe.discover_region(coords)

    # Evolve all
    universe.evolve_all(steps=3)

    print(f"\nUniverse stats: {universe.get_universe_stats()}")

    print("\n✨ Infinite possibilities from simple rules")
    print("✨ Like No Man's Sky: endless exploration")
    print("✨ Like Mandelbrot: infinite complexity")
