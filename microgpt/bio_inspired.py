"""
Bio-inspired AI: Genetic algorithms, neural evolution, and natural selection.
Evolutionary computation meets microgpt.
"""

import random
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field


@dataclass
class Genome:
    """Genetic representation of a neural network."""

    genes: List[float] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0
    species_id: str = "default"

    def mutate(self, rate: float = 0.1, strength: float = 0.1):
        """Apply mutation to genome."""
        for i in range(len(self.genes)):
            if random.random() < rate:
                self.genes[i] += random.gauss(0, strength)
                self.genes[i] = max(-1, min(1, self.genes[i]))  # Clip

    def crossover(self, other: "Genome") -> "Genome":
        """Create child through crossover."""
        child = Genome(
            generation=max(self.generation, other.generation) + 1, species_id=self.species_id
        )

        # Uniform crossover
        for g1, g2 in zip(self.genes, other.genes):
            if random.random() < 0.5:
                child.genes.append(g1)
            else:
                child.genes.append(g2)

        return child

    def clone(self) -> "Genome":
        """Create exact copy."""
        g = Genome(
            genes=self.genes.copy(),
            fitness=self.fitness,
            generation=self.generation,
            species_id=self.species_id,
        )
        return g


class NeuralPhenotype:
    """Neural network expressed from genome."""

    def __init__(self, genome: Genome, architecture: List[int]):
        self.genome = genome
        self.architecture = architecture
        self.weights = self._decode_genome()

    def _decode_genome(self) -> List[List[List[float]]]:
        """Decode genome into weight matrices."""
        weights = []
        gene_idx = 0

        for i in range(len(self.architecture) - 1):
            in_size = self.architecture[i]
            out_size = self.architecture[i + 1]

            layer_weights = []
            for _ in range(out_size):
                neuron_weights = []
                for _ in range(in_size):
                    if gene_idx < len(self.genome.genes):
                        neuron_weights.append(self.genome.genes[gene_idx])
                        gene_idx += 1
                    else:
                        neuron_weights.append(random.gauss(0, 0.1))
                layer_weights.append(neuron_weights)

            weights.append(layer_weights)

        return weights

    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through network."""
        x = inputs

        for layer_weights in self.weights:
            new_x = []
            for neuron_weights in layer_weights:
                # Weighted sum
                activation = sum(w * xi for w, xi in zip(neuron_weights, x))
                # ReLU
                new_x.append(max(0, activation))
            x = new_x

        return x

    def evaluate(self, fitness_func: Callable) -> float:
        """Evaluate fitness."""
        self.genome.fitness = fitness_func(self)
        return self.genome.fitness


class EvolutionaryAlgorithm:
    """Genetic algorithm for evolving neural networks."""

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: int = 5,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        self.population: List[Genome] = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def initialize(self, genome_size: int, architecture: List[int]):
        """Create initial population."""
        self.architecture = architecture

        for i in range(self.population_size):
            genome = Genome(
                genes=[random.gauss(0, 0.1) for _ in range(genome_size)],
                generation=0,
                species_id=f"species_{i % 5}",  # 5 species
            )
            self.population.append(genome)

    def evolve(self, fitness_func: Callable, generations: int = 100):
        """Run evolution."""
        for gen in range(generations):
            self.generation = gen

            # Evaluate population
            fitnesses = []
            for genome in self.population:
                phenotype = NeuralPhenotype(genome, self.architecture)
                fitness = phenotype.evaluate(fitness_func)
                fitnesses.append(fitness)

            # Statistics
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            if gen % 10 == 0:
                print(f"Gen {gen}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

            # Selection and reproduction
            self._create_next_generation(fitnesses)

    def _create_next_generation(self, fitnesses: List[float]):
        """Create next generation through selection."""
        # Sort by fitness
        sorted_pop = sorted(zip(self.population, fitnesses), key=lambda x: x[1], reverse=True)

        new_population = []

        # Elitism: keep best individuals
        for i in range(self.elitism):
            new_population.append(sorted_pop[i][0].clone())

        # Create rest through selection
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(sorted_pop)

            if random.random() < self.crossover_rate:
                parent2 = self._tournament_selection(sorted_pop)
                child = parent1.crossover(parent2)
            else:
                child = parent1.clone()

            child.mutate(self.mutation_rate)
            new_population.append(child)

        self.population = new_population

    def _tournament_selection(
        self, sorted_pop: List[Tuple[Genome, float]], tournament_size: int = 3
    ) -> Genome:
        """Tournament selection."""
        tournament = random.sample(sorted_pop, min(tournament_size, len(sorted_pop)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    def get_best(self) -> Genome:
        """Get best genome."""
        return max(self.population, key=lambda g: g.fitness)


class Speciation:
    """Speciation for maintaining diversity."""

    def __init__(self, compatibility_threshold: float = 3.0):
        self.compatibility_threshold = compatibility_threshold
        self.species: Dict[str, List[Genome]] = {}

    def speciate(self, population: List[Genome]):
        """Assign genomes to species."""
        self.species = {}

        for genome in population:
            found = False
            for species_id, members in self.species.items():
                if self._compatible(genome, members[0]):
                    members.append(genome)
                    genome.species_id = species_id
                    found = True
                    break

            if not found:
                new_id = f"species_{len(self.species)}"
                self.species[new_id] = [genome]
                genome.species_id = new_id

    def _compatible(self, g1: Genome, g2: Genome) -> bool:
        """Check if two genomes are compatible."""
        if len(g1.genes) != len(g2.genes):
            return False

        # Compatibility distance
        distance = sum(abs(a - b) for a, b in zip(g1.genes, g2.genes))
        distance /= len(g1.genes)

        return distance < self.compatibility_threshold

    def get_species_count(self) -> int:
        """Get number of species."""
        return len(self.species)


class AntColonyOptimizer:
    """Ant Colony Optimization for neural architecture search."""

    def __init__(self, n_ants: int = 20, evaporation: float = 0.1):
        self.n_ants = n_ants
        self.evaporation = evaporation
        self.pheromones: Dict[Tuple, float] = {}
        self.best_path = None
        self.best_score = float("inf")

    def optimize(
        self, choices: Dict[str, List[Any]], evaluate: Callable, iterations: int = 100
    ) -> Dict[str, Any]:
        """Find optimal architecture using ACO."""
        # Initialize pheromones
        for key, options in choices.items():
            for opt in options:
                self.pheromones[(key, opt)] = 1.0

        for iteration in range(iterations):
            all_paths = []

            # Each ant builds a solution
            for ant in range(self.n_ants):
                path = {}
                for key, options in choices.items():
                    # Probabilistic selection based on pheromones
                    probs = [self.pheromones.get((key, opt), 1.0) for opt in options]
                    total = sum(probs)
                    probs = [p / total for p in probs]

                    path[key] = random.choices(options, weights=probs)[0]

                score = evaluate(path)
                all_paths.append((path, score))

                if score < self.best_score:
                    self.best_score = score
                    self.best_path = path

            # Update pheromones
            for key in self.pheromones:
                self.pheromones[key] *= 1 - self.evaporation

            # Add new pheromones based on performance
            for path, score in all_paths:
                deposit = 1.0 / (1 + score)
                for key, value in path.items():
                    self.pheromones[(key, value)] += deposit

            if iteration % 10 == 0:
                print(f"ACO Iteration {iteration}: Best={self.best_score:.4f}")

        return self.best_path


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Bio-Inspired AI: Evolutionary Computation")
    print("Genetic Algorithms × Speciation × Ant Colony")
    print("=" * 70)

    # Genetic Algorithm
    print("\n1. Genetic Algorithm for Neural Networks")
    ga = EvolutionaryAlgorithm(population_size=50)
    ga.initialize(genome_size=100, architecture=[4, 8, 4, 2])

    # XOR-like fitness function
    def fitness(phenotype):
        # Test on simple patterns
        tests = [
            ([0, 0, 0, 0], [0, 0]),
            ([1, 1, 0, 0], [1, 0]),
            ([0, 0, 1, 1], [0, 1]),
            ([1, 1, 1, 1], [1, 1]),
        ]
        error = 0
        for inp, target in tests:
            out = phenotype.forward(inp)
            error += sum((o - t) ** 2 for o, t in zip(out, target))
        return 1.0 / (1 + error)

    ga.evolve(fitness, generations=50)
    best = ga.get_best()
    print(f"\nBest fitness: {best.fitness:.4f}")
    print(f"Generation: {best.generation}")

    # Speciation
    print("\n2. Speciation")
    speciation = Speciation(compatibility_threshold=0.5)
    speciation.speciate(ga.population)
    print(f"Number of species: {speciation.get_species_count()}")

    # Ant Colony Optimization
    print("\n3. Ant Colony Optimization")
    aco = AntColonyOptimizer(n_ants=10)

    choices = {
        "n_layer": [1, 2, 4, 8],
        "n_embd": [16, 32, 64],
        "n_head": [2, 4, 8],
        "lr": [0.001, 0.01, 0.1],
    }

    def eval_arch(arch):
        # Simulate architecture evaluation
        score = (arch["n_layer"] * arch["n_embd"]) / 100.0
        return score + random.gauss(0, 0.1)

    best_arch = aco.optimize(choices, eval_arch, iterations=30)
    print(f"Best architecture: {best_arch}")
    print(f"Best score: {aco.best_score:.4f}")

    print("\n✨ Evolutionary computation")
    print("✨ Natural selection of neural networks")
    print("✨ Speciation maintains diversity")
    print("✨ Swarm intelligence for architecture search")
