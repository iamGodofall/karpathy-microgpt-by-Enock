"""
Fractal Neural Networks inspired by Mandelbrot set.
Recursive, self-similar architectures with infinite depth potential.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FractalConfig:
    """Configuration for fractal network."""

    max_depth: int = 8  # Like Mandelbrot iteration limit
    escape_radius: float = 2.0  # Like Mandelbrot escape radius
    c_real: float = -0.7  # Mandelbrot c parameter (real)
    c_imag: float = 0.27015  # Mandelbrot c parameter (imaginary)
    zoom_factor: float = 0.5  # Self-similarity zoom
    branching_factor: int = 2  # How many sub-networks per level


class ComplexValue:
    """Complex number for fractal computations."""

    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if isinstance(other, ComplexValue):
            return ComplexValue(self.real + other.real, self.imag + other.imag)
        return ComplexValue(self.real + other, self.imag)

    def __mul__(self, other):
        if isinstance(other, ComplexValue):
            # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return ComplexValue(real, imag)
        return ComplexValue(self.real * other, self.imag * other)

    def __pow__(self, n: int):
        result = ComplexValue(1, 0)
        for _ in range(n):
            result = result * self
        return result

    def magnitude(self) -> float:
        return math.sqrt(self.real**2 + self.imag**2)

    def __repr__(self):
        return f"{self.real:.4f}{'+' if self.imag >= 0 else ''}{self.imag:.4f}i"


class FractalNeuron:
    """A neuron with fractal activation."""

    def __init__(self, config: FractalConfig):
        self.config = config
        self.weights = [random.gauss(0, 0.1) for _ in range(4)]  # Complex weights
        self.bias = ComplexValue(random.gauss(0, 0.1), random.gauss(0, 0.1))
        self.iterations = 0

    def activate(self, x: ComplexValue) -> ComplexValue:
        """Fractal activation: z = z² + c (Mandelbrot-style)."""
        c = ComplexValue(self.config.c_real, self.config.c_imag)
        z = x

        for i in range(self.config.max_depth):
            z = z * z + c

            # Check escape condition
            if z.magnitude() > self.config.escape_radius:
                self.iterations = i
                return z

        self.iterations = self.config.max_depth
        return z

    def forward(self, inputs: List[float]) -> float:
        """Forward pass with fractal activation."""
        # Convert inputs to complex
        real = sum(w * x for w, x in zip(self.weights[:2], inputs[:2]))
        imag = sum(w * x for w, x in zip(self.weights[2:], inputs[2:])) if len(inputs) > 2 else 0

        z = ComplexValue(real, imag) + self.bias
        result = self.activate(z)

        # Return magnitude as output
        return result.magnitude() / self.config.escape_radius


class FractalLayer:
    """Layer of fractal neurons with self-similar structure."""

    def __init__(self, in_features: int, out_features: int, config: FractalConfig, depth: int = 0):
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.depth = depth

        # Create neurons
        self.neurons = [FractalNeuron(config) for _ in range(out_features)]

        # Recursive sub-layers (fractal self-similarity)
        self.sub_layers = []
        if depth < config.max_depth and out_features > 1:
            # Create sub-layers with zoomed parameters
            sub_config = FractalConfig(
                max_depth=config.max_depth - 1,
                escape_radius=config.escape_radius * config.zoom_factor,
                c_real=config.c_real * config.zoom_factor,
                c_imag=config.c_imag * config.zoom_factor,
            )
            # Split neurons into groups
            group_size = max(1, out_features // config.branching_factor)
            for i in range(0, out_features, group_size):
                sub_layer = FractalLayer(
                    in_features, min(group_size, out_features - i), sub_config, depth + 1
                )
                self.sub_layers.append(sub_layer)

    def forward(self, inputs: List[float]) -> List[float]:
        """Forward through fractal layer."""
        # Main layer output
        outputs = [n.forward(inputs) for n in self.neurons]

        # Add sub-layer contributions (fractal detail)
        if self.sub_layers:
            sub_outputs = []
            for sub in self.sub_layers:
                sub_out = sub.forward(inputs)
                sub_outputs.extend(sub_out)

            # Blend main and sub-layer outputs
            blend_factor = 1.0 / (self.depth + 1)
            for i in range(len(outputs)):
                if i < len(sub_outputs):
                    outputs[i] = blend_factor * sub_outputs[i] + (1 - blend_factor) * outputs[i]

        return outputs

    def get_complexity(self) -> int:
        """Calculate structural complexity (like Mandelbrot set complexity)."""
        total = len(self.neurons)
        for sub in self.sub_layers:
            total += sub.get_complexity()
        return total


class FractalNetwork:
    """Complete fractal neural network."""

    def __init__(self, config: FractalConfig, layer_sizes: List[int]):
        self.config = config
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = FractalLayer(layer_sizes[i], layer_sizes[i + 1], config, depth=0)
            self.layers.append(layer)

    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through fractal network."""
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_total_complexity(self) -> int:
        """Get total network complexity."""
        return sum(layer.get_complexity() for layer in self.layers)

    def generate_fractal_pattern(self, width: int = 80, height: int = 40) -> str:
        """Generate ASCII art of the fractal structure."""
        result = []
        for y in range(height):
            row = []
            for x in range(width):
                # Map to complex plane
                cx = (x / width - 0.5) * 3.0
                cy = (y / height - 0.5) * 2.0

                c = ComplexValue(cx, cy)
                z = ComplexValue(0, 0)

                # Mandelbrot iteration
                iterations = 0
                for i in range(self.config.max_depth):
                    z = z * z + c
                    if z.magnitude() > self.config.escape_radius:
                        iterations = i
                        break

                # Map to character
                chars = " .:-=+*#%@"
                char_idx = min(iterations, len(chars) - 1)
                row.append(chars[char_idx])

            result.append("".join(row))

        return "\n".join(result)


class InfiniteFractalGenerator:
    """Generate infinite variations of fractal networks."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.generated_networks = []

    def generate(self, variation: int = 0) -> FractalNetwork:
        """Generate a new fractal network variation."""
        # Vary Mandelbrot parameters
        c_real = -0.7 + variation * 0.01
        c_imag = 0.27015 + variation * 0.005

        config = FractalConfig(
            max_depth=6 + variation % 4,
            c_real=c_real,
            c_imag=c_imag,
            zoom_factor=0.5 + variation * 0.05,
        )

        # Vary architecture
        layer_sizes = [16, 32, 16, 8]
        if variation % 2 == 0:
            layer_sizes = [32, 64, 32, 16]

        network = FractalNetwork(config, layer_sizes)
        self.generated_networks.append(network)

        return network

    def evolve(self, fitness_func: callable, generations: int = 10):
        """Evolve fractal networks using genetic algorithm."""
        population = [self.generate(i) for i in range(5)]

        for gen in range(generations):
            # Evaluate fitness
            fitness = [fitness_func(net) for net in population]

            # Select best
            best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
            best = population[best_idx]

            print(f"Generation {gen}: Best fitness = {fitness[best_idx]:.4f}")

            # Mutate and create new population
            new_population = [best]  # Keep best
            for _ in range(4):
                mutated = self._mutate(best)
                new_population.append(mutated)

            population = new_population

        return best

    def _mutate(self, network: FractalNetwork) -> FractalNetwork:
        """Mutate a network slightly."""
        # Create slightly varied version
        variation = len(self.generated_networks)
        return self.generate(variation)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Fractal Neural Network - Mandelbrot-Inspired")
    print("=" * 70)

    # Create fractal network
    config = FractalConfig(max_depth=8, c_real=-0.7, c_imag=0.27015)
    network = FractalNetwork(config, [4, 8, 4, 2])

    print(f"\nNetwork complexity: {network.get_total_complexity()} neurons")

    # Forward pass
    inputs = [0.5, -0.3, 0.8, 0.1]
    output = network.forward(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {[f'{o:.4f}' for o in output]}")

    # Generate fractal visualization
    print("\nFractal Pattern (Mandelbrot-inspired):")
    print(network.generate_fractal_pattern(60, 20))

    # Infinite generator
    print("\n" + "=" * 70)
    print("Infinite Fractal Generator")
    print("=" * 70)

    generator = InfiniteFractalGenerator(seed=42)

    for i in range(3):
        net = generator.generate(i)
        print(f"\nVariation {i}: complexity = {net.get_total_complexity()}")

    print("\n✨ Fractal networks: Infinite complexity from simple rules")
