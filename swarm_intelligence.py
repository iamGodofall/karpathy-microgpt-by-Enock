"""
Swarm Intelligence for microgpt.
Particle swarms, flocking behavior, and collective intelligence.
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class Particle:
    """Particle in swarm optimization."""
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float = float('inf')
    fitness: float = float('inf')
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]]):
        self.position = [random.uniform(b[0], b[1]) for b in bounds]
        self.velocity = [random.gauss(0, 0.1) for _ in range(dim)]
        self.best_position = self.position.copy()
        self.dim = dim
        self.bounds = bounds
    
    def update_velocity(self, global_best: List[float], 
                       w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """Update velocity using PSO formula."""
        for i in range(self.dim):
            r1, r2 = random.random(), random.random()
            
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            
            self.velocity[i] = w * self.velocity[i] + cognitive + social
    
    def update_position(self):
        """Update position based on velocity."""
        for i in range(self.dim):
            self.position[i] += self.velocity[i]
            # Boundary check
            self.position[i] = max(self.bounds[i][0], 
                                   min(self.bounds[i][1], self.position[i]))
    
    def evaluate(self, fitness_func: Callable):
        """Evaluate fitness."""
        self.fitness = fitness_func(self.position)
        
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization."""
    
    def __init__(self, n_particles: int = 30, dim: int = 10,
                 bounds: List[Tuple[float, float]] = None):
        self.n_particles = n_particles
        self.dim = dim
        self.bounds = bounds or [(-5, 5)] * dim
        
        self.particles: List[Particle] = []
        self.global_best_position: List[float] = None
        self.global_best_fitness = float('inf')
        self.history = []
    
    def initialize(self):
        """Initialize swarm."""
        self.particles = [
            Particle(self.dim, self.bounds) 
            for _ in range(self.n_particles)
        ]
    
    def optimize(self, fitness_func: Callable, iterations: int = 100) -> List[float]:
        """Run PSO."""
        self.initialize()
        
        for iteration in range(iterations):
            # Evaluate all particles
            for p in self.particles:
                p.evaluate(fitness_func)
                
                if p.fitness < self.global_best_fitness:
                    self.global_best_fitness = p.fitness
                    self.global_best_position = p.position.copy()
            
            # Update velocities and positions
            for p in self.particles:
                p.update_velocity(self.global_best_position)
                p.update_position()
            
            self.history.append(self.global_best_fitness)
            
            if iteration % 10 == 0:
                print(f"PSO Iteration {iteration}: Best={self.global_best_fitness:.6f}")
        
        return self.global_best_position
    
    def get_convergence_plot(self, width: int = 60) -> str:
        """ASCII convergence plot."""
        if not self.history:
            return "No history"
        
        lines = ["Convergence:"]
        max_val = max(self.history)
        min_val = min(self.history)
        range_val = max_val - min_val if max_val != min_val else 1
        
        for i, val in enumerate(self.history[::max(1, len(self.history)//20)]):
            height = int((val - min_val) / range_val * 10)
            bar = "█" * height + "░" * (10 - height)
            lines.append(f"{i:3d} |{bar}| {val:.4f}")
        
        return "\n".join(lines)


class Boid:
    """Boid for flocking simulation."""
    
    def __init__(self, x: float, y: float, dim: int = 2):
        self.position = [x, y] if dim == 2 else [x, y, 0]
        self.velocity = [random.gauss(0, 1) for _ in range(dim)]
        self.acceleration = [0.0] * dim
        self.dim = dim
        
        # Flocking parameters
        self.max_speed = 4
        self.max_force = 0.1
        self.perception = 50
    
    def flock(self, boids: List['Boid']):
        """Apply flocking rules."""
        separation = self._separate(boids)
        alignment = self._align(boids)
        cohesion = self._cohere(boids)
        
        # Weight and apply forces
        for i in range(self.dim):
            self.acceleration[i] += separation[i] * 1.5
            self.acceleration[i] += alignment[i] * 1.0
            self.acceleration[i] += cohesion[i] * 1.0
    
    def _separate(self, boids: List['Boid']) -> List[float]:
        """Separation: avoid crowding."""
        steer = [0.0] * self.dim
        count = 0
        
        for other in boids:
            d = self._distance(other)
            if 0 < d < self.perception / 2:
                diff = [self.position[i] - other.position[i] for i in range(self.dim)]
                diff = [d / (x + 1e-8) for x in diff]  # Normalize
                steer = [s + d for s, d in zip(steer, diff)]
                count += 1
        
        if count > 0:
            steer = [s / count for s in steer]
            steer = self._limit(steer, self.max_speed)
            steer = [s - v for s, v in zip(steer, self.velocity)]
            steer = self._limit(steer, self.max_force)
        
        return steer
    
    def _align(self, boids: List['Boid']) -> List[float]:
        """Alignment: steer towards average velocity."""
        sum_vel = [0.0] * self.dim
        count = 0
        
        for other in boids:
            d = self._distance(other)
            if 0 < d < self.perception:
                sum_vel = [s + v for s, v in zip(sum_vel, other.velocity)]
                count += 1
        
        if count > 0:
            avg_vel = [s / count for s in sum_vel]
            avg_vel = self._limit(avg_vel, self.max_speed)
            steer = [a - v for a, v in zip(avg_vel, self.velocity)]
            return self._limit(steer, self.max_force)
        
        return [0.0] * self.dim
    
    def _cohere(self, boids: List['Boid']) -> List[float]:
        """Cohesion: steer towards center of mass."""
        sum_pos = [0.0] * self.dim
        count = 0
        
        for other in boids:
            d = self._distance(other)
            if 0 < d < self.perception:
                sum_pos = [s + p for s, p in zip(sum_pos, other.position)]
                count += 1
        
        if count > 0:
            center = [s / count for s in sum_pos]
            desired = [c - p for c, p in zip(center, self.position)]
            desired = self._limit(desired, self.max_speed)
            steer = [d - v for d, v in zip(desired, self.velocity)]
            return self._limit(steer, self.max_force)
        
        return [0.0] * self.dim
    
    def _distance(self, other: 'Boid') -> float:
        """Calculate distance to another boid."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))
    
    def _limit(self, vector: List[float], max_val: float) -> List[float]:
        """Limit vector magnitude."""
        mag = math.sqrt(sum(x ** 2 for x in vector))
        if mag > max_val:
            return [x / mag * max_val for x in vector]
        return vector
    
    def update(self):
        """Update position."""
        for i in range(self.dim):
            self.velocity[i] += self.acceleration[i]
            self.velocity[i] = max(-self.max_speed, min(self.max_speed, self.velocity[i]))
            self.position[i] += self.velocity[i]
            self.acceleration[i] = 0


class Flock:
    """Flocking simulation."""
    
    def __init__(self, n_boids: int = 100, width: float = 800, height: float = 600):
        self.boids = [Boid(random.uniform(0, width), random.uniform(0, height)) 
                     for _ in range(n_boids)]
        self.width = width
        self.height = height
    
    def update(self):
        """Update all boids."""
        for boid in self.boids:
            boid.flock(self.boids)
            boid.update()
            
            # Wrap around edges
            boid.position[0] = boid.position[0] % self.width
            boid.position[1] = boid.position[1] % self.height
    
    def visualize(self, width: int = 60, height: int = 20) -> str:
        """ASCII visualization."""
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        for boid in self.boids:
            x = int(boid.position[0] / self.width * width) % width
            y = int(boid.position[1] / self.height * height) % height
            grid[y][x] = '*'
        
        return '\n'.join(''.join(row) for row in grid)


class CollectiveIntelligence:
    """Collective decision making through swarm voting."""
    
    def __init__(self, n_agents: int = 50):
        self.n_agents = n_agents
        self.agents: List[Dict] = []
        self.initialize()
    
    def initialize(self):
        """Initialize agents with random beliefs."""
        self.agents = [
            {
                'belief': random.random(),
                'confidence': random.random(),
                'influence': random.random()
            }
            for _ in range(self.n_agents)
        ]
    
    def vote(self, options: List[str]) -> Tuple[str, float]:
        """Collective voting."""
        votes = {opt: 0.0 for opt in options}
        
        for agent in self.agents:
            # Weighted random choice based on belief
            choice = random.choices(options, weights=[agent['belief'], 1-agent['belief']])[0]
            votes[choice] += agent['confidence'] * agent['influence']
        
        # Normalize
        total = sum(votes.values())
        if total > 0:
            votes = {k: v/total for k, v in votes.items()}
        
        winner = max(votes, key=votes.get)
        return winner, votes[winner]
    
    def update_beliefs(self, outcome: str, learning_rate: float = 0.1):
        """Update beliefs based on outcome."""
        for agent in self.agents:
            # Reinforce if correct
            if random.random() < agent['confidence']:
                agent['belief'] += learning_rate * (1 - agent['belief'])
            else:
                agent['belief'] -= learning_rate * agent['belief']
            
            agent['belief'] = max(0, min(1, agent['belief']))


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Swarm Intelligence")
    print("Particle Swarms × Flocking × Collective Intelligence")
    print("=" * 70)
    
    # PSO
    print("\n1. Particle Swarm Optimization")
    pso = ParticleSwarmOptimizer(n_particles=20, dim=5)
    
    def sphere(x):
        return sum(xi ** 2 for xi in x)
    
    best = pso.optimize(sphere, iterations=50)
    print(f"\nBest solution: {[f'{x:.4f}' for x in best]}")
    print(f"Best fitness: {pso.global_best_fitness:.6f}")
    
    # Flocking
    print("\n2. Flocking Simulation")
    flock = Flock(n_boids=50, width=100, height=30)
    
    print("Initial state:")
    print(flock.visualize())
    
    for _ in range(5):
        flock.update()
    
    print("\nAfter 5 updates:")
    print(flock.visualize())
    
    # Collective Intelligence
    print("\n3. Collective Intelligence")
    ci = CollectiveIntelligence(n_agents=100)
    
    options = ['option_a', 'option_b']
    for round in range(5):
        winner, confidence = ci.vote(options)
        ci.update_beliefs(winner)
        print(f"Round {round+1}: {winner} (confidence: {confidence:.2f})")
    
    print("\n✨ Swarm intelligence")
    print("✨ Emergent behavior from simple rules")
    print("✨ Collective decision making")
    print("✨ Nature-inspired optimization")
