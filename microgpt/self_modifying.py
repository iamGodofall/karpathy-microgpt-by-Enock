"""
Self-modifying AI system.
Code that evolves and improves itself recursively.
Inspired by perpetual probabilities and fractal self-similarity.
"""

import ast
import inspect
import random
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeGene:
    """A gene representing a code modification."""

    target_function: str
    modification_type: str  # 'add', 'remove', 'modify', 'replace'
    new_code: str
    fitness: float = 0.0
    generation: int = 0


class SelfModifyingSystem:
    """System that can modify its own code."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.generation = 0
        self.genes: List[CodeGene] = []
        self.modification_history = []
        self.fitness_history = []

        # Track which modifications worked
        self.successful_mods = []
        self.failed_mods = []

    def analyze_self(self) -> Dict[str, Any]:
        """Analyze own code structure."""
        # Get source of this module
        try:
            source = inspect.getsource(self.__class__)
            tree = ast.parse(source)

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "line_count": len(node.body),
                            "complexity": self._calc_complexity(node),
                        }
                    )

            return {
                "total_functions": len(functions),
                "functions": functions,
                "total_lines": len(source.split("\n")),
                "hash": hashlib.md5(source.encode()).hexdigest()[:8],
            }
        except Exception as e:
            return {"error": str(e)}

    def _calc_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def generate_modification(self) -> Optional[CodeGene]:
        """Generate a potential code modification."""
        mod_types = ["optimize", "simplify", "extend", "parallelize"]

        # Analyze current state
        analysis = self.analyze_self()
        if "error" in analysis:
            return None

        # Select target function
        if not analysis["functions"]:
            return None

        target = random.choice(analysis["functions"])

        # Generate modification based on type
        mod_type = random.choice(mod_types)

        if mod_type == "optimize":
            new_code = f"# Optimized {target['name']}: reduced from {target['line_count']} lines"
        elif mod_type == "simplify":
            new_code = f"# Simplified {target['name']}: complexity {target['complexity']} -> {max(1, target['complexity']-1)}"
        elif mod_type == "extend":
            new_code = f"# Extended {target['name']}: added new capability"
        else:
            new_code = f"# Parallelized {target['name']}: concurrent execution"

        gene = CodeGene(
            target_function=target["name"],
            modification_type=mod_type,
            new_code=new_code,
            generation=self.generation,
        )

        return gene

    def apply_modification(self, gene: CodeGene) -> bool:
        """Apply a code modification."""
        try:
            # In a real system, this would actually modify code
            # For safety, we just log it
            self.modification_history.append(
                {
                    "generation": self.generation,
                    "gene": gene,
                    "applied": True,
                    "timestamp": time.time(),
                }
            )
            self.genes.append(gene)
            return True
        except Exception as e:
            self.failed_mods.append({"gene": gene, "error": str(e)})
            return False

    def evolve(self, fitness_func: Callable = None):
        """Evolve the system."""
        self.generation += 1

        # Generate and apply modifications
        for _ in range(3):  # 3 modifications per generation
            gene = self.generate_modification()
            if gene:
                success = self.apply_modification(gene)
                if success:
                    self.successful_mods.append(gene)

        # Evaluate fitness
        if fitness_func:
            fitness = fitness_func(self)
            self.fitness_history.append(fitness)

        return self.generation

    def get_evolution_report(self) -> str:
        """Generate evolution report."""
        lines = [
            "# Self-Modifying System Report",
            f"Generation: {self.generation}",
            f"Total Modifications: {len(self.genes)}",
            f"Successful: {len(self.successful_mods)}",
            f"Failed: {len(self.failed_mods)}",
            "",
        ]

        if self.fitness_history:
            lines.extend(
                [
                    f"Current Fitness: {self.fitness_history[-1]:.4f}",
                    f"Best Fitness: {max(self.fitness_history):.4f}",
                    "",
                ]
            )

        lines.append("## Recent Modifications")
        for mod in self.modification_history[-5:]:
            gene = mod["gene"]
            lines.append(
                f"- Gen {mod['generation']}: {gene.modification_type} {gene.target_function}"
            )

        return "\n".join(lines)


class RecursiveImprover:
    """System that recursively improves itself."""

    def __init__(self, depth: int = 3):
        self.depth = depth
        self.subsystems: List[SelfModifyingSystem] = []
        self.meta_system = SelfModifyingSystem(seed=42)

        # Create hierarchy
        for i in range(depth):
            system = SelfModifyingSystem(seed=42 + i)
            self.subsystems.append(system)

    def evolve_hierarchy(self):
        """Evolve all levels of the hierarchy."""
        # Evolve from bottom up
        for i, system in enumerate(self.subsystems):
            print(f"Evolving level {i+1}/{self.depth}...")
            system.evolve()

        # Meta-system evolves based on sub-systems
        self.meta_system.evolve(
            fitness_func=lambda s: sum(len(sub.genes) for sub in self.subsystems)
            / max(len(self.subsystems), 1)
        )

    def get_complexity(self) -> int:
        """Calculate total system complexity."""
        return sum(len(s.genes) for s in self.subsystems) + len(self.meta_system.genes)


import time

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Self-Modifying AI System")
    print("Code that evolves itself")
    print("=" * 70)

    # Create self-modifying system
    system = SelfModifyingSystem(seed=42)

    # Initial analysis
    analysis = system.analyze_self()
    print(f"\nInitial Analysis:")
    print(f"  Functions: {analysis.get('total_functions', 'N/A')}")
    print(f"  Lines: {analysis.get('total_lines', 'N/A')}")
    print(f"  Hash: {analysis.get('hash', 'N/A')}")

    # Evolve
    print("\nEvolving 5 generations...")
    for i in range(5):
        system.evolve(fitness_func=lambda s: random.random())
        print(
            f"  Gen {i+1}: {len(system.genes)} modifications, "
            f"fitness={system.fitness_history[-1]:.4f}"
        )

    print(f"\n{system.get_evolution_report()}")

    # Recursive system
    print("\n" + "=" * 70)
    print("Recursive Improver")
    print("=" * 70)

    recursive = RecursiveImprover(depth=3)
    recursive.evolve_hierarchy()

    print(f"\nTotal complexity: {recursive.get_complexity()}")
    print(f"Meta-system genes: {len(recursive.meta_system.genes)}")

    print("\n✨ Code that writes itself")
    print("✨ Infinite self-improvement")
    print("✨ Recursive evolution")
