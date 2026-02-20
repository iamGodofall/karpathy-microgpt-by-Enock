"""
Model registry for microgpt.
Manage multiple models, versions, and deployments.
"""

import json
import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    model_id: str
    created_at: float
    config: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    tags: List[str]
    status: str = "active"  # active, deprecated, archived
    description: str = ""


class ModelRegistry:
    """Central registry for all models."""
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.models: Dict[str, List[ModelVersion]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load existing registry."""
        registry_file = self.registry_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.models[model_id] = [ModelVersion(**v) for v in versions]
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_dir / "registry.json"
        data = {
            model_id: [asdict(v) for v in versions]
            for model_id, versions in self.models.items()
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def register_model(self, model_id: str, config: Dict[str, Any],
                      metrics: Dict[str, float], artifacts: List[str],
                      tags: List[str] = None, description: str = "") -> ModelVersion:
        """Register a new model version."""
        # Generate version
        version_hash = hashlib.md5(
            f"{model_id}_{time.time()}".encode()
        ).hexdigest()[:8]
        
        version = ModelVersion(
            version=version_hash,
            model_id=model_id,
            created_at=time.time(),
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            tags=tags or [],
            description=description
        )
        
        if model_id not in self.models:
            self.models[model_id] = []
        
        self.models[model_id].append(version)
        self._save_registry()
        
        print(f"Registered {model_id} version {version_hash}")
        return version
    
    def get_model(self, model_id: str, version: str = None) -> Optional[ModelVersion]:
        """Get a specific model version."""
        if model_id not in self.models:
            return None
        
        if version:
            for v in self.models[model_id]:
                if v.version == version:
                    return v
            return None
        else:
            # Return latest
            return self.models[model_id][-1]
    
    def list_models(self) -> List[str]:
        """List all model IDs."""
        return list(self.models.keys())
    
    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return self.models.get(model_id, [])
    
    def get_best_version(self, model_id: str, metric: str = 'loss') -> Optional[ModelVersion]:
        """Get best version by metric."""
        versions = self.models.get(model_id, [])
        if not versions:
            return None
        
        return min(versions, key=lambda v: v.metrics.get(metric, float('inf')))
    
    def compare_versions(self, model_id: str, metric: str = 'loss') -> Dict[str, Any]:
        """Compare all versions of a model."""
        versions = self.models.get(model_id, [])
        if not versions:
            return {}
        
        sorted_versions = sorted(versions, key=lambda v: v.metrics.get(metric, float('inf')))
        
        return {
            'model_id': model_id,
            'metric': metric,
            'best_version': sorted_versions[0],
            'versions': sorted_versions,
            'improvement': (
                (sorted_versions[-1].metrics.get(metric, float('inf')) - 
                 sorted_versions[0].metrics.get(metric, float('inf'))) /
                max(sorted_versions[-1].metrics.get(metric, 1), 1e-8)
            )
        }
    
    def deprecate_version(self, model_id: str, version: str):
        """Deprecate a model version."""
        v = self.get_model(model_id, version)
        if v:
            v.status = "deprecated"
            self._save_registry()
            print(f"Deprecated {model_id} version {version}")
    
    def delete_version(self, model_id: str, version: str):
        """Delete a model version."""
        if model_id in self.models:
            self.models[model_id] = [
                v for v in self.models[model_id] 
                if v.version != version
            ]
            self._save_registry()
            print(f"Deleted {model_id} version {version}")
    
    def search_models(self, tag: str = None, min_metric: float = None,
                     metric_name: str = 'loss') -> List[ModelVersion]:
        """Search models by criteria."""
        results = []
        for model_id, versions in self.models.items():
            for v in versions:
                if tag and tag not in v.tags:
                    continue
                if min_metric is not None and v.metrics.get(metric_name, float('inf')) > min_metric:
                    continue
                results.append(v)
        
        return sorted(results, key=lambda v: v.metrics.get(metric_name, float('inf')))
    
    def generate_report(self, output_path: str = "registry_report.md"):
        """Generate registry report."""
        lines = ["# Model Registry Report\n"]
        lines.append(f"Generated: {datetime.now()}\n")
        
        for model_id, versions in self.models.items():
            lines.append(f"## {model_id}")
            lines.append(f"Total versions: {len(versions)}")
            lines.append(f"Active: {sum(1 for v in versions if v.status == 'active')}")
            
            if versions:
                best = self.get_best_version(model_id)
                lines.append(f"Best version: {best.version}")
                lines.append(f"Best metrics: {best.metrics}")
            
            lines.append("\n### Versions")
            for v in versions:
                lines.append(f"- **{v.version}** ({v.status})")
                lines.append(f"  - Created: {datetime.fromtimestamp(v.created_at)}")
                lines.append(f"  - Metrics: {v.metrics}")
                lines.append(f"  - Tags: {', '.join(v.tags) if v.tags else 'None'}")
                lines.append("")
        
        report = "\n".join(lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")
        return report


class ModelPromoter:
    """Promote models through environments (dev -> staging -> prod)."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.environments = ['dev', 'staging', 'prod']
    
    def promote(self, model_id: str, version: str, 
                from_env: str, to_env: str):
        """Promote model to next environment."""
        v = self.registry.get_model(model_id, version)
        if not v:
            print(f"Model {model_id} version {version} not found")
            return
        
        # Add environment tag
        v.tags = [t for t in v.tags if not t.startswith('env:')]
        v.tags.append(f"env:{to_env}")
        
        self.registry._save_registry()
        print(f"Promoted {model_id}:{version} from {from_env} to {to_env}")
    
    def get_models_in_env(self, env: str) -> List[ModelVersion]:
        """Get all models in an environment."""
        results = []
        for model_id, versions in self.registry.models.items():
            for v in versions:
                if f"env:{env}" in v.tags:
                    results.append(v)
        return results
    
    def rollback(self, model_id: str, env: str):
        """Rollback to previous version in environment."""
        env_models = self.get_models_in_env(env)
        model_versions = [v for v in env_models if v.model_id == model_id]
        
        if len(model_versions) < 2:
            print(f"No previous version to rollback to")
            return
        
        # Sort by time
        sorted_versions = sorted(model_versions, key=lambda v: v.created_at, reverse=True)
        current = sorted_versions[0]
        previous = sorted_versions[1]
        
        # Deprecate current
        self.registry.deprecate_version(model_id, current.version)
        
        # Promote previous
        self.promote(model_id, previous.version, env, env)
        
        print(f"Rolled back {model_id} in {env} to {previous.version}")


# Example usage
if __name__ == "__main__":
    registry = ModelRegistry()
    
    # Register some models
    for i in range(3):
        registry.register_model(
            model_id="gpt-small",
            config={'n_embd': 64, 'n_layer': 2},
            metrics={'loss': 2.0 - i * 0.3, 'accuracy': 0.5 + i * 0.1},
            artifacts=[f"model_v{i}.pkl"],
            tags=["experiment", "baseline"],
            description=f"Baseline model version {i}"
        )
    
    # List models
    print("\nRegistered models:")
    for model_id in registry.list_models():
        print(f"  {model_id}")
    
    # Get best version
    best = registry.get_best_version("gpt-small", "loss")
    print(f"\nBest version: {best.version} with loss {best.metrics['loss']:.4f}")
    
    # Compare versions
    comparison = registry.compare_versions("gpt-small")
    print(f"\nImprovement over versions: {comparison['improvement']*100:.1f}%")
    
    # Promote to production
    promoter = ModelPromoter(registry)
    promoter.promote("gpt-small", best.version, "dev", "prod")
    
    # Generate report
    registry.generate_report()
