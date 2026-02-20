"""
Deployment utilities for microgpt ecosystem.
Handles packaging, environment setup, and deployment configurations.
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str = "microgpt-deployment"
    version: str = "1.0.0"
    python_version: str = "3.8"
    platform: str = "cpu"  # cpu, cuda, metal
    quantization: Optional[str] = None  # int8, int4, None
    container: bool = False
    serverless: bool = False
    
    # Resource limits
    max_memory_gb: float = 4.0
    max_batch_size: int = 1
    max_concurrent: int = 10
    
    # Endpoints
    enable_api: bool = True
    enable_web: bool = True
    enable_websocket: bool = False


class DeploymentManager:
    """Manage deployment of microgpt models."""
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.project_root = Path(".")
    
    def create_deployment_package(self, output_dir: str = "deploy") -> str:
        """Create a deployment package."""
        deploy_dir = Path(output_dir)
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        essential_files = [
            "microgpt.py",
            "model.py",
            "config.py",
            "checkpoint.py",
            "data.py",
            "tokenizers.py",
            "openclaw_enhanced.py",
            "hrm_enhanced.py",
            "unified_integration.py",
            "api_server.py",
            "web_app.py",
            "requirements.txt",
        ]
        
        for file in essential_files:
            src = self.project_root / file
            if src.exists():
                shutil.copy(src, deploy_dir / file)
        
        # Create deployment config
        config_path = deploy_dir / "deployment.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Create startup script
        self._create_startup_script(deploy_dir)
        
        # Create Dockerfile if containerized
        if self.config.container:
            self._create_dockerfile(deploy_dir)
        
        print(f"Deployment package created at {deploy_dir.absolute()}")
        return str(deploy_dir)
    
    def _create_startup_script(self, deploy_dir: Path):
        """Create startup script."""
        script = '''#!/bin/bash
# microgpt Deployment Startup Script

# Load configuration
export MICROGPT_CONFIG=${MICROGPT_CONFIG:-config.yaml}

# Start services
if [ "$ENABLE_API" = "true" ]; then
    echo "Starting API server..."
    python api_server.py &
fi

if [ "$ENABLE_WEB" = "true" ]; then
    echo "Starting web interface..."
    python web_app.py &
fi

# Keep running
wait
'''
        script_path = deploy_dir / "start.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        script_path.chmod(0o755)
        
        # Windows batch file
        batch = '''@echo off
REM microgpt Deployment Startup Script

set MICROGPT_CONFIG=%MICROGPT_CONFIG:~config.yaml%

if "%ENABLE_API%"=="true" (
    echo Starting API server...
    start python api_server.py
)

if "%ENABLE_WEB%"=="true" (
    echo Starting web interface...
    start python web_app.py
)
'''
        batch_path = deploy_dir / "start.bat"
        with open(batch_path, 'w') as f:
            f.write(batch)
    
    def _create_dockerfile(self, deploy_dir: Path):
        """Create Dockerfile."""
        dockerfile = f'''FROM python:{self.config.python_version}-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV MICROGPT_CONFIG=/app/config.yaml
ENV ENABLE_API=true
ENV ENABLE_WEB=true

# Expose ports
EXPOSE 5000 8000

# Start
CMD ["./start.sh"]
'''
        dockerfile_path = deploy_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile)
        
        # Create docker-compose
        compose = '''version: '3.8'

services:
  microgpt:
    build: .
    ports:
      - "5000:5000"
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    environment:
      - ENABLE_API=true
      - ENABLE_WEB=true
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
'''
        compose_path = deploy_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose)
    
    def create_serverless_config(self, provider: str = "aws") -> Dict:
        """Create serverless deployment configuration."""
        if provider == "aws":
            return self._aws_lambda_config()
        elif provider == "gcp":
            return self._gcp_cloud_functions_config()
        elif provider == "azure":
            return self._azure_functions_config()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _aws_lambda_config(self) -> Dict:
        """AWS Lambda configuration."""
        return {
            "service": "microgpt-lambda",
            "provider": {
                "name": "aws",
                "runtime": f"python{self.config.python_version}",
                "memorySize": 3008,
                "timeout": 30,
            },
            "functions": {
                "generate": {
                    "handler": "handler.generate",
                    "events": [{"http": {"path": "generate", "method": "post"}}],
                },
                "chat": {
                    "handler": "handler.chat",
                    "events": [{"http": {"path": "chat", "method": "post"}}],
                }
            }
        }
    
    def _gcp_cloud_functions_config(self) -> Dict:
        """GCP Cloud Functions configuration."""
        return {
            "runtime": f"python{self.config.python_version}",
            "entryPoint": "generate",
            "memory": "2GB",
            "timeout": "60s",
            "environmentVariables": {
                "MICROGPT_CONFIG": "config.yaml"
            }
        }
    
    def _azure_functions_config(self) -> Dict:
        """Azure Functions configuration."""
        return {
            "scriptFile": "handler.py",
            "bindings": [
                {
                    "authLevel": "function",
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["post"]
                },
                {
                    "type": "http",
                    "direction": "out",
                    "name": "$return"
                }
            ]
        }
    
    def generate_handler(self, provider: str = "aws") -> str:
        """Generate serverless handler code."""
        handler = f'''import json
import os
import sys

# Add deployment directory to path
sys.path.insert(0, os.path.dirname(__file__))

from unified_integration import UnifiedAI, UnifiedConfig
from config import Config

# Load configuration
config = Config.from_yaml(os.environ.get('MICROGPT_CONFIG', 'config.yaml'))
ai = UnifiedAI(config)

def generate(event, context):
    """AWS Lambda handler for generation."""
    try:
        if isinstance(event, dict) and 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        prompt = body.get('prompt', '')
        max_length = body.get('max_length', 50)
        temperature = body.get('temperature', 0.7)
        
        result = ai.chat(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        return {{
            'statusCode': 200,
            'body': json.dumps(result),
            'headers': {{
                'Content-Type': 'application/json'
            }}
        }}
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}

def chat(event, context):
    """AWS Lambda handler for chat."""
    return generate(event, context)
'''
        return handler
    
    def create_environment_check(self) -> Dict[str, bool]:
        """Check deployment environment readiness."""
        checks = {
            'python_version': self._check_python_version(),
            'disk_space': self._check_disk_space(),
            'memory': self._check_memory(),
            'dependencies': self._check_dependencies(),
        }
        return checks
    
    def _check_python_version(self) -> bool:
        """Check Python version compatibility."""
        import sys
        version = sys.version_info
        return version.major == 3 and version.minor >= 8
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        return free_gb >= 1.0  # At least 1GB free
    
    def _check_memory(self) -> bool:
        """Check available memory."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available >= 2 * (1024**3)  # At least 2GB available
        except ImportError:
            return True  # Can't check, assume OK
    
    def _check_dependencies(self) -> bool:
        """Check if dependencies are installed."""
        try:
            import microgpt
            import model
            return True
        except ImportError:
            return False


# Example usage
if __name__ == "__main__":
    # Create deployment package
    config = DeploymentConfig(
        name="microgpt-prod",
        container=True,
        enable_api=True,
        enable_web=True
    )
    
    manager = DeploymentManager(config)
    
    # Check environment
    print("Environment Check:")
    checks = manager.create_environment_check()
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Create package
    print("\nCreating deployment package...")
    deploy_dir = manager.create_deployment_package()
    
    # Generate serverless handler
    handler_code = manager.generate_handler("aws")
    handler_path = Path(deploy_dir) / "handler.py"
    with open(handler_path, 'w') as f:
        f.write(handler_code)
    
    print(f"\nDeployment ready at: {deploy_dir}")
    print("To deploy:")
    print(f"  cd {deploy_dir}")
    print("  docker-compose up")
