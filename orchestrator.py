"""
Master orchestrator for the microgpt ecosystem.
Coordinates all components for seamless operation.
"""

import os
import json
import signal
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    # Component enablement
    enable_training: bool = True
    enable_api: bool = True
    enable_web: bool = False
    enable_monitoring: bool = True
    
    # Paths
    data_path: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Training
    auto_train: bool = False
    training_config_path: str = "config.yaml"
    
    # Serving
    api_port: int = 8000
    web_port: int = 5000
    
    # Monitoring
    monitor_interval: int = 60


class MicrogptOrchestrator:
    """Master orchestrator for all microgpt components."""
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.components: Dict[str, Any] = {}
        self.running = False
        self._threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self):
        """Initialize all components."""
        print("=" * 70)
        print("Initializing microgpt Orchestrator")
        print("=" * 70)
        
        # Create directories
        for dir_path in [self.config.data_path, self.config.checkpoint_dir, self.config.log_dir]:
            Path(dir_path).mkdir(exist_ok=True)
            print(f"✓ Directory ready: {dir_path}")
        
        # Load configuration
        if os.path.exists(self.config.training_config_path):
            from config import Config
            self.config_obj = Config.from_yaml(self.config.training_config_path)
            print(f"✓ Configuration loaded from {self.config.training_config_path}")
        else:
            from config import DEFAULT_CONFIG
            self.config_obj = DEFAULT_CONFIG
            print("✓ Using default configuration")
        
        # Initialize components based on config
        if self.config.enable_monitoring:
            self._init_monitoring()
        
        if self.config.enable_training:
            self._init_training()
        
        if self.config.enable_api:
            self._init_api()
        
        if self.config.enable_web:
            self._init_web()
        
        print("=" * 70)
        print("All components initialized")
        print("=" * 70)
    
    def _init_monitoring(self):
        """Initialize monitoring."""
        from monitoring import ModelMonitor
        self.components['monitor'] = ModelMonitor()
        print("✓ Monitoring initialized")
    
    def _init_training(self):
        """Initialize training components."""
        from model import GPT
        from trainer import Trainer
        from checkpoint import CheckpointManager
        
        model = GPT(self.config_obj.model)
        trainer = Trainer(model, self.config_obj.training)
        checkpoint_mgr = CheckpointManager(self.config.checkpoint_dir)
        
        self.components['model'] = model
        self.components['trainer'] = trainer
        self.components['checkpoint'] = checkpoint_mgr
        
        if self.config.enable_monitoring and 'monitor' in self.components:
            trainer.set_monitor(self.components['monitor'])
        
        print("✓ Training components initialized")
    
    def _init_api(self):
        """Initialize API server."""
        # API will be started in run()
        print("✓ API server ready to start")
    
    def _init_web(self):
        """Initialize web interface."""
        # Web will be started in run()
        print("✓ Web interface ready to start")
    
    def run(self):
        """Run the orchestrator."""
        self.running = True
        print("\n" + "=" * 70)
        print("Starting microgpt Services")
        print("=" * 70)
        
        # Start monitoring
        if self.config.enable_monitoring and 'monitor' in self.components:
            self.components['monitor'].start(self.config.monitor_interval)
        
        # Auto-train if enabled
        if self.config.auto_train and 'trainer' in self.components:
            self._start_training()
        
        # Start API server
        if self.config.enable_api:
            self._start_api()
        
        # Start web interface
        if self.config.enable_web:
            self._start_web()
        
        # Keep running
        print("\n" + "=" * 70)
        print("All services running. Press Ctrl+C to stop.")
        print("=" * 70)
        
        try:
            while self.running and not self._shutdown_event.is_set():
                self._shutdown_event.wait(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
    
    def _start_training(self):
        """Start training in background."""
        from data import Dataset, CharTokenizer
        
        def train_worker():
            # Load or create dataset
            data_path = os.path.join(self.config.data_path, "train.txt")
            if os.path.exists(data_path):
                with open(data_path) as f:
                    texts = [line.strip() for line in f if line.strip()]
            else:
                # Create sample data
                texts = ["hello world", "test data"] * 100
            
            tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")
            dataset = Dataset(texts, tokenizer, self.config_obj.model.block_size)
            
            # Train
            trainer = self.components['trainer']
            history = trainer.train(dataset)
            
            # Save final checkpoint
            checkpoint_mgr = self.components['checkpoint']
            state_dict = {k: v for k, v in self.components['model'].state_dict().items()}
            checkpoint_mgr.save_best(state_dict, self.config_obj, 
                                      len(history['train_losses']), 
                                      history['train_losses'][-1])
            
            print("Training completed")
        
        t = threading.Thread(target=train_worker, daemon=True)
        t.start()
        self._threads.append(t)
        print("✓ Training started in background")
    
    def _start_api(self):
        """Start API server."""
        def api_worker():
            from api_server import app
            app.run(host='0.0.0.0', port=self.config.api_port, debug=False)
        
        t = threading.Thread(target=api_worker, daemon=True)
        t.start()
        self._threads.append(t)
        print(f"✓ API server started on port {self.config.api_port}")
    
    def _start_web(self):
        """Start web interface."""
        def web_worker():
            from web_app import app
            app.run(host='0.0.0.0', port=self.config.web_port, debug=False)
        
        t = threading.Thread(target=web_worker, daemon=True)
        t.start()
        self._threads.append(t)
        print(f"✓ Web interface started on port {self.config.web_port}")
    
    def shutdown(self):
        """Graceful shutdown."""
        print("\n" + "=" * 70)
        print("Shutting down...")
        print("=" * 70)
        
        self.running = False
        self._shutdown_event.set()
        
        # Stop monitoring
        if 'monitor' in self.components:
            self.components['monitor'].stop()
            self.components['monitor'].save_report()
            print("✓ Monitoring stopped")
        
        # Wait for threads
        for t in self._threads:
            t.join(timeout=5)
        
        print("✓ All services stopped")
        print("=" * 70)
    
    def status(self) -> Dict[str, Any]:
        """Get current status."""
        status = {
            'running': self.running,
            'components': list(self.components.keys()),
            'config': asdict(self.config)
        }
        
        if 'monitor' in self.components:
            status['metrics'] = self.components['monitor'].metrics.all_stats()
            status['health'] = self.components['monitor'].health.status
        
        return status
    
    def save_state(self, path: str = "orchestrator_state.json"):
        """Save orchestrator state."""
        state = self.status()
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"State saved to {path}")


# CLI interface
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='microgpt Orchestrator')
    parser.add_argument('--config', default='orchestrator.json', help='Orchestrator config file')
    parser.add_argument('--train', action='store_true', help='Auto-start training')
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring')
    
    args = parser.parse_args()
    
    # Load or create config
    if os.path.exists(args.config):
        with open(args.config) as f:
            config_dict = json.load(f)
        config = OrchestratorConfig(**config_dict)
    else:
        config = OrchestratorConfig(
            enable_training=args.train,
            enable_api=args.api,
            enable_web=args.web,
            enable_monitoring=args.monitor
        )
        # Save for future runs
        with open(args.config, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    # Run orchestrator
    orchestrator = MicrogptOrchestrator(config)
    orchestrator.initialize()
    orchestrator.run()


if __name__ == "__main__":
    main()
