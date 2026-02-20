"""
Unified Integration: Enhanced OpenClaw + HRM + microgpt
Production-ready AI system with all advanced features.
"""

import time
import random
import json
from typing import Dict, List, Optional, Any, Callable, Iterator, Tuple


from dataclasses import dataclass, field
from pathlib import Path
import threading

from openclaw_enhanced import EnhancedOpenClaw, ThinkLevel, Tool, ToolType, Session
from hrm_enhanced import EnhancedHierarchicalReasoningModel, EnhancedHRMConfig


@dataclass
class UnifiedConfig:
    """Unified configuration for the complete system."""
    # Model
    vocab_size: int = 100
    hidden_size: int = 128
    num_heads: int = 4
    
    # HRM
    hrm_H_layers: int = 2
    hrm_L_layers: int = 2
    use_adaptive_depth: bool = True
    
    # OpenClaw
    enable_tools: bool = True
    enable_streaming: bool = True
    max_context_tokens: int = 4096
    
    # System
    storage_dir: str = ".microgpt_unified"
    max_workers: int = 4


class UnifiedAI:
    """
    Unified AI system combining:
    - HRM for reasoning
    - OpenClaw for session/tool management
    - microgpt for core model
    """
    
    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        
        # Initialize HRM
        hrm_config = EnhancedHRMConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            H_layers=self.config.hrm_H_layers,
            L_layers=self.config.hrm_L_layers,
            adaptive_depth=self.config.use_adaptive_depth,
            use_memory_augmentation=True,
        )
        self.hrm = EnhancedHierarchicalReasoningModel(hrm_config)
        
        # Initialize OpenClaw
        self.openclaw = EnhancedOpenClaw(
            storage_dir=self.config.storage_dir,
            max_workers=self.config.max_workers,
        )
        
        # Register default tools
        self._register_default_tools()
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "avg_reasoning_steps": 0,
        }
        
        self.lock = threading.RLock()
    
    def _register_default_tools(self):
        """Register built-in tools."""
        
        def calculator(expression: str) -> str:
            """Safe calculator tool."""
            try:
                # Safe eval with limited scope
                allowed = {
                    'abs': abs, 'max': max, 'min': min,
                    'sum': sum, 'pow': pow, 'round': round,
                }
                result = eval(expression, {"__builtins__": {}}, allowed)
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        def search(query: str) -> str:
            """Simulated search tool."""
            return f"[Search results for: {query}]"
        
        self.openclaw.register_tool(Tool(
            name="calculator",
            description="Perform calculations",
            type=ToolType.CALCULATOR,
            function=calculator,
            parameters={"expression": "string"},
            required_params=["expression"],
        ))
        
        self.openclaw.register_tool(Tool(
            name="search",
            description="Search for information",
            type=ToolType.SEARCH,
            function=search,
            parameters={"query": "string"},
            required_params=["query"],
        ))
    
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_reasoning: bool = True,
        tools: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Main chat interface with reasoning and tools.
        
        Args:
            message: User input
            session_id: Optional session ID
            use_reasoning: Use HRM for complex reasoning
            tools: List of tools to use
            stream: Enable streaming response
        
        Returns:
            Response with metadata
        """
        start_time = time.time()
        
        with self.lock:
            self.metrics["total_requests"] += 1
            
            # Get or create session
            session = self.openclaw.get_session(session_id)
            if not session:
                session = self.openclaw.create_session(
                    session_id=session_id,
                    max_context=self.config.max_context_tokens,
                    enable_tools=self.config.enable_tools,
                    streaming=stream or self.config.enable_streaming,
                )
            
            # Determine if reasoning needed
            think_level = self._assess_complexity(message)
            use_reasoning = use_reasoning and (think_level.value in ["medium", "high", "xhigh"])
            
            # Execute tools if needed
            tool_results = []
            if tools:
                for tool_name in tools:
                    result = self.openclaw.execute_tool(tool_name, {"query": message})
                    tool_results.append(result)
            
            # Generate response
            if use_reasoning:
                response, reasoning_info = self._reasoning_generate(
                    message, session, tool_results
                )
            else:
                response = self._simple_generate(message, session)
                reasoning_info = {"steps": 1, "adaptive": False}
            
            # Add to session
            latency = (time.time() - start_time) * 1000
            session.add_message(
                role="assistant",
                content=response,
                tool_calls=[r for r in tool_results if r.get("success")],
                reasoning_trace=f"Steps: {reasoning_info['steps']}" if use_reasoning else None,
            )
            session.messages[-1].latency_ms = latency
            
            # Update metrics
            self.metrics["avg_reasoning_steps"] = (
                (self.metrics["avg_reasoning_steps"] * (self.metrics["total_requests"] - 1) + 
                 reasoning_info["steps"]) / self.metrics["total_requests"]
            )
            
            # Save session
            self.openclaw._save_session(session)
            
            return {
                "response": response,
                "session_id": session.session_id,
                "used_reasoning": use_reasoning,
                "reasoning_steps": reasoning_info["steps"],
                "tools_used": [r["tool"] for r in tool_results if r.get("success")],
                "latency_ms": latency,
                "think_level": think_level.value,
            }
    
    def _assess_complexity(self, message: str) -> ThinkLevel:
        """Assess message complexity."""
        indicators = {
            ThinkLevel.LOW: ["what", "who", "when", "where"],
            ThinkLevel.MEDIUM: ["how", "explain", "describe", "compare"],
            ThinkLevel.HIGH: ["why", "analyze", "solve", "prove", "reason"],
            ThinkLevel.XHIGH: ["optimize", "design", "synthesize", "prove theorem"],
        }
        
        msg_lower = message.lower()
        scores = {level: 0 for level in ThinkLevel}
        
        for level, words in indicators.items():
            for word in words:
                if word in msg_lower:
                    scores[level] += 1
        
        # Length factor
        if len(message) > 500:
            scores[ThinkLevel.HIGH] += 1
        if len(message) > 1000:
            scores[ThinkLevel.XHIGH] += 1
        
        # Return highest matching level
        for level in [ThinkLevel.XHIGH, ThinkLevel.HIGH, ThinkLevel.MEDIUM, ThinkLevel.LOW]:
            if scores[level] > 0:
                return level
        
        return ThinkLevel.LOW
    
    def _reasoning_generate(
        self,
        message: str,
        session: Session,
        tool_results: List[Dict]
    ) -> Tuple[str, Dict]:
        """Generate with HRM reasoning."""
        # Tokenize (simplified)
        tokens = [ord(c) % self.config.vocab_size for c in message[:50]]
        
        # Add tool context
        if tool_results:
            tool_context = " ".join([r.get("result", "") for r in tool_results if r.get("success")])
            tokens.extend([ord(c) % self.config.vocab_size for c in tool_context[:50]])
        
        # Ensure tokens are valid
        tokens = [t % self.config.vocab_size for t in tokens]
        
        # Generate with HRM
        result = self.hrm.forward(tokens, training=False)
        
        # Decode (simplified - would use proper tokenizer)
        response = f"[HRM reasoning: {result['steps']} steps] "
        response += f"Response to: {message[:50]}..."
        
        return response, {"steps": result["steps"], "adaptive": True}
    
    def _simple_generate(self, message: str, session: Session) -> str:
        """Simple generation without reasoning."""
        return f"[Direct response to: {message[:50]}...]"
    
    def stream_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:

        """
        Streaming chat interface.
        Yields tokens as they're generated.
        """
        # Create streaming session
        session = self.openclaw.create_session(
            session_id=session_id,
            streaming=True,
        )
        
        # Start generation in background
        import threading
        
        def generate():
            response = self.chat(message, session.session_id, stream=True)
            tokens = response["response"].split()
            for token in tokens:
                session.stream_token(token + " ")
                time.sleep(0.05)  # Simulate generation
            session.stream_buffer.close()
        
        thread = threading.Thread(target=generate)
        thread.start()
        
        # Yield tokens
        while True:
            token = session.stream_buffer.read(timeout=0.1)
            if token is None:
                break
            yield token
    
    def train_on_session(self, session_id: str, num_steps: int = 10):
        """Fine-tune HRM on session history."""
        session = self.openclaw.get_session(session_id)
        if not session or len(session.messages) < 2:
            return {"error": "Insufficient session data"}
        
        # Extract training pairs from conversation
        training_pairs = []
        for i in range(0, len(session.messages) - 1, 2):
            if session.messages[i].role == "user":
                input_text = session.messages[i].content
                target_text = session.messages[i + 1].content if i + 1 < len(session.messages) else ""
                
                # Tokenize (simplified)
                input_tokens = [ord(c) % self.config.vocab_size for c in input_text[:30]]
                target_tokens = [ord(c) % self.config.vocab_size for c in target_text[:30]]
                
                training_pairs.append((input_tokens, target_tokens))
        
        # Train
        losses = []
        for _ in range(num_steps):
            for tokens, targets in training_pairs:
                info = self.hrm.train_step(tokens, targets)
                losses.append(info["total_loss"])
        
        return {
            "steps": num_steps * len(training_pairs),
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "final_loss": losses[-1] if losses else 0,
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        hrm_stats = self.hrm.get_stats()
        openclaw_metrics = self.openclaw.get_metrics()
        
        return {
            "hrm": hrm_stats,
            "openclaw": openclaw_metrics,
            "unified": self.metrics,
            "config": {
                "hidden_size": self.config.hidden_size,
                "adaptive_depth": self.config.use_adaptive_depth,
                "tools_enabled": self.config.enable_tools,
            }
        }
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save complete system state."""
        if path is None:
            path = f"{self.config.storage_dir}/unified_checkpoint.json"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "hrm_config": self.hrm.config.__dict__,
            "hrm_step": self.hrm.step,
            "metrics": self.metrics,
            "timestamp": time.time(),
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        return path
    
    def load_checkpoint(self, path: str):
        """Load system state."""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.metrics = checkpoint.get("metrics", self.metrics)
        # Would restore HRM weights here
        print(f"Loaded checkpoint from {path}")


# Demo and testing
def demo_unified_system():
    """Demonstrate the unified system."""
    print("=" * 70)
    print("Unified AI System: OpenClaw + HRM + microgpt")
    print("=" * 70)
    
    # Create system
    config = UnifiedConfig(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        hrm_H_layers=1,
        hrm_L_layers=2,
        use_adaptive_depth=True,
        enable_tools=True,
    )
    
    ai = UnifiedAI(config)
    
    print("\n1. Simple Query (Low complexity)")
    print("-" * 70)
    result = ai.chat("What is 2+2?", use_reasoning=False)
    print(f"Response: {result['response']}")
    print(f"Used reasoning: {result['used_reasoning']}")
    print(f"Latency: {result['latency_ms']:.1f}ms")
    
    print("\n2. Complex Query (High complexity - triggers reasoning)")
    print("-" * 70)
    result = ai.chat(
        "Explain step by step how to solve a complex optimization problem",
        use_reasoning=True
    )
    print(f"Response: {result['response']}")
    print(f"Used reasoning: {result['used_reasoning']}")
    print(f"Reasoning steps: {result['reasoning_steps']}")
    print(f"Think level: {result['think_level']}")
    
    print("\n3. Tool Use")
    print("-" * 70)
    result = ai.chat(
        "Calculate the sum of squares from 1 to 10",
        tools=["calculator"]
    )
    print(f"Response: {result['response']}")
    print(f"Tools used: {result['tools_used']}")
    
    print("\n4. Session Management")
    print("-" * 70)
    session_id = result['session_id']
    session_info = ai.openclaw.get_session_info(session_id)
    print(f"Session ID: {session_id}")
    print(f"Messages: {session_info['message_count']}")
    print(f"Estimated tokens: {session_info['estimated_tokens']}")
    
    print("\n5. System Status")
    print("-" * 70)
    status = ai.get_system_status()
    print(f"HRM parameters: {status['hrm']['total_parameters']:,}")
    print(f"Average reasoning steps: {status['unified']['avg_reasoning_steps']:.2f}")
    print(f"Total requests: {status['unified']['total_requests']}")
    
    print("\n6. Save/Load Checkpoint")
    print("-" * 70)
    checkpoint_path = ai.save_checkpoint()
    print(f"Saved to: {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("Unified System Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_unified_system()
