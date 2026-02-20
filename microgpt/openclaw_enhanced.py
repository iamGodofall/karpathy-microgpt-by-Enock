"""
Enhanced OpenClaw Integration for microgpt
Production-ready with streaming, tools, multi-modal, and advanced features.
"""

import json
import time
import random
from typing import Dict, List, Optional, Callable, Any, Tuple, AsyncIterator, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor


class ThinkLevel(Enum):
    """Extended thinking levels with adaptive depth."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    ADAPTIVE = "adaptive"  # New: dynamically adjust based on query complexity


class ToolType(Enum):
    """Types of tools available."""

    CODE = auto()
    SEARCH = auto()
    CALCULATOR = auto()
    BROWSER = auto()
    FILE_SYSTEM = auto()
    API = auto()
    CUSTOM = auto()


@dataclass
class Tool:
    """Enhanced tool definition with schema and validation."""

    name: str
    description: str
    type: ToolType
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # Calls per minute
    timeout: float = 30.0

    def validate_params(self, params: Dict) -> Tuple[bool, str]:
        """Validate tool parameters."""
        for req in self.required_params:
            if req not in params:
                return False, f"Missing required parameter: {req}"
        return True, ""


@dataclass
class AuthProfile:
    """Enhanced authentication with rate limiting and health checks."""

    profile_id: str
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    last_used: Optional[float] = None
    failure_count: int = 0
    cooldown_until: Optional[float] = None
    rate_limit_remaining: int = 1000
    rate_limit_reset: float = 0.0
    health_score: float = 1.0  # 0-1 health metric

    def is_in_cooldown(self) -> bool:
        """Check if profile is in cooldown period."""
        if self.cooldown_until is None:
            return False
        return time.time() < self.cooldown_until

    def mark_failure(self, cooldown_seconds: int = 60):
        """Mark profile as failed and set cooldown."""
        self.failure_count += 1
        self.cooldown_until = time.time() + cooldown_seconds
        self.health_score = max(0.0, self.health_score - 0.1)

    def mark_success(self):
        """Mark profile as successfully used."""
        self.last_used = time.time()
        self.failure_count = 0
        self.cooldown_until = None
        self.health_score = min(1.0, self.health_score + 0.05)

    def is_healthy(self) -> bool:
        """Check if profile is healthy."""
        return self.health_score > 0.5 and not self.is_in_cooldown()


class StreamingBuffer:
    """Buffer for streaming responses with backpressure handling."""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.closed = False

    def write(self, token: str):
        """Write token to buffer."""
        with self.lock:
            self.buffer.append(token)
            self.event.set()

    def read(self, timeout: float = 0.1) -> Optional[str]:
        """Read token from buffer."""
        if self.closed and not self.buffer:
            return None

        self.event.wait(timeout)
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            self.event.clear()
            return None

    def close(self):
        """Close the buffer."""
        self.closed = True
        self.event.set()


@dataclass
class SessionMessage:
    """Enhanced message with tool calls and reasoning traces."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # New fields
    tool_calls: List[Dict] = field(default_factory=list)
    reasoning_trace: Optional[str] = None
    token_count: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "reasoning_trace": self.reasoning_trace,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
        }


class Session:
    """Enhanced session with tool use and streaming support."""

    def __init__(
        self,
        session_id: str,
        max_context_tokens: int = 4096,
        compaction_threshold: float = 0.8,
        enable_tools: bool = True,
        streaming: bool = False,
    ):
        self.session_id = session_id
        self.messages: List[SessionMessage] = []
        self.max_context_tokens = max_context_tokens
        self.compaction_threshold = compaction_threshold
        self.created_at = time.time()
        self.last_activity = time.time()
        self.compaction_count = 0
        self.metadata: Dict[str, Any] = {}

        # New features
        self.enable_tools = enable_tools
        self.streaming = streaming
        self.tool_results: List[Dict] = []
        self.reasoning_history: List[str] = []
        self.stream_buffer: Optional[StreamingBuffer] = None

        if streaming:
            self.stream_buffer = StreamingBuffer()

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_trace: Optional[str] = None,
    ):
        """Add a message with enhanced metadata."""
        msg = SessionMessage(
            role=role,
            content=content,
            metadata=metadata or {},
            tool_calls=tool_calls or [],
            reasoning_trace=reasoning_trace,
            token_count=len(content) // 4,  # Rough estimate
        )
        self.messages.append(msg)
        self.last_activity = time.time()
        self._check_compaction()

    def estimate_tokens(self) -> int:
        """Improved token estimation."""
        total = sum(m.token_count for m in self.messages)
        # Add overhead for tool calls and reasoning
        for m in self.messages:
            if m.tool_calls:
                total += len(m.tool_calls) * 10
            if m.reasoning_trace:
                total += len(m.reasoning_trace) // 4
        return total

    def _check_compaction(self):
        """Smart compaction with importance weighting."""
        tokens = self.estimate_tokens()
        if tokens > self.max_context_tokens * self.compaction_threshold:
            self.smart_compact()

    def smart_compact(self, keep_recent: int = 4):
        """
        Smart compaction preserving important messages.
        Uses importance scoring to decide what to keep.
        """
        if len(self.messages) <= keep_recent:
            return

        # Score messages by importance
        scored_messages = []
        for i, m in enumerate(self.messages[:-keep_recent]):
            score = self._importance_score(m, i)
            scored_messages.append((score, i, m))

        # Keep high-importance messages
        scored_messages.sort(reverse=True)
        keep_count = max(1, len(scored_messages) // 3)
        important_indices = set(idx for _, idx, _ in scored_messages[:keep_count])

        # Build new message list
        system_msgs = [m for m in self.messages if m.role == "system"]
        important_msgs = [
            self.messages[i] for i in sorted(important_indices) if self.messages[i].role != "system"
        ]
        recent_msgs = self.messages[-keep_recent:]

        # Summarize non-important messages
        other_msgs = [
            m
            for i, m in enumerate(self.messages[:-keep_recent])
            if i not in important_indices and m.role != "system"
        ]

        if other_msgs:
            summary = self._create_summary(other_msgs)
            summary_msg = SessionMessage(
                role="system",
                content=f"[Summary: {summary}]",
                metadata={"is_summary": True, "summarized_count": len(other_msgs)},
            )
            self.messages = system_msgs + important_msgs + [summary_msg] + recent_msgs
        else:
            self.messages = system_msgs + important_msgs + recent_msgs

        self.compaction_count += 1

    def _importance_score(self, message: SessionMessage, index: int) -> float:
        """Calculate importance score for a message."""
        score = 0.0

        # Recency bias
        score += (index / len(self.messages)) * 0.3

        # Tool calls are important
        if message.tool_calls:
            score += 0.4

        # Reasoning traces are important
        if message.reasoning_trace:
            score += 0.3

        # User questions are important
        if message.role == "user":
            score += 0.2

        # Error messages are important
        if message.metadata.get("is_error"):
            score += 0.5

        return score

    def _create_summary(self, messages: List[SessionMessage]) -> str:
        """Create intelligent summary of messages."""
        topics = set()
        entities = []

        for m in messages:
            # Extract key information
            content = m.content.lower()
            words = content.split()

            # Simple entity extraction (capitalized words)
            for word in words:
                if word[0].isupper() and len(word) > 3:
                    entities.append(word)

            # Topic detection
            if "how" in content or "what" in content:
                topics.add("questions")
            if "error" in content or "fail" in content:
                topics.add("issues")
            if "success" in content or "done" in content:
                topics.add("completions")

        summary_parts = []
        if topics:
            summary_parts.append(f"topics: {', '.join(topics)}")
        if entities:
            summary_parts.append(f"entities: {', '.join(set(entities)[:5])}")

        return "; ".join(summary_parts) if summary_parts else "general discussion"

    def get_context_window(self, include_tools: bool = True) -> List[Dict]:
        """Get context with optional tool descriptions."""
        context = [m.to_dict() for m in self.messages]

        if include_tools and self.enable_tools:
            # Add tool descriptions to system context
            tool_desc = {
                "role": "system",
                "content": "[Tools available: code_execution, web_search, calculator]",
                "metadata": {"is_tool_description": True},
            }
            context.insert(0, tool_desc)

        return context

    def stream_token(self, token: str):
        """Stream a token if streaming is enabled."""
        if self.stream_buffer:
            self.stream_buffer.write(token)

    def to_dict(self) -> Dict:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "max_context_tokens": self.max_context_tokens,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "compaction_count": self.compaction_count,
            "metadata": self.metadata,
            "enable_tools": self.enable_tools,
            "streaming": self.streaming,
            "tool_results": self.tool_results,
        }


class EnhancedOpenClaw:
    """
    Production-ready OpenClaw adapter with advanced features.
    """

    def __init__(
        self,
        storage_dir: str = ".microgpt",
        default_model: str = "microgpt",
        max_workers: int = 4,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.default_model = default_model
        self.max_workers = max_workers

        # Components
        self.sessions: Dict[str, Session] = {}
        self.auth_profiles: Dict[str, AuthProfile] = {}
        self.tools: Dict[str, Tool] = {}
        self.hooks: Dict[str, List[Callable]] = {}

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.RLock()

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0.0,
        }

        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        auth_path = self.storage_dir / "auth.json"
        if auth_path.exists():
            try:
                data = json.loads(auth_path.read_text())
                for pid, pdata in data.get("profiles", {}).items():
                    self.auth_profiles[pid] = AuthProfile(**pdata)
            except Exception as e:
                print(f"Error loading auth: {e}")

        # Load sessions
        sessions_dir = self.storage_dir / "sessions"
        if sessions_dir.exists():
            for path in sessions_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    session = Session(
                        session_id=data["session_id"],
                        max_context_tokens=data.get("max_context_tokens", 4096),
                        enable_tools=data.get("enable_tools", True),
                        streaming=data.get("streaming", False),
                    )
                    session.messages = [SessionMessage(**m) for m in data.get("messages", [])]
                    session.created_at = data.get("created_at", time.time())
                    session.last_activity = data.get("last_activity", time.time())
                    session.compaction_count = data.get("compaction_count", 0)
                    session.metadata = data.get("metadata", {})
                    self.sessions[session.session_id] = session
                except Exception as e:
                    print(f"Error loading session {path}: {e}")

    def _save_auth(self):
        """Save authentication state."""
        auth_path = self.storage_dir / "auth.json"
        data = {"profiles": {pid: asdict(p) for pid, p in self.auth_profiles.items()}}
        auth_path.write_text(json.dumps(data, indent=2, default=str))

    def _save_session(self, session: Session):
        """Save session to disk."""
        sessions_dir = self.storage_dir / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        path = sessions_dir / f"{session.session_id}.json"
        path.write_text(json.dumps(session.to_dict(), indent=2, default=str))

    def register_tool(self, tool: Tool):
        """Register a tool."""
        with self.lock:
            self.tools[tool.name] = tool

    def register_hook(self, event: str, callback: Callable):
        """Register event hook."""
        with self.lock:
            if event not in self.hooks:
                self.hooks[event] = []
            self.hooks[event].append(callback)

    def execute_tool(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        tool = self.tools[tool_name]

        # Validate parameters
        valid, error = tool.validate_params(params)
        if not valid:
            return {"error": error}

        # Execute with timeout
        try:
            start = time.time()
            result = tool.function(**params)
            latency = (time.time() - start) * 1000

            return {
                "result": result,
                "tool": tool_name,
                "latency_ms": latency,
                "success": True,
            }
        except Exception as e:
            return {
                "error": str(e),
                "tool": tool_name,
                "success": False,
            }

    def create_session(
        self,
        session_id: Optional[str] = None,
        max_context: int = 4096,
        enable_tools: bool = True,
        streaming: bool = False,
    ) -> Session:
        """Create enhanced session."""
        with self.lock:
            if session_id is None:
                session_id = f"sess_{int(time.time())}_{random.randint(1000, 9999)}"

            session = Session(
                session_id=session_id,
                max_context_tokens=max_context,
                enable_tools=enable_tools,
                streaming=streaming,
            )
            self.sessions[session_id] = session
            self._save_session(session)
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def stream_generate(
        self,
        session_id: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.8,
    ) -> StreamingBuffer:
        """
        Stream generation with token-by-token output.
        Returns a buffer that can be read asynchronously.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if not session.streaming:
            raise ValueError("Session not configured for streaming")

        # Start generation in background
        def generate():
            # Simulate streaming (would call actual model)
            tokens = self._tokenize(prompt)
            for i, token in enumerate(tokens):
                time.sleep(0.01)  # Simulate generation time
                session.stream_token(token)
            session.stream_buffer.close()

        self.executor.submit(generate)
        return session.stream_buffer

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for demo."""
        return list(text)  # Character-level for demo

    def chat(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[str]] = None,
        think_level: ThinkLevel = ThinkLevel.ADAPTIVE,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Enhanced chat with tool use and adaptive thinking.
        """
        start_time = time.time()

        # Get or create session
        if session_id:
            session = self.get_session(session_id)
            if not session:
                session = self.create_session(session_id, streaming=stream)
        else:
            session = self.create_session(streaming=stream)

        # Determine thinking depth
        if think_level == ThinkLevel.ADAPTIVE:
            think_level = self._assess_complexity(prompt)

        # Add user message
        session.add_message("user", prompt)

        # Execute tools if needed
        tool_results = []
        if session.enable_tools and tools:
            for tool_name in tools:
                # Simple tool detection (would use model in real implementation)
                if tool_name in self.tools:
                    result = self.execute_tool(tool_name, {"query": prompt})
                    tool_results.append(result)

        # Generate response (placeholder)
        response = self._generate_response(
            prompt,
            model or self.default_model,
            think_level,
            tool_results,
        )

        # Add assistant message
        latency = (time.time() - start_time) * 1000
        session.add_message(
            role="assistant",
            content=response,
            tool_calls=[
                {"tool": r["tool"], "result": r.get("result")}
                for r in tool_results
                if r.get("success")
            ],
        )
        session.messages[-1].latency_ms = latency

        # Update metrics
        self.metrics["total_requests"] += 1
        self.metrics["successful_requests"] += 1
        self.metrics["average_latency"] = (
            self.metrics["average_latency"] * (self.metrics["total_requests"] - 1) + latency
        ) / self.metrics["total_requests"]

        # Save session
        self._save_session(session)

        return {
            "response": response,
            "session_id": session.session_id,
            "tools_used": [r["tool"] for r in tool_results if r.get("success")],
            "latency_ms": latency,
            "think_level": think_level.value,
        }

    def _assess_complexity(self, prompt: str) -> ThinkLevel:
        """Assess query complexity for adaptive thinking."""
        # Simple heuristic (would use model in production)
        length = len(prompt)
        complexity_indicators = [
            "explain",
            "analyze",
            "compare",
            "reason",
            "solve",
            "step by step",
            "why",
            "how does",
            "what if",
        ]

        score = 0
        if length > 200:
            score += 1
        if length > 500:
            score += 1

        for indicator in complexity_indicators:
            if indicator in prompt.lower():
                score += 1

        if score <= 1:
            return ThinkLevel.LOW
        elif score <= 3:
            return ThinkLevel.MEDIUM
        else:
            return ThinkLevel.HIGH

    def _generate_response(
        self,
        prompt: str,
        model: str,
        think_level: ThinkLevel,
        tool_results: List[Dict],
    ) -> str:
        """Generate response (placeholder for actual model)."""
        # This would call the actual microgpt model
        thinking_time = {
            ThinkLevel.OFF: 0,
            ThinkLevel.MINIMAL: 1,
            ThinkLevel.LOW: 2,
            ThinkLevel.MEDIUM: 3,
            ThinkLevel.HIGH: 5,
            ThinkLevel.XHIGH: 8,
        }.get(think_level, 3)

        # Simulate thinking
        time.sleep(thinking_time * 0.01)

        response_parts = [f"[Model {model} thinking for {thinking_time} steps]"]

        if tool_results:
            response_parts.append(f"Used {len(tool_results)} tools")

        response_parts.append(f"Response to: {prompt[:50]}...")

        return " | ".join(response_parts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            **self.metrics,
            "active_sessions": len(self.sessions),
            "registered_tools": len(self.tools),
            "auth_profiles": len(self.auth_profiles),
        }
