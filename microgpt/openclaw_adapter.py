"""
OpenClaw Architecture Adaptation for microgpt.
Adapts key concepts from OpenClaw's agent runtime to pure Python.
"""

import json
import time
import random
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ThinkLevel(Enum):
    """Thinking levels for reasoning models."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class AuthProfile:
    """Authentication profile for model providers."""

    profile_id: str
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    last_used: Optional[float] = None
    failure_count: int = 0
    cooldown_until: Optional[float] = None

    def is_in_cooldown(self) -> bool:
        """Check if profile is in cooldown period."""
        if self.cooldown_until is None:
            return False
        return time.time() < self.cooldown_until

    def mark_failure(self, cooldown_seconds: int = 60):
        """Mark profile as failed and set cooldown."""
        self.failure_count += 1
        self.cooldown_until = time.time() + cooldown_seconds

    def mark_success(self):
        """Mark profile as successfully used."""
        self.last_used = time.time()
        self.failure_count = 0
        self.cooldown_until = None


class AuthProfileStore:
    """Manages multiple authentication profiles with rotation."""

    def __init__(self, storage_path: Optional[str] = None):
        self.profiles: Dict[str, AuthProfile] = {}
        self.storage_path = Path(storage_path) if storage_path else None
        self._load()

    def _load(self):
        """Load profiles from storage."""
        if self.storage_path and self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for pid, pdata in data.get("profiles", {}).items():
                    self.profiles[pid] = AuthProfile(**pdata)
            except Exception:
                pass

    def save(self):
        """Save profiles to storage."""
        if self.storage_path:
            data = {
                "profiles": {
                    pid: {
                        "profile_id": p.profile_id,
                        "provider": p.provider,
                        "api_key": p.api_key,
                        "base_url": p.base_url,
                        "last_used": p.last_used,
                        "failure_count": p.failure_count,
                        "cooldown_until": p.cooldown_until,
                    }
                    for pid, p in self.profiles.items()
                }
            }
            self.storage_path.write_text(json.dumps(data, indent=2))

    def add_profile(self, profile: AuthProfile):
        """Add or update a profile."""
        self.profiles[profile.profile_id] = profile
        self.save()

    def get_profile(self, profile_id: str) -> Optional[AuthProfile]:
        """Get a profile by ID."""
        return self.profiles.get(profile_id)

    def list_profiles(self, provider: Optional[str] = None) -> List[AuthProfile]:
        """List profiles, optionally filtered by provider."""
        profiles = list(self.profiles.values())
        if provider:
            profiles = [p for p in profiles if p.provider == provider]
        return profiles

    def get_available_profile(
        self, provider: str, preferred: Optional[str] = None
    ) -> Optional[AuthProfile]:
        """Get an available (not in cooldown) profile for a provider."""
        candidates = self.list_profiles(provider)

        # Try preferred first
        if preferred and preferred in self.profiles:
            profile = self.profiles[preferred]
            if not profile.is_in_cooldown():
                return profile

        # Find any available profile
        for profile in candidates:
            if not profile.is_in_cooldown():
                return profile

        return None


@dataclass
class SessionMessage:
    """A message in a conversation session."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class Session:
    """Manages a conversation session with context window handling."""

    def __init__(
        self,
        session_id: str,
        max_context_tokens: int = 4096,
        compaction_threshold: float = 0.8,
    ):
        self.session_id = session_id
        self.messages: List[SessionMessage] = []
        self.max_context_tokens = max_context_tokens
        self.compaction_threshold = compaction_threshold
        self.created_at = time.time()
        self.last_activity = time.time()
        self.compaction_count = 0
        self.metadata: Dict[str, Any] = {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the session."""
        msg = SessionMessage(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.last_activity = time.time()

        # Check if compaction needed
        self._check_compaction()

    def estimate_tokens(self) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: 4 chars ≈ 1 token
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4

    def _check_compaction(self):
        """Check if session needs compaction."""
        tokens = self.estimate_tokens()
        if tokens > self.max_context_tokens * self.compaction_threshold:
            self.compact()

    def compact(self, keep_recent: int = 4):
        """
        Compact session by summarizing old messages.
        Keeps recent messages and system messages.
        """
        if len(self.messages) <= keep_recent:
            return

        # Separate system messages and recent messages
        system_msgs = [m for m in self.messages if m.role == "system"]
        recent_msgs = self.messages[-keep_recent:]

        # Summarize older messages
        old_msgs = self.messages[:-keep_recent]
        old_msgs = [m for m in old_msgs if m.role != "system"]

        if old_msgs:
            summary = self._summarize_messages(old_msgs)
            summary_msg = SessionMessage(
                role="system",
                content=f"[Previous conversation summary: {summary}]",
                metadata={"is_summary": True},
            )
            self.messages = system_msgs + [summary_msg] + recent_msgs
            self.compaction_count += 1

    def _summarize_messages(self, messages: List[SessionMessage]) -> str:
        """Create a summary of messages (placeholder)."""
        # In a real implementation, this would use the model to summarize
        topics = set()
        for m in messages:
            # Extract key topics (simple approach)
            words = m.content.lower().split()[:10]
            topics.update(words)

        return f"Discussed: {', '.join(list(topics)[:5])}..."

    def get_context_window(self) -> List[Dict]:
        """Get messages within context window."""
        return [m.to_dict() for m in self.messages]

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
        }


class SessionManager:
    """Manages multiple sessions."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.sessions: Dict[str, Session] = {}
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    def _session_path(self, session_id: str) -> Path:
        """Get storage path for a session."""
        return self.storage_dir / f"{session_id}.json"

    def _load_all(self):
        """Load all sessions from storage."""
        if not self.storage_dir:
            return

        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                session = Session(
                    session_id=data["session_id"],
                    max_context_tokens=data.get("max_context_tokens", 4096),
                )
                session.messages = [SessionMessage(**m) for m in data.get("messages", [])]
                session.created_at = data.get("created_at", time.time())
                session.last_activity = data.get("last_activity", time.time())
                session.compaction_count = data.get("compaction_count", 0)
                session.metadata = data.get("metadata", {})
                self.sessions[session.session_id] = session
            except Exception:
                pass

    def create_session(
        self,
        session_id: Optional[str] = None,
        max_context_tokens: int = 4096,
    ) -> Session:
        """Create a new session."""
        if session_id is None:
            session_id = f"sess_{int(time.time())}_{random.randint(1000, 9999)}"

        session = Session(session_id, max_context_tokens)
        self.sessions[session_id] = session
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def get_or_create(self, session_id: str, max_context_tokens: int = 4096) -> Session:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id, max_context_tokens)
        return session

    def _save_session(self, session: Session):
        """Save a session to storage."""
        if self.storage_dir:
            path = self._session_path(session.session_id)
            path.write_text(json.dumps(session.to_dict(), indent=2))

    def save_all(self):
        """Save all sessions."""
        for session in self.sessions.values():
            self._save_session(session)

    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self.sessions.keys())

    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.storage_dir:
                path = self._session_path(session_id)
                path.unlink(missing_ok=True)


class ModelFallback:
    """Handles model fallback on failures."""

    def __init__(self, models: List[str], fallback_chain: Optional[List[str]] = None):
        self.models = models
        self.fallback_chain = fallback_chain or []
        self.current_index = 0
        self.attempted = set()

    def get_current_model(self) -> Optional[str]:
        """Get current model to try."""
        if self.current_index < len(self.fallback_chain):
            return self.fallback_chain[self.current_index]
        return None

    def advance(self) -> bool:
        """Move to next fallback model."""
        self.current_index += 1
        return self.current_index < len(self.fallback_chain)

    def is_fallback_exhausted(self) -> bool:
        """Check if all fallbacks have been tried."""
        return self.current_index >= len(self.fallback_chain)


class OpenClawAdapter:
    """
    Main adapter bringing OpenClaw concepts to microgpt.
    Provides session management, auth profiles, and model fallback.
    """

    def __init__(
        self,
        storage_dir: str = ".microgpt",
        default_model: str = "microgpt",
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.sessions = SessionManager(str(self.storage_dir / "sessions"))
        self.auth_store = AuthProfileStore(str(self.storage_dir / "auth.json"))
        self.default_model = default_model

        # Tool registry
        self.tools: Dict[str, Callable] = {}

        # Event hooks
        self.hooks: Dict[str, List[Callable]] = {}

    def register_tool(self, name: str, func: Callable):
        """Register a tool function."""
        self.tools[name] = func

    def register_hook(self, event: str, callback: Callable):
        """Register an event hook."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)

    async def run_hook(self, event: str, *args, **kwargs):
        """Run hooks for an event."""
        for hook in self.hooks.get(event, []):
            try:
                result = hook(*args, **kwargs)
                if hasattr(result, "__await__"):
                    result = await result
            except Exception as e:
                print(f"Hook error for {event}: {e}")

    def create_session(
        self,
        session_id: Optional[str] = None,
        max_context: int = 4096,
    ) -> Session:
        """Create a new conversation session."""
        return self.sessions.create_session(session_id, max_context)

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        return self.sessions.get_session(session_id)

    def run_with_fallback(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Run inference with automatic model fallback.
        Adapted from OpenClaw's pi-embedded-runner.
        """
        session = self.get_or_create_session(session_id)
        fallback = ModelFallback(
            models=[model or self.default_model], fallback_chain=fallback_models or []
        )

        last_error = None

        while not fallback.is_fallback_exhausted():
            current_model = fallback.get_current_model()

            try:
                # Run hook before model execution
                self.run_hook("before_model_run", prompt, current_model)

                # Execute (placeholder - would call actual model)
                result = self._execute_model(prompt, current_model, session)

                # Add to session history
                session.add_message("user", prompt)
                session.add_message("assistant", result.get("text", ""))

                return {
                    "success": True,
                    "model": current_model,
                    "result": result,
                    "session_id": session.session_id,
                }

            except Exception as e:
                last_error = e
                fallback.advance()

        # All fallbacks exhausted
        return {
            "success": False,
            "error": str(last_error),
            "session_id": session.session_id if session else None,
        }

    def _execute_model(
        self,
        prompt: str,
        model: str,
        session: Session,
    ) -> Dict[str, Any]:
        """
        Execute model inference.
        Placeholder - would integrate with actual model.
        """
        # This would call the actual microgpt model
        # For now, return a placeholder
        return {
            "text": f"[Model {model} response to: {prompt[:50]}...]",
            "tokens_used": len(prompt) // 4,
        }

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get or create a session."""
        if session_id:
            return self.sessions.get_or_create(session_id)
        return self.sessions.create_session()

    def compact_session(self, session_id: str):
        """Manually compact a session."""
        session = self.sessions.get_session(session_id)
        if session:
            session.compact()
            self.sessions._save_session(session)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information."""
        session = self.sessions.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "message_count": len(session.messages),
                "estimated_tokens": session.estimate_tokens(),
                "compaction_count": session.compaction_count,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
            }
        return None
