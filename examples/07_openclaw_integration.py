"""
Example: microgpt with OpenClaw Architecture
Demonstrates session management, auth profiles, and model fallback.
"""

import sys

sys.path.insert(0, "..")

from microgpt_openclaw_integration import MicroGPTWithOpenClaw, MicroGPTConfig


def demo_basic_chat():
    """Demonstrate basic chat with session management."""
    print("=" * 60)
    print("Demo: Basic Chat with Session Management")
    print("=" * 60)

    config = MicroGPTConfig(
        n_layer=2,  # Small model for demo
        n_embd=64,
        n_head=4,
        block_size=64,
    )

    model = MicroGPTWithOpenClaw(config)

    # Chat in a session
    session_id = "demo_session_001"

    print("\n--- Conversation 1 ---")
    result = model.chat("Hello, how are you?", session_id=session_id)
    print(f"User: Hello, how are you?")
    print(f"Assistant: {result['response'][:100]}...")
    print(f"Session: {result['session_id']}, Tokens: {result['tokens_used']}")

    print("\n--- Conversation 2 (same session) ---")
    result = model.chat("Tell me a story", session_id=session_id)
    print(f"User: Tell me a story")
    print(f"Assistant: {result['response'][:100]}...")
    print(f"Tokens: {result['tokens_used']}, Compactions: {result['compaction_count']}")

    # Check session info
    info = model.adapter.get_session_info(session_id)
    print(f"\nSession info: {info}")


def demo_auth_profiles():
    """Demonstrate authentication profile management."""
    print("\n" + "=" * 60)
    print("Demo: Auth Profile Management")
    print("=" * 60)

    from openclaw_adapter import AuthProfile

    model = MicroGPTWithOpenClaw()

    # Add auth profiles
    profiles = [
        AuthProfile("openai_1", "openai", api_key="sk-..."),
        AuthProfile("anthropic_1", "anthropic", api_key="sk-ant-..."),
        AuthProfile("local_1", "local"),
    ]

    for p in profiles:
        model.adapter.auth_store.add_profile(p)

    # List profiles
    print("\nAll profiles:")
    for p in model.adapter.auth_store.list_profiles():
        print(f"  - {p.profile_id} ({p.provider})")

    # Get available profile
    available = model.adapter.auth_store.get_available_profile("openai")
    print(f"\nAvailable OpenAI profile: {available.profile_id if available else None}")

    # Simulate failure and cooldown
    available.mark_failure(cooldown_seconds=60)
    print(f"Profile in cooldown: {available.is_in_cooldown()}")


def demo_session_compaction():
    """Demonstrate automatic session compaction."""
    print("\n" + "=" * 60)
    print("Demo: Session Compaction")
    print("=" * 60)

    config = MicroGPTConfig(max_context_tokens=100)  # Small for demo
    model = MicroGPTWithOpenClaw(config)

    session = model.adapter.create_session("compact_demo", max_context=100)

    # Add many messages to trigger compaction
    print("\nAdding messages...")
    for i in range(20):
        session.add_message("user", f"Message number {i} with some content " * 5)
        if i % 5 == 0:
            print(
                f"  Messages: {len(session.messages)}, "
                f"Tokens: {session.estimate_tokens()}, "
                f"Compactions: {session.compaction_count}"
            )

    print(f"\nFinal: {len(session.messages)} messages, " f"{session.compaction_count} compactions")


def demo_model_stats():
    """Show model statistics."""
    print("\n" + "=" * 60)
    print("Demo: Model Statistics")
    print("=" * 60)

    config = MicroGPTConfig(n_layer=4, n_embd=128, n_head=4)
    model = MicroGPTWithOpenClaw(config)

    # Initialize params to get accurate count
    model._init_params()

    stats = model.get_stats()
    print("\nModel Statistics:")
    print(f"  Parameters: {stats['model']['num_parameters']:,}")
    print(f"  Vocab size: {stats['model']['vocab_size']}")
    print(f"  Training step: {stats['model']['step']}")
    print(f"  Sessions: {stats['sessions']['count']}")


def demo_generation_options():
    """Demonstrate different generation options."""
    print("\n" + "=" * 60)
    print("Demo: Generation with Different Sampling")
    print("=" * 60)

    config = MicroGPTConfig()
    model = MicroGPTWithOpenClaw(config)
    model._init_params()  # Initialize for generation

    prompt = "Hello"

    print(f"\nPrompt: '{prompt}'")

    # Different sampling strategies
    strategies = [
        ("Greedy (temp=0.1)", {"temperature": 0.1}),
        ("Default (temp=0.8)", {"temperature": 0.8}),
        ("Creative (temp=1.2)", {"temperature": 1.2}),
        ("Top-k=5", {"temperature": 0.8, "top_k": 5}),
        ("Top-p=0.5", {"temperature": 0.8, "top_p": 0.5}),
    ]

    for name, kwargs in strategies:
        result = model.generate(prompt=prompt, max_length=50, **kwargs)
        print(f"\n{name}:")
        print(f"  {result[:80]}...")


def main():
    """Run all demos."""
    print("microgpt + OpenClaw Integration Examples")
    print("=" * 60)

    try:
        demo_basic_chat()
    except Exception as e:
        print(f"Chat demo error: {e}")

    try:
        demo_auth_profiles()
    except Exception as e:
        print(f"Auth demo error: {e}")

    try:
        demo_session_compaction()
    except Exception as e:
        print(f"Compaction demo error: {e}")

    try:
        demo_model_stats()
    except Exception as e:
        print(f"Stats demo error: {e}")

    try:
        demo_generation_options()
    except Exception as e:
        print(f"Generation demo error: {e}")

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
