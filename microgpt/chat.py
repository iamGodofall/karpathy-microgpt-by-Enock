"""
Interactive chat interface for microgpt.

Supports conversation history and context management.
"""

import random
from typing import List, Dict, Optional
from .model import GPT
from .data import CharTokenizer


class Conversation:
    """Manages conversation history and context."""

    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.history.append({"role": role, "content": content})

        # Trim history if too long
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

    def get_context(self) -> str:
        """Get conversation context as string."""
        context = []
        for msg in self.history:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            context.append(prefix + msg["content"])
        return "\n".join(context)

    def clear(self):
        """Clear conversation history."""
        self.history = []


class ChatBot:
    """Interactive chat bot using microgpt."""

    def __init__(self, model: GPT, tokenizer: CharTokenizer, system_prompt: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation = Conversation()
        self.system_prompt = system_prompt or "You are a helpful assistant."

        self.model.set_training(False)

    def generate_response(
        self, user_input: str, max_length: int = 100, temperature: float = 0.7
    ) -> str:
        """Generate response to user input."""
        # Add user message to history
        self.conversation.add_message("user", user_input)

        # Build prompt with context
        context = self.conversation.get_context()
        prompt = f"{self.system_prompt}\n\n{context}\nAssistant: "

        # Encode prompt
        tokens = self.tokenizer.encode(prompt)

        # Generate continuation
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        # Prime with prompt
        for i, token in enumerate(tokens[:-1]):
            _ = self.model.forward(token, i, keys, values)

        # Generate response
        generated_tokens = []
        current_token = tokens[-1] if tokens else self.tokenizer.bos_token

        for pos in range(len(tokens) - 1, len(tokens) - 1 + max_length):
            logits = self.model.forward(current_token, pos, keys, values)

            # Apply temperature
            scaled = [logit / temperature for logit in logits]
            probs = [p.data for p in self._softmax(scaled)]

            # Sample
            current_token = random.choices(range(len(probs)), weights=probs)[0]
            generated_tokens.append(current_token)

            # Stop on special tokens or natural breaks
            if current_token == self.tokenizer.bos_token:
                break

            # Stop at sentence end (heuristic)
            char = self.tokenizer.idx_to_char.get(current_token, "")
            if char in ".!?" and len(generated_tokens) > 10:
                break

        # Decode response
        response = self.tokenizer.decode(generated_tokens).strip()

        # Add to history
        self.conversation.add_message("assistant", response)

        return response

    def _softmax(self, logits: List) -> List:
        """Numerically stable softmax."""
        import math

        max_val = max(logit.data if hasattr(logit, "data") else logit for logit in logits)
        exps = [
            math.exp((logit.data if hasattr(logit, "data") else logit) - max_val)
            for logit in logits
        ]
        total = sum(exps)
        from model import Value

        return [Value(e / total) for e in exps]

    def chat_loop(self):
        """Run interactive chat loop."""
        print("=" * 70)
        print("MICROGPT CHAT")
        print("=" * 70)
        print(f"System: {self.system_prompt}")
        print("Commands: /clear (clear history), /quit (exit)")
        print("-" * 70)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                    print("Goodbye!")
                    break

                if user_input.lower() == "/clear":
                    self.conversation.clear()
                    print("Conversation history cleared.")
                    continue

                response = self.generate_response(user_input)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


class RolePlayBot(ChatBot):
    """Chat bot with specific role/personality."""

    ROLES = {
        "shakespeare": "You are William Shakespeare. Speak in iambic pentameter and Early Modern English.",
        "pirate": "You are a pirate. Speak like a swashbuckling buccaneer from the high seas.",
        "scientist": "You are a brilliant scientist. Explain things with precision and curiosity.",
        "poet": "You are a romantic poet. Respond with beautiful, flowing verse.",
        "chef": "You are a master chef. Give detailed cooking advice and recipes.",
        "default": "You are a helpful assistant.",
    }

    def __init__(self, model: GPT, tokenizer: CharTokenizer, role: str = "default"):
        system_prompt = self.ROLES.get(role, self.ROLES["default"])
        super().__init__(model, tokenizer, system_prompt)
        self.role = role

    def list_roles(self):
        """List available roles."""
        print("Available roles:")
        for role, description in self.ROLES.items():
            print(f"  {role}: {description[:50]}...")


class CodeAssistant(ChatBot):
    """Specialized chat bot for programming help."""

    def __init__(self, model: GPT, tokenizer: CharTokenizer):
        system_prompt = """You are a helpful coding assistant. 
        Provide clear, well-commented code examples. 
        Explain your reasoning step by step."""
        super().__init__(model, tokenizer, system_prompt)

    def generate_code(self, description: str, language: str = "python") -> str:
        """Generate code for a specific task."""
        prompt = f"Write {language} code to {description}:\n\n```"

        tokens = self.tokenizer.encode(prompt)

        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]

        # Generate
        for i, token in enumerate(tokens[:-1]):
            _ = self.model.forward(token, i, keys, values)

        generated = []
        current = tokens[-1]

        for pos in range(len(tokens) - 1, len(tokens) + 200):
            logits = self.model.forward(current, pos, keys, values)
            probs = [p.data for p in self._softmax(logits)]
            current = random.choices(range(len(probs)), weights=probs)[0]
            generated.append(current)

            # Stop at code block end
            if self.tokenizer.decode([current]) == "`":
                break

        code = self.tokenizer.decode(generated)
        return code.strip("`")


def load_chatbot(checkpoint_path: str, role: str = "default") -> ChatBot:
    """Load a chatbot from checkpoint."""
    from checkpoint import CheckpointManager

    checkpoint_mgr = CheckpointManager()
    checkpoint = checkpoint_mgr.load_pickle(checkpoint_path)

    config = checkpoint["config"]

    model = GPT(
        vocab_size=config["model"]["vocab_size"],
        block_size=config["model"]["block_size"],
        n_layer=config["model"]["n_layer"],
        n_embd=config["model"]["n_embd"],
        n_head=config["model"]["n_head"],
        use_gelu=config["model"].get("use_gelu", False),
        use_layernorm=config["model"].get("use_layernorm", False),
    )

    # Load weights
    for name, matrix_data in checkpoint["state_dict"].items():
        for i, row in enumerate(matrix_data):
            for j, val in enumerate(row):
                model.state_dict[name][i][j].data = val

    # Create tokenizer
    tokenizer = CharTokenizer()
    # Would need to load actual vocab from checkpoint
    # For now, use a simple vocab
    tokenizer.fit(["hello world example"])

    if role == "code":
        return CodeAssistant(model, tokenizer)
    elif role in RolePlayBot.ROLES:
        return RolePlayBot(model, tokenizer, role)
    else:
        return ChatBot(model, tokenizer)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chat with microgpt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--role", type=str, default="default", help="Bot personality role")

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    bot = load_chatbot(args.checkpoint, args.role)

    if hasattr(bot, "list_roles"):
        bot.list_roles()

    bot.chat_loop()


if __name__ == "__main__":
    main()
