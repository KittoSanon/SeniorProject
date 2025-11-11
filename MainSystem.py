from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
import os
import urllib.request
import urllib.error


# ---------- Core components ----------

@dataclass
class ShortTermMemory:
    """Simple in-memory buffer of recent conversation turns."""
    max_turns: int = 20
    turns: List[Tuple[str, str]] = field(default_factory=list)  # (user, assistant)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))
        if len(self.turns) > self.max_turns:
            # Keep only the most recent max_turns
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        self.turns.clear()

    def summary(self) -> str:
        if not self.turns:
            return "(empty)"
        return "\n".join(f"U: {u}\nA: {a}" for u, a in self.turns[-self.max_turns :])


class SafeCalculator:
    """
    Safely evaluate basic arithmetic expressions from text.
    Supports +, -, *, /, //, %, **, parentheses, and unary +/-. No names or function calls.
    """

    _allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def try_calculate(self, text: str) -> Optional[float]:
        """
        Try to parse and evaluate an arithmetic expression from the entire text.
        Returns a float/int on success, or None if the text isn't a pure expression.
        """
        try:
            node = ast.parse(text.strip(), mode="eval")
        except SyntaxError:
            return None
        try:
            value = self._eval_node(node.body)
        except Exception:
            return None
        return value

    def _eval_node(self, node):
        if isinstance(node, ast.BinOp) and type(node.op) in self._allowed_ops:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._allowed_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._allowed_ops:
            operand = self._eval_node(node.operand)
            return self._allowed_ops[type(node.op)](operand)
        if isinstance(node, ast.Num):  # py<3.8 compatibility
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        # Disallow everything else (names, calls, etc.)
        raise ValueError("Unsupported expression")


def process_input(user_text: str, memory: ShortTermMemory, calculator: SafeCalculator) -> str:
    """
    Conversational processor:
    - First, attempt to evaluate if the input is a pure arithmetic expression.
    - Otherwise, provide a simple conversational response using short-term memory context.
    """
    # 1) Try calculation
    calc_result = calculator.try_calculate(user_text)
    if calc_result is not None:
        return f"The result is {calc_result}"

    # 2) Simple conversational echo with context length
    memory_size = len(memory.turns)
    if user_text.strip().endswith("?"):
        return f"You asked a question. I currently remember {memory_size} turn(s)."
    return f"I hear you: '{user_text.strip()}'. I remember {memory_size} turn(s)."


# ---------- User recognition ----------

@dataclass
class UserProfile:
    name: str
    created_at_iso: str
    last_seen_iso: str

    @staticmethod
    def new(name: str) -> "UserProfile":
        now = datetime.utcnow().isoformat()
        return UserProfile(name=name, created_at_iso=now, last_seen_iso=now)

    def touch(self) -> None:
        self.last_seen_iso = datetime.utcnow().isoformat()


class UserRegistry:
    """
    Persistent user registry backed by a small JSON file in the project directory.
    Keys are lowercase names; values are user profile dicts.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._users = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._users, f, indent=2)
        tmp_path.replace(self.path)

    def get_or_create(self, name: str) -> UserProfile:
        key = name.strip().lower()
        if not key:
            raise ValueError("User name cannot be empty.")
        raw = self._users.get(key)
        if raw is None:
            profile = UserProfile.new(name=name.strip())
            self._users[key] = profile.__dict__
            self._save()
            return profile
        profile = UserProfile(**raw)
        profile.touch()
        self._users[key] = profile.__dict__
        self._save()
        return profile


# ---------- AI provider (optional) ----------

class AIClient:
    """
    Minimal HTTP JSON client for chat completion style APIs.
    Supports OpenAI-compatible endpoints (e.g., OpenAI, Azure OpenAI with compat, local servers).
    Environment variables:
      AI_API_KEY       - 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
      AI_BASE_URL      - base URL, defaults to https://api.openai.com/v1
      AI_MODEL         - model name, defaults to gpt-4o-mini
    If not configured, the client is disabled and returns None for responses.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("AI_API_KEY", "").strip()
        self.base_url = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("AI_MODEL", "gpt-4o-mini").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    def set_model(self, model: str) -> None:
        if model.strip():
            self.model = model.strip()

    def chat(self, system_prompt: str, messages: List[Tuple[str, str]]) -> Optional[str]:
        """
        messages: list of (role, content), role in {'system','user','assistant'}
        Returns assistant message string on success, else None.
        """
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}]
            + [{"role": r, "content": c} for r, c in messages],
            "temperature": 0.7,
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                doc = json.loads(raw.decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            return None

        # OpenAI-compatible shape
        try:
            return doc["choices"][0]["message"]["content"]
        except Exception:
            return None


def main() -> None:
    """
    Minimal conversational loop with memory and calculator.
    Commands:
      /help  - show help
      /mem   - show recent memory
      /clear - clear memory
      /ai    - toggle AI provider on/off (if configured)
      /model <name> - set AI model name
      exit   - quit
    """
    # Initialize services
    memory = ShortTermMemory()
    calculator = SafeCalculator()
    registry = UserRegistry(path=Path(__file__).with_name("users.json"))
    ai = AIClient()
    ai_enabled = ai.enabled

    # Onboarding and recognition
    print("Welcome. Before we chat, what's your name?")
    while True:
        try:
            name_input = input("Name> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if name_input:
            break
        print("Please enter a non-empty name.")

    profile = registry.get_or_create(name_input)
    if profile.created_at_iso == profile.last_seen_iso:
        print(f"Nice to meet you, {profile.name}! I've created your profile.")
    else:
        print(f"Welcome back, {profile.name}! Recognized you from previous visits.")

    print("Main system ready. Type your text and press Enter. Type 'exit' to quit.")
    print("Commands: /help, /mem, /clear, /ai, /model <name>")
    if not ai_enabled:
        print("AI provider not configured. Set AI_API_KEY (and optionally AI_BASE_URL, AI_MODEL).")
    while True:
        try:
            user_text = input("Instruction> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_text.strip().lower() == "exit":
            print("Goodbye.")
            break

        # Commands
        cmd = user_text.strip().lower()
        if cmd == "/help":
            print("Type arithmetic (e.g., 2+2*3) to calculate, or chat naturally.")
            print("Commands: /help, /mem, /clear, /ai, /model <name>, exit")
            continue
        if cmd == "/mem":
            print("Short term memory:")
            print(memory.summary())
            continue
        if cmd == "/clear":
            memory.clear()
            print("Memory cleared.")
            continue
        if cmd == "/ai":
            ai_enabled = not ai_enabled and ai.enabled
            print(f"AI is now {'ON' if ai_enabled else 'OFF'}.")
            continue
        if cmd.startswith("/model "):
            _, _, model_name = user_text.partition(" ")
            if model_name.strip():
                ai.set_model(model_name.strip())
                print(f"Model set to: {ai.model}")
            else:
                print("Usage: /model <name>")
            continue

        # If AI is enabled, construct a context and get AI response first
        result: Optional[str] = None
        if ai_enabled:
            system_prompt = (
                "You are a helpful assistant. Personalize replies using the user's name. "
                "Be concise, reference short-term memory if relevant, and solve basic problems."
            )
            history: List[Tuple[str, str]] = []
            # include up to last 6 turns for context
            for u, a in memory.turns[-6:]:
                history.append(("user", u))
                history.append(("assistant", a))
            # add current user message with name prefix
            history.append(("user", f"{profile.name}: {user_text}"))
            ai_reply = ai.chat(system_prompt, history)
            if ai_reply:
                result = ai_reply

        # Fallback to local processor if AI disabled or failed
        if result is None:
            result = process_input(user_text, memory, calculator)

        print(f"Result OUTPUT: {result}")
        memory.add(user_text, result)


if __name__ == "__main__":
    main()


