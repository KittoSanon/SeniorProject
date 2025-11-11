from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
import os

# ---------- Core components ----------

@dataclass
class ShortTermMemory:
    """Simple in-memory buffer of recent conversation turns."""
    max_turns: int = 20
    turns: List[Tuple[str, str]] = field(default_factory=list)  # (user, assistant)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))
        if len(self.turns) > self.max_turns:
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
        """Return numeric result if `text` is a pure arithmetic expression; else None."""
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
        raise ValueError("Unsupported expression")


def process_input(user_text: str, memory: ShortTermMemory, calculator: SafeCalculator) -> str:
    """
    Conversational processor when AI is off/unavailable.
    """
    calc_result = calculator.try_calculate(user_text)
    if calc_result is not None:
        # แสดงเป็น int ถ้าลงตัว
        if isinstance(calc_result, float) and calc_result.is_integer():
            calc_result = int(calc_result)
        return f"The result is {calc_result}"

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


# ---------- Gemini provider ----------

class GeminiClient:
    """
    Minimal Gemini client using google-generativeai (AI Studio).
    Env:
      GEMINI_API_KEY  - required
      GEMINI_MODEL    - default 'gemini-1.5-flash' (or 'gemini-1.5-pro', etc.)
    """
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        self._model = None

        try:
            import google.generativeai as genai  # type: ignore
            self._genai = genai
            if self.api_key:
                self._genai.configure(api_key=self.api_key)
        except Exception:
            self._genai = None  # library not installed or other error

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self._genai is not None)

    def set_model(self, model: str) -> None:
        if model.strip():
            self.model_name = model.strip()
            self._model = None  # re-init next call

    def _ensure_model(self, system_prompt: str):
        if not self.enabled:
            return None
        if self._model is None:
            # system_instruction จะถูกแนบเข้าไปที่ระดับโมเดล
            self._model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
        return self._model

    def chat(self, system_prompt: str, messages: List[Tuple[str, str]]) -> Optional[str]:
        """
        messages: list of (role, content) with role in {'system','user','assistant'}.
        We pass system prompt via system_instruction; rest go as contents.
        """
        if not self.enabled:
            return None

        model = self._ensure_model(system_prompt)
        if model is None:
            return None

        # แปลงเป็น contents ตามสไตล์ Gemini
        contents = []
        for role, content in messages:
            if role == "system":
                # ข้าม เพราะใช้ system_instruction แล้ว
                continue
            r = "user" if role == "user" else "model"
            contents.append({"role": r, "parts": [content]})

        try:
            resp = model.generate_content(
                contents,
                generation_config={"temperature": 0.7},
            )
            text = getattr(resp, "text", None)
            if text:
                return text
            # เผื่อบางรุ่นคืนรูปแบบอื่น
            if hasattr(resp, "candidates") and resp.candidates:
                parts = resp.candidates[0].content.parts
                return "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            return None
        return None


# ---------- App loop ----------

def main() -> None:
    """
    Minimal conversational loop with memory, calculator, and Gemini.
    Commands:
      /help        - show help
      /mem         - show recent memory
      /clear       - clear memory
      /ai          - toggle Gemini on/off (if configured)
      /model <name>- set Gemini model name
      exit         - quit
    """
    memory = ShortTermMemory()
    calculator = SafeCalculator()
    registry = UserRegistry(path=Path(__file__).with_name("users.json"))
    ai = GeminiClient()
    ai_enabled = ai.enabled

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
        print("Gemini not configured. Set GEMINI_API_KEY (and optionally GEMINI_MODEL).")

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
            if not ai.enabled:
                print("Gemini is not configured. Set GEMINI_API_KEY first.")
            else:
                ai_enabled = not ai_enabled
                print(f"AI is now {'ON' if ai_enabled else 'OFF'}.")
            continue
        if cmd.startswith("/model "):
            _, _, model_name = user_text.partition(" ")
            if model_name.strip():
                ai.set_model(model_name.strip())
                print(f"Model set to: {ai.model_name}")
            else:
                print("Usage: /model <name>")
            continue

        # If AI is enabled, build context and ask Gemini first
        result: Optional[str] = None
        if ai_enabled:
            system_prompt = (
                "You are a helpful assistant. Personalize replies using the user's name. "
                "Be concise, reference short-term memory if relevant, and solve basic problems."
            )
            history: List[Tuple[str, str]] = []
            for u, a in memory.turns[-6:]:
                history.append(("user", u))
                history.append(("assistant", a))
            history.append(("user", f"{profile.name}: {user_text}"))

            ai_reply = ai.chat(system_prompt, history)
            if ai_reply:
                result = ai_reply

        # Fallback to local processor
        if result is None:
            result = process_input(user_text, memory, calculator)

        print(f"Result OUTPUT: {result}")
        memory.add(user_text, result)


if __name__ == "__main__":
    main()
