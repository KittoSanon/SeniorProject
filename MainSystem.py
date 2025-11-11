from __future__ import annotations

import ast
import operator
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone
import json
import os

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    load_dotenv(override=False)
except ImportError:
    pass

# ---------- Core components ----------

class ShortTermMemory:
    """In-memory buffer using deque for O(1) operations."""
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))

    def clear(self) -> None:
        self.turns.clear()

    def get_history(self, window: int) -> list[tuple[str, str]]:
        return list(self.turns)[-window:]

    def summary(self) -> str:
        return "\n".join(f"U: {u}\nA: {a}" for u, a in self.turns) or "(empty)"


class SafeCalculator:
    """Safe arithmetic expression evaluator."""
    _ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def try_calculate(self, text: str) -> Optional[float]:
        try:
            node = ast.parse(text.strip(), mode="eval")
            return self._eval(node.body)
        except Exception:
            return None

    def _eval(self, node) -> float:
        if isinstance(node, ast.BinOp) and type(node.op) in self._ops:
            return self._ops[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._ops:
            return self._ops[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Num):  # Python <3.8
            return node.n
        raise ValueError("Unsupported")


def short_answer_local(text: str) -> Optional[str]:
    """Fast local answers for simple queries."""
    t = text.strip().lower()
    if not t:
        return None
    
    now = datetime.now()
    if "day" in t and ("what" in t or "which" in t or "today" in t):
        return f"today is {now.strftime('%A').lower()}!"
    if "date" in t and ("what" in t or "today" in t):
        return f"today is {now.strftime('%Y-%m-%d').lower()}."
    if "time" in t and ("what" in t or "now" in t or "current" in t):
        return f"the time is {now.strftime('%H:%M').lower()}."
    if "month" in t and ("what" in t or "today" in t):
        return f"it is {now.strftime('%B').lower()}."
    if "year" in t and ("what" in t or "now" in t):
        return f"it is {now.strftime('%Y').lower()}."
    return None


class TTS:
    """Lazy-loaded text-to-speech using pyttsx3."""
    def __init__(self):
        self._engine = None
        self._pyttsx3 = None
        try:
            import pyttsx3
            self._pyttsx3 = pyttsx3
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._pyttsx3 is not None

    def _get_engine(self):
        if self._engine is None and self._pyttsx3:
            try:
                self._engine = self._pyttsx3.init()
                try:
                    rate = self._engine.getProperty("rate")
                    if isinstance(rate, (int, float)):
                        self._engine.setProperty("rate", int(rate * 1.1))
                except Exception:
                    pass
            except Exception:
                self._engine = None
        return self._engine

    def speak(self, text: str) -> bool:
        if not text.strip():
            return False
        engine = self._get_engine()
        if not engine:
            return False
        trimmed = text.strip()
        if len(trimmed) > 220:
            trimmed = trimmed[:217].rsplit(" ", 1)[0] + "..."
        try:
            engine.stop()
            engine.say(trimmed)
            engine.runAndWait()
            return True
        except Exception:
            self._engine = None
            return False


def process_input(user_text: str, memory: ShortTermMemory, calculator: SafeCalculator) -> str:
    """Fallback processor when AI is unavailable."""
    result = calculator.try_calculate(user_text)
    if result is not None:
        return f"The result is {int(result) if result.is_integer() else result}"
    count = len(memory.turns)
    if user_text.strip().endswith("?"):
        return f"You asked a question. I remember {count} turn(s)."
    return f"I hear you: '{user_text.strip()}'. I remember {count} turn(s)."


# ---------- User recognition ----------

@dataclass
class UserProfile:
    name: str
    created_at_iso: str
    last_seen_iso: str

    @staticmethod
    def new(name: str) -> "UserProfile":
        now = datetime.now(timezone.utc).isoformat()
        return UserProfile(name, now, now)

    def touch(self) -> None:
        self.last_seen_iso = datetime.now(timezone.utc).isoformat()


class UserRegistry:
    """Persistent user registry with optimized I/O."""
    def __init__(self, path: Path):
        self.path = path
        self._users: dict[str, dict] = self._load()
        self._dirty = False

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        if not self._dirty:
            return
        try:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._users, f, indent=2)
            tmp.replace(self.path)
            self._dirty = False
        except Exception:
            pass

    def get_or_create(self, name: str) -> UserProfile:
        key = name.strip().lower()
        if not key:
            raise ValueError("User name cannot be empty")
        raw = self._users.get(key)
        if raw is None:
            profile = UserProfile.new(name.strip())
            self._users[key] = asdict(profile)
            self._dirty = True
        else:
            profile = UserProfile(**raw)
            profile.touch()
            self._users[key] = asdict(profile)
            self._dirty = True
        self._save()
        return profile


# ---------- Gemini provider ----------

class GeminiClient:
    """Optimized Gemini client with model caching."""
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest").strip()
        self._model = None
        self._model_prompt = None
        self.last_error: Optional[str] = None
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.8") or 0.8)
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512") or 512)
        self._max_tokens_default = self.max_output_tokens
        try:
            import google.generativeai as genai
            self._genai = genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
        except ImportError:
            self._genai = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self._genai)

    def set_model(self, model: str) -> None:
        if model := model.strip():
            self.model_name = model
            self._model = None
            self._model_prompt = None

    def set_temperature(self, temp: float) -> None:
        self.temperature = max(0.0, min(1.0, float(temp)))

    def set_max_output_tokens(self, value: int) -> None:
        try:
            if 0 < (v := int(value)) <= 2048:
                self.max_output_tokens = max(32, v)
        except (ValueError, TypeError):
            pass

    def reset_max_output_tokens(self) -> None:
        self.max_output_tokens = self._max_tokens_default

    def list_models(self) -> list[tuple[str, bool]]:
        if not self.enabled:
            return []
        try:
            return [
                (m.name, "generateContent" in (m.supported_generation_methods or []))
                for m in self._genai.list_models()
                if hasattr(m, "name")
            ]
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []

    def _get_model(self, system_prompt: str):
        if not self.enabled:
            return None
        if self._model is None or self._model_prompt != system_prompt:
            self._model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
            self._model_prompt = system_prompt
        return self._model

    def chat(self, system_prompt: str, messages: list[tuple[str, str]]) -> Optional[str]:
        if not self.enabled:
            return None
        model = self._get_model(system_prompt)
        if not model:
            return None
        contents = [
            {"role": "user" if r == "user" else "model", "parts": [c]}
            for r, c in messages if r != "system"
        ]
        try:
            resp = model.generate_content(
                contents,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "candidate_count": 1,
                    "response_mime_type": "text/plain",
                },
            )
            if resp.candidates and (cand := resp.candidates[0]):
                if content := getattr(cand, "content", None):
                    if parts := getattr(content, "parts", None):
                        text = "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
                        if text := text.strip():
                            self.last_error = None
                            return text
                self.last_error = f"No text (finish_reason={getattr(cand, 'finish_reason', 'unknown')})"
            else:
                self.last_error = "Empty response"
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
        return None


# ---------- App loop ----------

def _build_system_prompt(name: str, tone: str, brief: bool) -> str:
    """Build cached system prompt."""
    parts = [
        "You are a warm, human-like assistant.",
        f"Adopt a {tone} tone with natural, conversational phrasing and contractions.",
        f"The user's name is {name}. Avoid repeating the name unless greeting them explicitly or it is contextually necessary.",
        "Be concise but helpful. Use plain language and avoid sounding robotic.",
        "Reference the short-term memory if relevant to keep continuity.",
        "When appropriate, end with a brief, friendly follow-up question to keep the flow.",
        "Do not overuse emojis; use them sparingly only if tone is playful.",
    ]
    if brief:
        parts.extend([
            "Keep responses short and crystal-clear. Aim for 1–2 concise sentences.",
            "Prefer plain text answers without preamble. Avoid repeating the question.",
        ])
    if tone == "playful":
        parts.append("Allow light humor and the occasional emoji when it makes sense.")
    elif tone == "professional":
        parts.append("Keep a clear, calm, and respectful tone without slang.")
    return " ".join(parts)


def main() -> None:
    """Optimized conversational loop with command dispatcher."""
    memory = ShortTermMemory()
    calculator = SafeCalculator()
    registry = UserRegistry(Path(__file__).with_name("users.json"))
    ai = GeminiClient()
    tts = TTS()
    
    # State
    ai_enabled = ai.enabled
    speech_enabled = tts.available
    tone = "friendly"
    fast_mode = False
    fast_mode_prev_tokens = ai.max_output_tokens
    brief_mode = True
    short_mode = True

    # Get user name
    print("Welcome. Before we chat, what's your name?")
    while True:
        try:
            if name := input("Name> ").strip():
                break
            print("Please enter a non-empty name.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

    profile = registry.get_or_create(name)
    print(f"{'Nice to meet you' if profile.created_at_iso == profile.last_seen_iso else 'Welcome back'}, {profile.name}!")
    print("Main system ready. Type 'exit' to quit or '/help' for commands.")
    if not ai_enabled:
        print("Gemini not configured. Set GEMINI_API_KEY.")

    # Main loop
    cached_prompt = None
    cached_prompt_key = None
    while True:
        try:
            user_text = input("Instruction> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() == "exit":
            print("Goodbye.")
            break

        cmd_parts = user_text.split(None, 1)
        cmd = cmd_parts[0].lower()
        arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

        # Handle commands
        if cmd == "/help":
            print("Commands: /mem, /clear, /ai, /model <name>, /models, /fast, /brief, /short,")
            print("/speak, /temp <0-1>, /tone <friendly|professional|playful>, /status, /debug, exit")
            continue
        elif cmd == "/mem":
            print("Memory:\n" + memory.summary())
            continue
        elif cmd == "/clear":
            memory.clear()
            print("Memory cleared.")
            continue
        elif cmd == "/ai":
            if not ai.enabled:
                print("AI not configured.")
            else:
                ai_enabled = not ai_enabled
                print(f"AI: {'ON' if ai_enabled else 'OFF'}")
            continue
        elif cmd == "/temp":
            try:
                ai.set_temperature(float(arg))
                print(f"Temperature: {ai.temperature}")
            except ValueError:
                print("Usage: /temp <0-1>")
            continue
        elif cmd == "/tone":
            if arg.lower() in {"friendly", "professional", "playful"}:
                tone = arg.lower()
                cached_prompt = None
                cached_prompt_key = None
                print(f"Tone: {tone}")
            else:
                print("Usage: /tone <friendly|professional|playful>")
            continue
        elif cmd == "/model":
            if arg:
                ai.set_model(arg)
                print(f"Model: {ai.model_name}")
            else:
                print("Usage: /model <name>")
            continue
        elif cmd == "/models":
            if not ai.enabled:
                print("AI not configured.")
            else:
                models = [name for name, supports in ai.list_models() if supports]
                if models:
                    print("Available models:")
                    for m in models:
                        print(f"  - {m}")
                else:
                    print("No models found." + (f" Error: {ai.last_error}" if ai.last_error else ""))
            continue
        elif cmd == "/fast":
            fast_mode = not fast_mode
            if fast_mode:
                fast_mode_prev_tokens = ai.max_output_tokens
                ai.set_max_output_tokens(min(ai.max_output_tokens, 256))
            else:
                ai.set_max_output_tokens(fast_mode_prev_tokens)
            print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
            continue
        elif cmd == "/brief":
            brief_mode = not brief_mode
            cached_prompt = None
            cached_prompt_key = None
            print(f"Brief mode: {'ON' if brief_mode else 'OFF'}")
            continue
        elif cmd == "/short":
            short_mode = not short_mode
            print(f"Short mode: {'ON' if short_mode else 'OFF'}")
            continue
        elif cmd == "/speak":
            if not tts.available:
                print("TTS not available. Install 'pyttsx3'.")
            else:
                speech_enabled = not speech_enabled
                print(f"Speech: {'ON' if speech_enabled else 'OFF'}")
            continue
        elif cmd == "/status":
            print(f"AI: {'ON' if ai_enabled else 'OFF'} ({'configured' if ai.enabled else 'not configured'})")
            print(f"Model: {ai.model_name}, Temp: {ai.temperature}, Tokens: {ai.max_output_tokens}")
            print(f"Fast: {fast_mode}, Brief: {brief_mode}, Short: {short_mode}, Speech: {speech_enabled}")
            continue
        elif cmd == "/debug":
            print(f"API Key: {'set' if ai.api_key else 'not set'}")
            print(f"GenAI lib: {'installed' if ai._genai else 'not installed'}")
            print(f"AI enabled: {ai.enabled}, Last error: {ai.last_error or 'none'}")
            continue

        # Process user input
        result = None
        if short_mode and (result := short_answer_local(user_text)):
            pass
        elif ai_enabled:
            # Cache system prompt
            prompt_key = (profile.name, tone, brief_mode)
            if cached_prompt_key != prompt_key:
                cached_prompt = _build_system_prompt(profile.name, tone, brief_mode)
                cached_prompt_key = prompt_key
            
            # Build history
            window = 2 if fast_mode else 6
            history = []
            for u, a in memory.get_history(window):
                history.append(("user", u))
                history.append(("assistant", a))
            history.append(("user", user_text))
            
            result = ai.chat(cached_prompt, history)
            if not result and ai.last_error:
                print(f"[AI] {ai.last_error}")

        if not result:
            result = process_input(user_text, memory, calculator)

        print(f"Result OUTPUT: {result}")
        if speech_enabled:
            tts.speak(result)
        memory.add(user_text, result)


if __name__ == "__main__":
    main()
