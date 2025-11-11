from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime, timezone
import os
 
# Load environment variables from a local .env file if present
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    _ENV_LOADED = False
    # Prefer .env next to this script
    try:
        _ENV_LOADED = load_dotenv(dotenv_path=Path(__file__).with_name(".env")) or _ENV_LOADED
    except Exception:
        pass
    # Also try to locate any .env from current/parent directories (e.g., when running from project root)
    try:
        _found_env = find_dotenv(usecwd=True)
        if _found_env:
            _ENV_LOADED = load_dotenv(_found_env) or _ENV_LOADED
    except Exception:
        pass
except Exception:
    # dotenv is optional; if not installed, environment must be set by the OS/shell
    _ENV_LOADED = False

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


def short_answer_local(user_text: str) -> Optional[str]:
    """
    Extremely concise local answers for simple queries, bypassing AI.
    Returns a lowercase, short string ending with '!' when appropriate.
    """
    t = user_text.strip().lower()
    if not t:
        return None

    now = datetime.now()

    # What day is it today?
    if any(kw in t for kw in ["what day", "what's the day", "day today", "today day", "which day"]):
        return f"today is {now.strftime('%A').lower()}!"

    # What date is it today?
    if any(kw in t for kw in ["what date", "date today", "today date"]):
        return f"today is {now.strftime('%Y-%m-%d').lower()}."

    # What time is it?
    if any(kw in t for kw in ["what time", "time now", "current time"]):
        return f"the time is {now.strftime('%H:%M').lower()}."

    # What month / year?
    if "what month" in t or "month today" in t:
        return f"it is {now.strftime('%B').lower()}."
    if "what year" in t or "year now" in t:
        return f"it is {now.strftime('%Y').lower()}."

    return None


class SpeechSynthesizer:
    """Optional text-to-speech using pyttsx3."""

    def __init__(self) -> None:
        try:
            import pyttsx3  # type: ignore

            self._pyttsx3 = pyttsx3
        except Exception:
            self._pyttsx3 = None
        self._engine = None
        self._initialized = False
        self._init_engine()

    def _init_engine(self) -> None:
        if self._pyttsx3 is None:
            return
        try:
            self._engine = self._pyttsx3.init()
            # Slightly faster voice to keep latency low
            try:
                rate = self._engine.getProperty("rate")
                if rate and isinstance(rate, (int, float)):
                    self._engine.setProperty("rate", int(rate * 1.1))
            except Exception:
                pass
            self._initialized = True
        except Exception:
            self._engine = None
            self._initialized = False

    def _ensure_engine(self):
        if self._engine is None and self._pyttsx3 is not None:
            self._init_engine()
        return self._engine

    @property
    def available(self) -> bool:
        return self._ensure_engine() is not None

    def speak(self, text: str) -> None:
        engine = self._ensure_engine()
        if engine is None or not text.strip():
            return
        # Limit overly long passages
        snippet = text.strip()
        if len(snippet) > 220:
            snippet = snippet[:217].rsplit(" ", 1)[0].rstrip() + "..."
        try:
            try:
                engine.stop()
            except Exception:
                pass
            engine.say(snippet)
            engine.runAndWait()
        except Exception:
            # Try reinitializing once
            self._engine = None
            engine = self._ensure_engine()
            if engine is None:
                return
            try:
                engine.stop()
            except Exception:
                pass
            try:
                engine.say(snippet)
                engine.runAndWait()
            except Exception:
                self._engine = None


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
        now = datetime.now(timezone.utc).isoformat()
        return UserProfile(name=name, created_at_iso=now, last_seen_iso=now)

    def touch(self) -> None:
        self.last_seen_iso = datetime.now(timezone.utc).isoformat()


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
        # Use a more widely-available alias by default
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest").strip()
        self._model = None
        self.last_error: Optional[str] = None
        # Generation controls
        try:
            self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.8"))
        except Exception:
            self.temperature = 0.8
        try:
            self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
        except Exception:
            self.max_output_tokens = 512
        self._max_output_tokens_default = self.max_output_tokens

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

    def set_temperature(self, temperature: float) -> None:
        # Clamp temperature for safety
        t = max(0.0, min(1.0, float(temperature)))
        self.temperature = t

    def set_max_output_tokens(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            return
        if v <= 0:
            return
        # Clamp to reasonable bounds for latency vs. completeness
        self.max_output_tokens = max(32, min(2048, v))

    def reset_max_output_tokens(self) -> None:
        self.max_output_tokens = self._max_output_tokens_default

    def list_models(self) -> List[Tuple[str, bool]]:
        """
        Returns a list of (model_name, supports_generate_content).
        """
        if not self.enabled:
            return []
        try:
            models = []
            for m in self._genai.list_models():  # type: ignore
                name = getattr(m, "name", "")
                # supported_generation_methods might include "generateContent" or "createContent"
                methods = set(getattr(m, "supported_generation_methods", []) or [])
                supports = "generateContent" in methods or "createContent" in methods
                models.append((name, supports))
            return models
        except Exception as e:
            self.last_error = f"ListModels error: {type(e).__name__}: {e}"
            return []

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
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "candidate_count": 1,
                    "response_mime_type": "text/plain",
                },
            )

            # Avoid using resp.text quick accessor; inspect candidates safely
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                finish_reason = getattr(cand, "finish_reason", None)
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if parts:
                    text_out = "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
                    if text_out.strip():
                        self.last_error = None
                        return text_out.strip()
                # No parts or empty text; record reason
                if finish_reason is not None:
                    self.last_error = f"No text in response (finish_reason={finish_reason})"
                else:
                    self.last_error = "No text in response"
                return None

            self.last_error = "Empty response (no candidates)"
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        return None


# ---------- Text-to-Speech (optional) ----------

class TtsSpeaker:
    """
    Simple local TTS using pyttsx3 (offline). If pyttsx3 is not installed, stays disabled.
    """
    def __init__(self) -> None:
        self._engine = None
        self.last_error: Optional[str] = None
        try:
            import pyttsx3  # type: ignore
            self._pyttsx3 = pyttsx3
            self._engine = pyttsx3.init()
            try:
                self._engine.setProperty("rate", int(os.getenv("TTS_RATE", "185")))
            except Exception:
                pass
            try:
                self._engine.setProperty("volume", float(os.getenv("TTS_VOLUME", "1.0")))
            except Exception:
                pass
        except Exception as e:
            self._pyttsx3 = None  # type: ignore
            self.last_error = f"TTS init error: {type(e).__name__}: {e}"

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def speak(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            trimmed = text.strip()
            if len(trimmed) > 600:
                trimmed = trimmed[:600] + "..."
            self._engine.say(trimmed)
            self._engine.runAndWait()
            return True
        except Exception as e:
            self.last_error = f"TTS speak error: {type(e).__name__}: {e}"
            return False

    def list_voices(self) -> List[Tuple[str, str]]:
        if not self.enabled:
            return []
        try:
            voices = self._engine.getProperty("voices")
            out: List[Tuple[str, str]] = []
            for v in voices:
                vid = getattr(v, "id", "")
                vname = getattr(v, "name", "")
                out.append((vid, vname))
            return out
        except Exception as e:
            self.last_error = f"TTS list voices error: {type(e).__name__}: {e}"
            return []

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
    speaker = SpeechSynthesizer()
    speech_enabled = speaker.available

    print("Welcome. Before we chat, what's your name?")
    # Conversational style controls
    selected_tone = "friendly"  # friendly | professional | playful
    fast_mode = False
    fast_mode_prev_tokens = ai.max_output_tokens
    brief_mode = True  # keep answers short and clear by default
    short_mode = True  # ultra-short one-liners for simple questions

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
    print("Commands: /help, /mem, /clear, /ai, /model <name>, /fast")
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
            print("Commands:")
            print("- /mem                Show short-term memory")
            print("- /clear              Clear memory")
            print("- /ai                 Toggle AI on/off")
            print("- /model <name>       Set model")
            print("- /models             List available models")
            print("- /fast               Toggle fast mode (short context & shorter replies)")
            print("- /brief              Toggle brief mode (1–2 sentences, very clear)")
            print("- /short              Toggle ultra-short answers for simple queries")
            print("- /speak              Toggle text-to-speech playback")
            print("- /temp <0-1>         Set creativity (temperature)")
            print("- /tone <friendly|professional|playful>  Set tone")
            print("- /status             AI status")
            print("- /debug              Diagnostics")
            print("- exit                Quit")
            continue
        if cmd.startswith("/temp "):
            _, _, t_str = user_text.partition(" ")
            try:
                t_val = float(t_str.strip())
                ai.set_temperature(t_val)
                print(f"Temperature set to: {ai.temperature}")
            except Exception:
                print("Usage: /temp <number between 0 and 1>")
            continue
        if cmd.startswith("/tone "):
            _, _, tone_val = user_text.partition(" ")
            tone_val = tone_val.strip().lower()
            if tone_val in {"friendly", "professional", "playful"}:
                # stash on the fly on the function scope via closure variable trick
                selected_tone = tone_val
                print(f"Tone set to: {tone_val}")
            else:
                print("Usage: /tone <friendly|professional|playful>")
            continue
        if cmd == "/mem":
            print("Short term memory:")
            print(memory.summary())
            continue
        if cmd == "/clear":
            memory.clear()
            print("Memory cleared.")
            continue
        if cmd == "/fast":
            fast_mode = not fast_mode
            if fast_mode:
                fast_mode_prev_tokens = ai.max_output_tokens
                ai.set_max_output_tokens(min(ai.max_output_tokens, 256))
                print("Fast mode ON: using shorter history window and capped reply length to lower latency.")
            else:
                ai.set_max_output_tokens(fast_mode_prev_tokens)
                print("Fast mode OFF: restored previous reply length and context window.")
            continue
        if cmd == "/brief":
            brief_mode = not brief_mode
            print(f"Brief mode {'ON' if brief_mode else 'OFF'}: answers will be {'short and clear' if brief_mode else 'normal length'}.")
            continue
        if cmd == "/short":
            short_mode = not short_mode
            print(f"Short mode {'ON' if short_mode else 'OFF'}: ultra-brief one-line answers for simple questions.")
            continue
        if cmd == "/speak":
            if not speaker.available:
                print("Text-to-speech engine not available. Install 'pyttsx3' and restart.")
                continue
            speech_enabled = not speech_enabled
            print(f"Speech {'enabled' if speech_enabled else 'disabled'}.")
            continue
        if cmd == "/models":
            if not ai.enabled:
                print("AI not configured. Set GEMINI_API_KEY, then try again.")
                continue
            items = ai.list_models()
            if not items:
                print("No models returned, or error occurred.")
                if ai.last_error:
                    print(f"Last error: {ai.last_error}")
                continue
            print("Available models (supports text generation):")
            shown = 0
            for name, supports in items:
                # Show only text-capable models to keep output short
                if supports:
                    print(f"- {name}")
                    shown += 1
            if shown == 0:
                print("(None with generateContent/createContent support found. Try different API key or account.)")
            print("Use '/model <name>' to switch.")
            continue
        if cmd == "/status":
            print(f"AI configured: {ai.enabled}")
            print(f"AI toggled ON now: {ai_enabled}")
            print(f"Model name: {ai.model_name}")
            print(f"google-generativeai installed: {getattr(ai, '_genai', None) is not None}")
            print(f"GEMINI_API_KEY present: {bool(os.getenv('GEMINI_API_KEY'))}")
            print(f"Temperature: {ai.temperature}")
            print(f"Max output tokens: {ai.max_output_tokens}")
            print(f"Fast mode: {fast_mode}")
            print(f"Brief mode: {brief_mode}")
            print(f"Short mode: {short_mode}")
            print(f"Speech available: {speaker.available}")
            print(f"Speech enabled: {speech_enabled and speaker.available}")
            continue
        if cmd == "/debug":
            env_path = Path(__file__).with_name(".env")
            key = os.getenv("GEMINI_API_KEY", "")
            print("Debug info:")
            print(f"- cwd: {os.getcwd()}")
            print(f"- script: {__file__}")
            print(f"- .env sibling path: {str(env_path)} (exists: {env_path.exists()})")
            # Whether we loaded any .env according to startup
            try:
                loaded_flag = _ENV_LOADED  # type: ignore
            except Exception:
                loaded_flag = False
            print(f"- dotenv loaded: {loaded_flag}")
            print(f"- has GEMINI_API_KEY: {bool(key)}")
            print(f"- GEMINI_API_KEY length: {len(key)}")
            print(f"- GEMINI_MODEL: {os.getenv('GEMINI_MODEL', '(unset)')}")
            print(f"- google-generativeai installed: {ai._genai is not None}")  # type: ignore
            print(f"- ai.enabled: {ai.enabled}")
            print(f"- last AI error: {ai.last_error}")
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

        # Ultra-short local replies for simple queries (bypass AI entirely)
        if short_mode:
            local_short = short_answer_local(user_text)
            if local_short is not None:
                result = local_short

        if ai_enabled and result is None:
            # Compose a human-like system prompt based on tone
            base_prompt = [
                "You are a warm, human-like assistant.",
                f"Adopt a {selected_tone} tone with natural, conversational phrasing and contractions.",
                f"The user's name is {profile.name}. Avoid repeating the name unless greeting them explicitly or it is contextually necessary.",
                "Be concise but helpful. Use plain language and avoid sounding robotic.",
                "Reference the short-term memory if relevant to keep continuity.",
                "When appropriate, end with a brief, friendly follow-up question to keep the flow.",
                "Do not overuse emojis; use them sparingly only if tone is playful.",
            ]
            if brief_mode:
                base_prompt.append("Keep responses short and crystal-clear. Aim for 1–2 concise sentences.")
                base_prompt.append("Prefer plain text answers without preamble. Avoid repeating the question.")
            if selected_tone == "playful":
                base_prompt.append("Allow light humor and the occasional emoji when it makes sense.")
            if selected_tone == "professional":
                base_prompt.append("Keep a clear, calm, and respectful tone without slang.")
            system_prompt = " ".join(base_prompt)
            history: List[Tuple[str, str]] = []
            history_window = 2 if fast_mode else 6
            for u, a in memory.turns[-history_window:]:
                history.append(("user", u))
                history.append(("assistant", a))
            history.append(("user", user_text))

            ai_reply = ai.chat(system_prompt, history)
            if ai_reply:
                result = ai_reply
            else:
                # Surface reason to user for easier troubleshooting
                if ai.enabled:
                    err = ai.last_error or "Unknown AI error"
                    print(f"[AI notice] Could not get AI answer, falling back. Reason: {err}")
                else:
                    print("[AI notice] AI not configured. Use /debug and /status to troubleshoot.")

        # Fallback to local processor
        if result is None:
            result = process_input(user_text, memory, calculator)

        print(f"Result OUTPUT: {result}")
        if speech_enabled and speaker.available:
            speaker.speak(result)
        memory.add(user_text, result)


if __name__ == "__main__":
    main()
