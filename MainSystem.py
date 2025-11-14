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

# ---------- Load .env if available ----------

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    load_dotenv(override=False)
except ImportError:
    pass


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


# ---------- Short-term memory ----------

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


# ---------- Input processor ----------
class InputProcessor:
    """Input processor that receives input, processes with GeminiClient, and outputs results."""
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.memory = ShortTermMemory()
        self.tts = TTS()
        self.tone = "friendly"
        self.brief_mode = True
        self._cached_prompt = None
        self._cached_prompt_key = None
    
    def input(self, prompt: str = "Instruction> ") -> Optional[str]:
        """Receive input from user."""
        try:
            user_input = input(prompt).strip()
            return user_input if user_input else None
        except (EOFError, KeyboardInterrupt):
            return None
    
    def processor(self, user_text: str, user_name: str = "User") -> Optional[str]:
        """Process user input using GeminiClient."""
        if not self.gemini_client.enabled:
            return None
        
        # Build system prompt
        prompt_key = (user_name, self.tone, self.brief_mode)
        if self._cached_prompt_key != prompt_key:
            self._cached_prompt = self._build_system_prompt(user_name, self.tone, self.brief_mode)
            self._cached_prompt_key = prompt_key
        
        # Build conversation history
        history = []
        for u, a in self.memory.get_history(6):
            history.append(("user", u))
            history.append(("assistant", a))
        history.append(("user", user_text))
        
        # Process with Gemini
        result = self.gemini_client.chat(self._cached_prompt, history)
        if not result and self.gemini_client.last_error:
            return f"[AI Error] {self.gemini_client.last_error}"
        
        return result
    
    def output(self, result: str, use_speech: bool = False) -> None:
        """Output result to user."""
        if result:
            print(f"Result OUTPUT: {result}")
            if use_speech and self.tts.available:
                self.tts.speak(result)
    
    def add_to_memory(self, user_text: str, assistant_text: str) -> None:
        """Add conversation turn to memory."""
        self.memory.add(user_text, assistant_text)
    
    def _build_system_prompt(self, name: str, tone: str, brief: bool) -> str:
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

