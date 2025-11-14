# class TTS:
#     """Lazy-loaded text-to-speech using pyttsx3."""
#     def __init__(self):
#         self._engine = None
#         self._pyttsx3 = None
#         try:
#             import pyttsx3
#             self._pyttsx3 = pyttsx3
#         except ImportError:
#             pass

#     @property
#     def available(self) -> bool:
#         return self._pyttsx3 is not None

#     def _get_engine(self):
#         if self._engine is None and self._pyttsx3:
#             try:
#                 self._engine = self._pyttsx3.init()
#                 try:
#                     rate = self._engine.getProperty("rate")
#                     if isinstance(rate, (int, float)):
#                         self._engine.setProperty("rate", int(rate * 1.1))
#                 except Exception:
#                     pass
#             except Exception:
#                 self._engine = None
#         return self._engine

#     def speak(self, text: str) -> bool:
#         if not text.strip():
#             return False
#         engine = self._get_engine()
#         if not engine:
#             return False
#         trimmed = text.strip()
#         if len(trimmed) > 220:
#             trimmed = trimmed[:217].rsplit(" ", 1)[0] + "..."
#         try:
#             engine.stop()
#             engine.say(trimmed)
#             engine.runAndWait()
#             return True
#         except Exception:
#             self._engine = None
#             return False

