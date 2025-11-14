import os
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv
from collections import deque

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

r = sr.Recognizer()
mic = sr.Microphone()
tts = pyttsx3.init()
memory = deque(maxlen=6)

SYSTEM = (
    "You are Kaew, a warm human-like AI friend. "
    "Speak naturally, casual, emotional, real. "
    "Understand feelings, maintain context, adapt tone. "
    "Keep answers concise unless asked. "
    "Think deeply using implicit chain-of-thought reasoning, "
    "but respond with clear final answers only."
)

def listen():
    with mic as s:
        r.adjust_for_ambient_noise(s)
        audio = r.listen(s)
    try:
        return r.recognize_google(audio, language='th-TH')
    except:
        return None

def speak(t):
    print("AI:", t)
    tts.say(t)
    tts.runAndWait()

def build_prompt(user_text):
    h = "".join(f"User: {u}\nAI: {a}\n" for u,a in memory)
    return SYSTEM + "\n" + h + f"\nUser: {user_text}\nAI:"

def ask(user_text):
    prompt = build_prompt(user_text)
    resp = model.generate_content(prompt)
    reply = resp.text.strip()
    memory.append((user_text, reply))
    return reply

def main():
    print("ready")
    while True:
        text = listen()
        if not text:
            continue
        print("You:", text)
        if text in ["หยุด", "พอแล้ว", "ออก"]:
            speak("โอเค")
            break
        speak(ask(text))

if __name__ == "__main__":
    main()
