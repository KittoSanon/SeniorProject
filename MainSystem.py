import os
import asyncio
import uuid
import webbrowser
import subprocess
import speech_recognition as sr
import pygame
import edge_tts
import google.generativeai as genai
from dotenv import load_dotenv
from collections import deque

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction=(
        "You are Kaew, a warm Thai girl AI friend who speaks naturally without emojis. "
        "You can work as an agent: open YouTube, open websites, open apps. "
        "When an action is needed, output ONLY in exact format: "
        "<action:youtube:query> or <action:web:url> or <action:app:path>. "
        "Do NOT include any additional text inside the action tag."
    )
)

VOICE = "th-TH-AcharaNeural"

r = sr.Recognizer()
mic = sr.Microphone()
memory = deque(maxlen=6)

pygame.mixer.init()


async def speak(text):
    temp = f"tts_{uuid.uuid4().hex}.mp3"
    tts = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="+12%",
        pitch="+10Hz"
    )
    await tts.save(temp)

    pygame.mixer.music.load(temp)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.05)

    try: os.remove(temp)
    except: pass


def listen():
    with mic as s:
        r.adjust_for_ambient_noise(s)
        audio = r.listen(s)
    try:
        return r.recognize_google(audio, language="th-TH")
    except:
        return None


def build_msgs():
    msgs = []
    for u, a in memory:
        msgs.append({"role": "user", "parts": [{"text": u}]})
        msgs.append({"role": "model", "parts": [{"text": a}]})
    return msgs


def do_action(action: str):
    parts = action.split(":", 2)

    if len(parts) < 2:
        return

    kind = parts[0]
    value = parts[1]

    if kind == "youtube":
        url = "https://www.youtube.com/results?search_query=" + value
        webbrowser.open(url)

    elif kind == "web":
        webbrowser.open(value)

    elif kind == "app":
        subprocess.Popen(value, shell=True)


def ask(text):
    msgs = build_msgs()
    msgs.append({"role": "user", "parts": [{"text": text}]})

    resp = model.generate_content(msgs)
    reply = resp.text.strip()

    memory.append((text, reply))

    # -----------------------------
    # Handle Action command
    # -----------------------------
    if reply.startswith("<action:") and reply.endswith(">"):
        action_content = reply.replace("<action:", "").replace(">", "")
        do_action(action_content)
        return None

    return reply


async def main():
    print("ready")

    while True:
        text = listen()
        if not text:
            continue

        print("You:", text)

        if text in ["หยุด", "ออก", "เลิก", "พอ"]:
            await speak("ไว้คุยกันใหม่นะ")
            break

        reply = ask(text)

        if reply:
            print("AI:", reply)
            await speak(reply)


if __name__ == "__main__":
    asyncio.run(main())
