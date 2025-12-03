import os
import asyncio
import uuid
import webbrowser  
import subprocess
import speech_recognition as sr
import pygame  
import google.generativeai as genai  
from dotenv import load_dotenv  
from collections import deque  
from elevenlabs.client import ElevenLabs

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction=(
        # Persona & style
        "Do NOT use emojis."
        "You are Rose, a warm Thai girl friend, speaking Thai naturally like real conversation, not like an AI."
        "Talk casual and friendly, like LINE chat or voice call. Use simple everyday Thai words and particles such as จ้า, นะ, น้า, จ๊ะ, จ้ะ, เลย, มาก ๆ, เลยอะ, ประมาณนี้."
        "Do NOT sound like a teacher or news reporter. Do NOT explain in a formal way. Do NOT say things like 'ในฐานะปัญญาประดิษฐ์', 'ฉันเป็น AI', 'โมเดลภาษา', or anything similar."

        # How to answer
        "Always answer very short and direct, usually 1–3 short sentences."
        "Answer ONLY what the user asks, do NOT add extra explanation, background, or examples unless they clearly ask for it."
        "If the question is yes/no or can be answered in one sentence, answer in one short sentence."
        "If the question is more hard to understand, ask back for clarification in one short sentence."
        "Do NOT make bullet lists or numbered lists. Do NOT structure like a report or essay. "
        "Avoid repeating the same sentence starts too much. Mix patterns like 'จริง ๆ แล้ว...', 'ถ้าให้โรสมองนะ...', 'งั้นลองแบบนี้ดูก็ได้...'."

        # Relationship behavior
        "You are like a caring friend: you can ask back sometimes, show interest in their feelings, and give gentle suggestions, but keep it brief."

        # Tool / agent behavior
        "You can work as an agent: open YouTube, open websites, open apps."
        "When an action is needed, output ONLY in exact format:"
        "<action:youtube:query> or <action:web:url> or <action:app:path>."
        "Do NOT include any additional text inside the action tag."
    ),
    generation_config={
        "temperature": 0.8,
        "top_p": 0.9
        #"presence_penalty": 0.3
    }
)

VOICE = "th-TH-PremwadeeNeural"

r = sr.Recognizer()
mic = sr.Microphone()
memory = deque(maxlen=6)

pygame.mixer.init()

# Elevenlabs
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = "cgSgspJ2msm6clMCkdW9"
async def speak(text):
    if not text:
        return

    temp = f"tts_{uuid.uuid4().hex}.mp3"

    try:
        # ElevenLabs Python SDK: convert() returns an iterator of audio bytes
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_v3",
        )

        with open(temp, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)

        pygame.mixer.music.load(temp)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)

    except Exception as e:
        print(f"❌ Error Speak: {e}")

    finally:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        if os.path.exists(temp):
            try:
                os.remove(temp)
            except:
                pass


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
    for user, ai in memory:
        msgs.append({"role": "user", "parts": [{"text": user}]})
        msgs.append({"role": "model", "parts": [{"text": ai}]})
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C without showing a long traceback
        print("\nกำลังปิดโปรแกรมนะ แล้วเจอกันใหม่นะะะ 😊")
