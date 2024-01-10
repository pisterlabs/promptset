import threading
import openai
import os
import time
from gtts import gTTS
# from speak import speak
from pygame import mixer
from pydub import AudioSegment
# openai.api_key = os.environ.get("openai_key")
# from speak import speed_change


def speed_change(sound, speed=1.5):
    # Manually override the frame_rate. This will also change the pitch (unless `sound._spawn` is used).
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    }).set_frame_rate(sound.frame_rate)
    return sound_with_altered_frame_rate


all_chunks = ""

mixer.init()
openai.api_key = os.environ.get("openai_key")
print(f"key: {os.environ.get('openai_key')}")


def play_tts(text, end_line=True):
    try:
        text = text.rstrip().lstrip()
        while (mixer.music.get_busy()):
            # print("waiting for last answer...")
            pass
        mixer.music.unload()
        try:
            os.remove("tts.mp3")
            os.remove("tts_fast.mp3")
        except:
            pass
        tts = gTTS(text, lang='iw')
        tts.save("tts.mp3")
        sound = AudioSegment.from_file("tts.mp3", format="mp3")
        # Speed up by 1.5 times (you can adjust the value to your liking)
        fast_sound = speed_change(sound, 1.4)
        fast_sound.export("tts_fast.mp3", format="mp3")
        mixer.music.load("tts_fast.mp3")
        print(text, end=('\n' if end_line else ' '))
        mixer.music.play()
    except Exception as e:
        print(e)


def run_ask_gpt(prompt):
    x = threading.Thread(target=ask_gpt4, args=(prompt,), daemon=True)
    x.start()


def ask_gpt(prompt, model="gpt-3.5-turbo"):
    # print("debug, ask gpt")
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0, stream=True)
    chunks = ""
    all_chunks = ""
    i = 0
    for chunk in response:
        i += 1
        response_text = chunk.choices[0].delta.get("content", "")
        # print(response_text)
        chunks += response_text
        all_chunks += response_text
        if '\n' in chunks or '.' in chunks or i > 100:
            print(chunks)
            if i >= 2:
                play_tts(chunks, end_line=('\n' in chunks))
            # speak(chunks)
            chunks = ""
            i = 0
    play_tts(chunks)
    while (mixer.music.get_busy()):
        # print("waiting for last answer...")
        time.sleep(0.1)
    # speak(chunks)
    # return all_chunks
    print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    print(all_chunks)
    return all_chunks


def ask_gpt3(prompt):
    ask_gpt(prompt, model="gpt-3.5-turbo")


def ask_gpt4(prompt):
    return ask_gpt(prompt, model="gpt-4")
