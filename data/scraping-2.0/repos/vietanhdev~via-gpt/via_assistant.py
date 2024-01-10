import os
import time
import openai
import threading
from datetime import datetime
from dotenv import load_dotenv

import pvporcupine
from pvrecorder import PvRecorder

import speech_recognition as sr
from gtts import gTTS
import playsound
from vnm_utils import check_pattern_in_text, standardize_tone

load_dotenv()
openai.api_key=os.environ.get("OPENAI_KEY", "")

speech_recognizer = sr.Recognizer()
with sr.Microphone() as source:
    speech_recognizer.adjust_for_ambient_noise(source)
trigger_voice = False
last_trigged_time = 0

def play_audio_from_text(text):
    """Play audio from text"""
    tts = gTTS(text, lang='vi')
    tts.save("output.mp3")
    playsound.playsound('output.mp3', True)

def listen_and_respond():
    """Listen and response to user commands"""
    print("Listening...")
    playsound.playsound("activate.wav", True)
    r = sr.Recognizer()
    r.pause_threshold = 2
    with sr.Microphone() as source:
        print ('Say Something!')
        audio = r.listen(source, timeout=5)
        print ('Done!')
        try:
            text = r.recognize_google(audio, language='vi-VN')
        except:
            print("Could not listen to you!")
            return
        print(text)

    # Process text
    text = text.lower()
    text = standardize_tone(text)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Bạn là VIA, trí tuệ nhân tạo được phát triển bởi Maker Việt. Hãy nói tiếng Việt khi được hỏi bất cứ điều gì."}, {"role": "user", "content": f"{text}"}])
    response_text = response.choices[0].message.content
    play_audio_from_text(response_text)


def trigger_wakeword():
    """Activate listening
    """
    global trigger_voice
    trigger_voice = True

def recognize_wake_word_thread():
    print('Listening ... (press Ctrl+C to exit)')
    access_key = os.environ.get("PORCUPINE_KEY", "")
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=["OK-VIA_en_raspberry-pi_v3_0_0.ppn"],
        sensitivities=[0.5],
    )
    recorder = PvRecorder(
        frame_length=porcupine.frame_length,
    )
    recorder.start()
    try:
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print('[%s] Detected wake word' % (str(datetime.now())))
                trigger_wakeword()  
    except KeyboardInterrupt:
        print('Stopping ...')
    finally:
        recorder.delete()
        porcupine.delete()
        
t = threading.Thread(target=recognize_wake_word_thread)
t.start()

# Main loop
while True:
    time.sleep(0.1)
    if trigger_voice:
        listen_and_respond()
        if time.time() - last_trigged_time > 30:
            trigger_voice = False
