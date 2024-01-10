# Package initialization
from pathlib import Path
import openai
from pynput import keyboard
import pyautogui
import os
import pytesseract
import time
import playsound


def on_press(key):
    if key == keyboard.Key.f10:
        os.remove('screenshot.png') if os.path.exists(
            'screenshot.png') else None
        print(time.time())
        screenshot = pyautogui.screenshot(region=(245, 855, 720, 200))
        # screenshot.save('screenshot.png')

        # ocr
        text = pytesseract.image_to_string(screenshot, timeout=1)
        # remove all newlines
        text = text.replace('\n', ' ')
        text = text[text.find(":")+1:]
        screenshot.close()

        os.remove('speech.mp3') if os.path.exists(
            'speech.mp3') else None
        # Create speech file
        speech_file_path = Path(__file__).parent / "speech.mp3"
        response = openai.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        response.stream_to_file(speech_file_path)
        playsound.playsound(speech_file_path, True)

    return True


def on_release(key):
    if key == keyboard.Key.f11:
        # Stop listener
        return False


with open("key.ini", "r") as f:
    openai.api_key = f.read()


# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

"""
text = ""



"""
