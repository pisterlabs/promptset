# Import libraries 
from decouple import config 
from openai import OpenAI
from pathlib import Path
import playsound
import time

# Enviroment variables
OPENAI_API_KEY = config('OPENAI_API_KEY')

# Configs
speech_file_path = Path(__file__).parent / "lesson.mp3"
music = Path(__file__).parent / "katyusha.mp3"
client = OpenAI(api_key=OPENAI_API_KEY)

# Functions
def speech_to_text():
    with open ("p8/katyusha.mp3", "rb") as f:
        transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=f, 
        response_format="text"
        )
    return transcription

def text_to_speech(text):
    audio = client.audio.speech.create(
        model = "tts-1",
        voice = "alloy",
        input = text
    )
    audio.stream_to_file(speech_file_path)
    return speech_file_path

def save(text):
    with open("p8/lesson.txt", "w", encoding="utf-16") as f:
        f.write(text)
    return f

def translate(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system","content": "You are an expert russian translator"},
            {"role": "user", "content": text}
        ])
    translation = response.choices[0].message.content
    return translation

def play():
    playsound.playsound(speech_file_path)

# Main section
def main():
    print("Starting translation service...")
    text = speech_to_text()
    print(f'Oh my god, are you russian?. Katyusha is my favorite song ever!\r\n{text}')
    time.sleep(2)
    save(text)
    translation = translate(text)
    print(f'Jesus, thath thanslation looks kinda weird, let me read it for you:\r\n{translation}')
    text_to_speech(translation)
    time.sleep(2)
    print("\nNow you can listen to the translation")
    play()
    print("\nBy the way, lets listen the real one")
    playsound.playsound(music)

if __name__ == "__main__":
    main()