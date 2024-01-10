# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
import time
from utils.recording import record_audio
from utils.gtts_synthing import synthing
from dotenv import load_dotenv

character_dict = {
    "honeyBee": "speak in a sweet and friendly tone, like a cute honey bee",
    "currywurst": "speak in a humorous, loud and cheecky tone, like a Berlin currywurst",
    "treasureChest": "speak in a mysterious and dreamy way, like a treasure chest"
}

def speak(text):
    #voice = "-v 'Eddy (Deutsch (Deutschland))'"
    voice = ""
    print("\n " + text)
    os.system("say -r180 "+voice + " " + text)

def transcribe_audio(filename):
    audio_file = open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print("Ich habe folgendes verstanden:")
    print(transcript.text)
    return transcript.text

def query_chatgpt(prompt):
    messages = []
    messages.append(
        {"role": "system", "content": character_dict["honeyBee"]})

    message = prompt
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply

def play_audio():
    os.system("afplay " + "output_gtts.mp3")

def main():
    os.system("clear")
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    soundfile_name = "input.wav"

    print("Hallo ich bin der Awesomebot vom CityLAB Berlin!")

    while True:
        record_audio()
        start_time = time.time()
        prompt = transcribe_audio(soundfile_name)
        end_time = time.time()
        print("time of whisper:", end_time - start_time)
        #speak(prompt)
        start_time2 = time.time()
        reply = query_chatgpt(prompt)
        end_time2 = time.time()
        print("time of chatgpt:", end_time2 - start_time2)
        #speak(reply)
        #request_speech(reply)
        synthing(reply)
        play_audio()

if __name__ == '__main__':
    main()
