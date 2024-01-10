import os
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import openai
from dotenv import dotenv_values

# Controll parameters
config = dotenv_values(".env")
power = True
mute = False
listening = True

def speak(text):
    print("GPT: " + text)

    if not mute:

        # using gTTS and playsound
        file_name = "voice.mp3"
        tts = gTTS(text=text, lang="en", slow=False)
        if os.path.exists(file_name):
            os.remove(file_name)
        tts.save(file_name)
        playsound(file_name)


def listen(recogniszer, audio):
    if listening:
        try:
            text = recogniszer.recognize_google(audio, language="en")
            print("(User voice): " + text)
            answer(text)
        except sr.UnknownValueError:
            pass
            # waiting
            # print("Voice Recognition Failed")
        except sr.RequestError as e:
            print(f"Requset failed: {0}".format(e))


def answer(question):
    if question == "exit":
        stop_listening(wait_for_stop=False)  # stop listening
        print("GPT ended")
        return "POWER_OFF"
    elif question == "mute":
        print("GPT muted")
        return "MUTE"
    elif question == "unmute":
        print("GPT unmuted")
        return "UNMUTE"
    elif question == "activate listening":
        print("GPT listening activated")
        return "ACTIVATE_LISTENING"
    elif question == "stop listening":
        print("GPT listening disabled")
        return "STOP_LISTENING"
    else:
        openai.organization = config["ORGANIZATION_KEY"]
        openai.api_key = config["API_KEY"]
        openai.Model.list()

        answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
        )["choices"][0]["message"]["content"]

        speak(answer)


# Introduction, instructions and help
print("[GPT SPEAKER]")
print(" COMMAND: ")
print("\texit: end program\n\tmute: mute speaker\n\tunmute: unmute speaker\n\tstop listening: disable listening\n\tactivate listening: start voice recognition\n")

r = sr.Recognizer()
m = sr.Microphone()

speak("Hi, how can I help you?")  # need to install PyAudio

stop_listening = r.listen_in_background(m, listen)
# stop_listening(wait_for_stop = False) # stop listening

while power:
    # Can make command for mute, IOT functions
    if listening:
        welcome_text = "Type command here (or voice): "
    else:
        welcome_text = "Type command here: "

    type_command = input(welcome_text)
    command_code = answer(type_command)

    if command_code == "POWER_OFF":
        power = False
    elif command_code == "MUTE":
        mute = True
    elif command_code == "UNMUTE":
        mute = False
    elif command_code == "STOP_LISTENING":
        listening = False
    elif command_code == "ACTIVATE_LISTENING":
        listening = True
