import openai
import os
from elevenlabs import set_api_key, generate, save
import speech_recognition as sr
from SpeechRecorder import SpeechRecorder
from ProjectControls import TTS_API_KEY, StartPrompt
from playsound import playsound

filename = 'sound.wav'


def AnswerQuestion(location, recorder=SpeechRecorder(30, filename)):
    reply = ''
    openai.api_key = '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'
    messages = [
        {"role": "system", "content": "You are a intelligent assistant."}]

    sp = recorder
    SpeechToText = sr.Recognizer()

    sp.listen()

    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = SpeechToText.record(source)
        # recognize (convert from speech to text)
        speech = SpeechToText.recognize_google(audio_data)

        speech = StartPrompt + location + "\n\n" + speech
        # print(speech)

        messages.append({"role": "user", "content": speech})

        chatAnswer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)

        reply = chatAnswer.choices[0].message.content

    return reply


def initSpeech(text):
    set_api_key(TTS_API_KEY)

    audio = generate(text=text,
                     voice="Bella",
                     model="eleven_monolingual_v1",)
    os.remove("output.mp3")

    save(audio, 'output.mp3')
    return 'output.mp3'


def saySpeech(path, block=True):
    playsound(path, block=block)
# def SayText(text, speed=200):
#     SpeechEngine = pyttsx3.init()
#     # Set the voice property to one of the available voices
#     voices = SpeechEngine.getProperty('voices')

#     # Set the voice to the second voice in the list
#     SpeechEngine.setProperty('voice', voices[VoiceId].id)
#     SpeechEngine.setProperty('rate', speed)

#     SpeechEngine.say(text)
#     SpeechEngine.runAndWait()
