import openai
import pyttsx3
import speech_recognition as sr
import time



open.ai_key = ""

engine = pyttsx3.init()
wakeWord = "Genuis"
assistantName = "Jarvis"
startupPhrase = "I am at your service, sir"

def transcribeAudioToTest (filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print("Skipping unkown error")


def generateResponse(prompt):
    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = prompt,
        max_tokens = 4000,
        n = 1,
        stop = None,
        temperature = 0.5,
    )
    return response["choices"][0]["text"]


def speakText(text):
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        print("Say " + wakeWord + " to wake up " + assistantName)
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == wakeWord:
                    filename = "input.wav"
                    print(startupPhrase)
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit = None, timeout = None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                    text = transcribeAudioToTest(filename)
                    if text:
                        response = generateResponse(text)
                        print(response)
                        speakText(response)
            except Exception as e:
                print("An error occured: {}".format(e))


if __name__ == "__main__":
    main()
