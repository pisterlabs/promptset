import pyttsx3
import speech_recognition as sr
import openai
import env

# Key
openai.api_key = env.OPEN_AI_KEY

# Start up our speech recognition engine
engine = pyttsx3.init()


def reply(word):
    engine.setProperty('rate', 135)
    engine.setProperty('volume', 0.8)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    engine.say(str(word))
    engine.runAndWait()
    engine.stop()


activated = False
while True:

    # Actual speech recognition
    rec = sr.Recognizer()
    rec.dynamic_energy_threshold = True
    print(activated)
    with sr.Microphone() as source:
        audio = rec.record(source, duration=4)
        try:
            startKey = rec.recognize_google(audio)
        except sr.UnknownValueError:
            pass
        print(startKey)
        if 'chat' in startKey:
            activated = True
        if activated:

            pyttsx3.speak('What is your query?')
            audio = rec.listen(source)

            try:
                audio = rec.listen(source, timeout=5)
            except sr.WaitTimeoutError:
                pass

            try:
                text = rec.recognize_google(audio)
                print(text)
            except sr.UnknownValueError:
                pyttsx3.speak('I could not hear your answer clearly.')
            except NameError:
                break
            if text:
                pyttsx3.speak('I am querying your answer. I will be finished shortly.')
                discussion = openai.Completion.create(
                    prompt=text,
                    engine='text-davinci-002',
                    max_tokens=1000,
                )

            answer = discussion.choices[0].text

            if answer:
                print(answer)
                pyttsx3.speak(answer)
            activated = False
