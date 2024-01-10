import pyttsx3 as tts
import speech_recognition as sr
import openai
import asyncio

async def get_response(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",  # use a smaller and faster model
        prompt=prompt,
        max_tokens=50,  # reduce the number of tokens generated
        temperature=0.5,
    )
    return completions.choices[0].text

async def listen_and_respond():
    openai.api_key = "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"
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300

    while True:
        try:
            print("Your turn to speak")

            with sr.Microphone() as source:
                print("please speak")
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                print("listening")
                audio = recognizer.listen(source)
                print("got it")

            recognizer_input = recognizer.recognize_google(audio).lower()
            print(recognizer_input)

            response = await get_response(recognizer_input)

            engine = tts.init()
            voices = engine.getProperty('voices')
            engine.setProperty('rate', 155)
            engine.setProperty('voice', voices[1].id)
            engine.say(response)
            engine.runAndWait()

        except Exception as e:
            print(e)
            engine = tts.init()
            voices = engine.getProperty('voices')
            engine.setProperty('rate', 155)
            engine.setProperty('voice', voices[1].id)
            engine.say("Please come back later")
            engine.runAndWait()
            print("Thankyou")
            break

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(listen_and_respond())
