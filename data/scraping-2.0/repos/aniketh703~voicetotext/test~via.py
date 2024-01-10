import openai
import speech_recognition as sr
openai.api_key = ""
r = sr.Recognizer()
mic = sr.Microphone()
languages = ['en-US', 'es-MX', 'fr-FR']
transcriptions = {}
with mic as source:
    r.adjust_for_ambient_noise(source)
    print("Say something!")
    while True:
        audio = r.listen(source)
        for lang in languages:
            try:
                text = r.recognize_google(audio, language=lang)
                transcriptions[lang] = text
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=f"Translate '{text}' to Spanish",
                    max_tokens=60,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )
                translation = response.choices[0].text
                print(f'Original ({lang}): {text}')
                print(f'Translation: {translation}')
            except sr.UnknownValueError:
                transcriptions[lang] = "Could not understand audio"
            except sr.RequestError as e:
                transcriptions[lang] = f"Could not request results from Google Speech Recognition service; {e}"