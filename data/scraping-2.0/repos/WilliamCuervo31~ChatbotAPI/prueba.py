from openai import OpenAI
import speech_recognition as sr
import pyttsx3

client = OpenAI(
    api_key = "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",
)

engine = pyttsx3.init()

recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
        
    print("Ahora puedes hacer tu consulta: ")
    audio = recognizer.listen(source)

    text = recognizer.recognize_google(audio, language = "ES")

    consulta = "Tu consulta fue: ", text
    print(f"Tu consulta fue: {text}")
    engine.say(consulta)
    engine.runAndWait()

    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": text,
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=2000,
    )

    respuesta_chat = chat_completion.choices[0].message.content

    print(respuesta_chat)
    engine.say(respuesta_chat)
    engine.runAndWait()
        


