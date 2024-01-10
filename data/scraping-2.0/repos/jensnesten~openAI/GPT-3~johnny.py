import speech_recognition as sr 
import openai
import time

openai.api_key = "OPENAI_API_KEY"

def transcribe(audio):
    recognizer = sr.Recognizer()
    text = recognizer.recognize_google(audio)
    return text 

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text
    return message

while True:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = recognizer.listen(source)
        
    prompt = transcribe(audio)
    response = generate_response(prompt)
    for char in response:
        print(char, end="", flush=True)
        time.sleep(0.025)
    print("")




