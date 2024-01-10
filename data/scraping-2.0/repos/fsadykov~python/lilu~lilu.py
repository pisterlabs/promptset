class Lilu():
    def __init__(self,):
        pass



import openai, os
import speech_recognition as sr
import subprocess

# Set the API key
openai.api_key = os.getenv('OPEN_AI_TOKEN')
r = sr.Recognizer()

# Use the GPT-3 language model to generate text
prompt = "How you would asnswer to question 'What is the difference between pod and deployment'"
model = "text-davinci-002"


while True:
    # Start listening for voice input
    # with sr.Microphone() as source:
    #     r.adjust_for_ambient_noise(source)
    #     print("Listening...")
    #     audio = r.listen(source)

    prompt = input('Ask me question: ')

    # Transcribe the audio to text
    # prompt = r.recognize_google(audio)


    completions = openai.Completion.create(engine=model, prompt=prompt, max_tokens=1024, n=1,stop=None,temperature=0.5)
    message = completions.choices[0].text
    print(message)
    # Set up the speech recognition module


    # Call the say command
    # subprocess.run(["say", message])



