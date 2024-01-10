# Créé par François Costard
import openai
import pyttsx3

# Replace with your own API key
openai.api_key = "Ajoutez_votre_clé"

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )
    return response["choices"][0]["text"]

def speak(text):
    if text:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        print("Je suis crevé, désolé   ")

question = "Crache ta question... ? "
prompt = question
text = generate_text(prompt)
speak(text)

while True:
    question = input("Docteur Robot,à ton service... ? ")
    if question.lower() in ["quit", "exit", "Tatao!!!"]:
        break
    if question == "":
        question = "D'autres questions... ?"
    prompt = question
    text = generate_text(prompt)
    if text:
        print(text)
        speak(text)
        speak(question)
    else:
        print("Je suis sans voix...")
