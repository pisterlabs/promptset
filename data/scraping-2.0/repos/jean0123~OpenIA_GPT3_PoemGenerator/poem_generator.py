import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the data from User
autor = input(
    "Please enter the name of the person who wants to generate a poem: ")
longitud = int(
    input("Please enter the maximum length of the poem [letters]: "))
listado_palabras = input(
    "Please enter a list of keywords to use in the poem, separated by commas: ")

# Organize the data
separar_palabras = listado_palabras.replace(" ", "").split(",")
concatenar_palabras = "\n".join(separar_palabras)
generar_prompt = f"Write a poem with the following words:\n\n{concatenar_palabras}"

# Create the poem
print("Generating poem...\n")
response = openai.Completion.create(
    engine="davinci-instruct-beta-v3",
    prompt=generar_prompt,
    temperature=1.0,
    max_tokens=longitud,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0
)
print("Generated poem!")
print(response.choices[0].text)
print("\nBy: " + autor)
