import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# print (api_key)

openai.api_key = api_key

# models = openai.Model.list()
# print(models)

model = 'text-davinci-002'
# prompt = "¿Cuál es la capital de Francia?"
# prompt = "Inventa un poema de 10 palabras sobre python"
# prompt = "De que trata la pelicula El Padrino 2"
prompt = "Elije un buen nombre para un elefante"

response = openai.Completion.create(
  engine=model,
  prompt=prompt,
  n=1,
  temperature=1,
  max_tokens=100,
)

# text = response['choices'][0]['text'].strip()

# print(response)
# print(text)
for idx, option in enumerate(response['choices']):
  print(f"Option {idx + 1}: {option['text']}\n")