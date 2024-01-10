import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

modelo = "gpt-3.5-turbo-instruct"
prompt = "Elije un buen nombre para un perro"


respuesta = openai.Completion.create(
    engine=modelo,
    prompt=prompt,
    n=1,
    temperature=1,  ##nivel de libertad
    max_tokens=50   ##maxima cantidad de palabras
)
for idx, opcion in enumerate(respuesta.choices):
    texto_generado = opcion.text.strip()
    print(f"Respuesta {idx + 1}: {texto_generado}\n")