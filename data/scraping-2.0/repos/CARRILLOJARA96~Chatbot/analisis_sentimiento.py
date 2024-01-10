import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def analizar_sentimientos(texto):
    prompt = f"Analiza el sentimiento predominante en el siguienye texto: '{texto}'. El sentimiento es :"
    respuesta=openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        n=1,
        max_tokens=100,
        temperature=0.5
    )
    return respuesta.choices[0].text.strip()

texto_para_analizar=input("Ingresa un texto: ")
sentimiento=analizar_sentimientos(texto_para_analizar)
print(sentimiento)