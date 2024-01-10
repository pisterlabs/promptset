import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def clasificar_texto(texto):
    categorias = [
        "arte",
        "ciencia",
        "deportes",
        "economia",
        "educación",
        "entretenimiento",
        "medio ambiente",
        "politica",
        "salud",
        "tecnología"
    ]
    prompt = f"Por favor clasifica el siguiente texto '{texto}' en una de estas categorias: {','.join(categorias)}. La categoria es: "
    respuesta = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        n=1,
        max_tokens=100,
        temperature=0.5
    )
    return respuesta.choices[0].text.strip()

texto_para_clasificar = input("Ingrese un texto: ")
clasificacion = clasificar_texto(texto_para_clasificar)
print("El texto se clasifica como: ")
print(clasificacion)