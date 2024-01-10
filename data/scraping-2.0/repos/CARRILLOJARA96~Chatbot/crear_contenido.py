import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def crear_contenido(tema, tokens, temperatura, modelo="gpt-3.5-turbo-instruct"):
    prompt = f"Por favor escribe un articulo corto sobre el tema: ´{tema}\n\n"
    respuesta = openai.Completion.create(
        engine=modelo,
        prompt=prompt,
        n=1,
        max_tokens=tokens,
        temperature=temperatura
    )
    return respuesta.choices[0].text.strip()

def resumir_texto(texto, tokens, temperatura, modelo="gpt-3.5-turbo-instruct"):
    prompt = f"Por favor resume el siguiente texto en español: {texto}\n\n"
    respuesta = openai.Completion.create(
        engine=modelo,
        prompt=prompt,
        n=1,
        max_tokens=tokens,
        temperature=temperatura
    )
    return respuesta.choices[0].text.strip()

##tema= input("Elige un temapara tu artículo: ")
##tokens = int(input("Cuántos tokens máximo tendrá tu artículo: "))
##temperatura = int(input("Del 1 al 10, que tan creativo quieres que sea tu artículo: "))
##articulo_creado = crear_contenido(tema,tokens,temperatura)
##print(articulo_creado)

original= input("Pega aqui el artículo que quieras resumir: ")
tokens = int(input("Cuántos tokens máximo tendrá tu resumen: "))
temperatura = int(input("Del 1 al 10, que tan creativo quieres que sea tu resumen: "))
resumen = resumir_texto(original,tokens,temperatura)
print(resumen)