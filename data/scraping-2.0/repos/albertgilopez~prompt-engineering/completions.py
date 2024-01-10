# Importar la librería dotenv para cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import json

client = OpenAI() # Crear una instancia del cliente de OpenAI

def complete(text):
    
    completion = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"""
        Eres un chatbot asistente virtual, que ayuda a los usuarios a aprender inglés.
        Assistant: Hola soy tu asistente, preguntame lo que quieras.
        User: Hola, como se dice hola en inglés?
        Assistant: se dice 'Hi'
        User: {text}
        Assistant:
        """,
        max_tokens=100,
        temperature=0.5,
        frequency_penalty=1,
        user="albert",
    )

    return completion


response = complete("Cómo se dice 'Donde esta el baño?'")

# Convertir response.choices a una lista de diccionarios
choices = [{"text": choice.text, "finish_reason": choice.finish_reason, "index": choice.index} for choice in response.choices]

# Convertir response.usage en un diccionario
response_usage = {
    "completion_tokens": response.usage.completion_tokens,
    "prompt_tokens": response.usage.prompt_tokens,
    "total_tokens": response.usage.total_tokens
}

# Extraer los datos relevantes de la respuesta
response_data = {
    "id": response.id,
    "created": response.created,
    "model": response.model,
    "choices": choices,
    "usage": response_usage
}

print(response)
print(json.dumps(response_data, indent=4, ensure_ascii=False))
