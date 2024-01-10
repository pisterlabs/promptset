import openai
import requests


def conexion(prompt):
    """Función que se encarga de conectar con la API de la IA seleccionada y regresa la respuesta de la IA."""

    openai.api_key = "xxxxxxxx"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return completion["choices"][0]["message"]["content"]
