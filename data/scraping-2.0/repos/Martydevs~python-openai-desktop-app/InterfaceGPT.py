import os
import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


def pregunta(datos: str):
    res = ""
    modelo = "text-davinci-002"
    respuesta = openai.Completion.create(
        engine=modelo, prompt=datos, n=1, max_tokens=150, temperature=0.5
    )
    return respuesta.choices[0].text.strip()
