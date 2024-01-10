import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

modelo = "gpt-3.5-turbo-instruct"
prompt = "quien fue el inka atahualpa del perú"


respuesta = openai.Completion.create(
    engine=modelo,
    prompt=prompt,
    n=1,
    temperature=1,  ##nivel de libertad
    max_tokens=100   
)
texto_generado = respuesta.choices[0].text.strip()
print(texto_generado)