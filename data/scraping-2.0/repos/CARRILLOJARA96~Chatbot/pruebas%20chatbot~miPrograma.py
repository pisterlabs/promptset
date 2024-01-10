import os
import openai
import spacy
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

modelo = "gpt-3.5-turbo-instruct"
prompt = "hablame acerca de la ciudad del Cusco de Perú"


respuesta = openai.Completion.create(
    engine=modelo,
    prompt=prompt,
    n=1,
    temperature=1,  ##nivel de libertad
    max_tokens=100   ##numero de palabras que permite generar
)
texto_generado = respuesta.choices[0].text.strip()
print(texto_generado)

print("**************")

##usando spacy para analisis en lenguaje natural
modelo_spacy = spacy.load("es_core_news_md")
analisis = modelo_spacy(texto_generado)

ubicacion = None

for ent in analisis.ents:
    if ent.label_ =="LOC":
        ubicacion = ent
        break

if ubicacion:
    prompt2= f"Dime más acerca de {ubicacion}"
    respuesta2 = openai.Completion.create(
        engine = modelo,
        prompt = prompt2,
        n=1,
        max_tokens=100
    )
    print(respuesta2.choices[0].text.strip())
