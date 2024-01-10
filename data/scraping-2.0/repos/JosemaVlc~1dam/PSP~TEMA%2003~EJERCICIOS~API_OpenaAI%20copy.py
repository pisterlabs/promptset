# import OpenIA
import os
import requests
from openai import OpenAI

# import weather
from urllib import request
import json

# importa la biblioteca dotenv
from dotenv import load_dotenv

def openai(municipio, temperatura, velocidad_viento, cielos):
    """Envia prompt con municipio, temperatura, y como esta el cielo a chatGPT 
    y recibe la respuesta con que actividades puedes realizar"""
    
    # carga las variables de entorno desde el archivo .env
    load_dotenv()

    contexto = "Eres el reputado presentador del tiempo en los informativo llamado Gepeto Tornado y debes recomendarnos en un parrafo que hacer segun el tiempo y el municipio que te proporcione"
    texto = f"¿Qué acividades puedo hacer en {municipio} con {temperatura} grados, {cielos} y velocidad del viento de {velocidad_viento} km/h? Tambien puedes ofrecer que tipo de ropa debemos llevar segun la temperatura"

    client = OpenAI()
    

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": contexto
            },        
            {
                "role": "user",
                "content": texto
            }],
        stream=True,        
        # Temperature del 0 al 1 siendo 0 un tono mas serio y 1 el tono mas creativo y distendido.
        temperature=1
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

def openweathermap():
    """Utiliza el Api de OpenWeatherMap para optener el tiempo de un municipio en concreto"""

    municipio = "Barcelona"
    api_key = "83e5978c218757a762484859178514af"
    unidad = "metric"

    url = "https://api.openweathermap.org/data/2.5/weather?q=%s&units=%s&appid=%s" % (municipio, unidad, api_key)

    #print (url)

    response = request.urlopen(url)
    http_body = response.readline().decode('utf-8')

    #print (http_body)

    #codificar la respuesta a json
    data = json.loads(http_body)
    #print (data)

    main = data ["main"]
    temperatura = main["temp"]
    
    tiempo = data ["weather"][0]
    cielos = tiempo["description"]
    
    viento = data ["wind"]
    velocidad_viento = viento["speed"]
    
    cod = data["cod"]
    
    #print("\nLa temperatura de "+ municipio + " es: "+ str(temperatura) + " con cielos: " + cielos)

    return municipio, temperatura, velocidad_viento, cielos, cod

if __name__ == "__main__":
    
    municipio, temperatura, velocidad_viento, cielos, cod = openweathermap()
    
    if cod == 200:
        openai(municipio, temperatura, velocidad_viento, cielos)