import openai
import time
import os
from dotenv import load_dotenv
from DB.obtener_peliculas import obtener_peliculas_BD
from dotenv import load_dotenv
import re

load_dotenv()

# Configura la API key
openai.api_key = os.getenv("API_KEY")

def IA_peliculas(genero, actor ,username,n):

    peliculas=obtener_peliculas_BD(username)

    peliculas_gustadas=[]

    n=10-n

    if n==1:
        promt="Recomiendame unicamente  solo el nombre de 1 pelicula distinta sin informacion antes ni despues,"
        
        if genero:
            promt += "que sea del genero,"
            for i in genero:
                promt = promt + i + ","

        if actor:
            promt += " donde participo el actor, "
            for i in actor:
                promt = promt + i + ","                 

        if len(peliculas)!=0:
            promt+="distintas pero sabiendo que me gustaron las peliculas, "   

            for i in peliculas:
                if peliculas[i]==1:
                    peliculas_gustadas.append(i)
                    promt = promt + i + ","


            promt+="y sabiendo que no me gustaron las peliculas,"
            for i in peliculas:
                if peliculas[i]==0:
                    promt = promt + i + ","                 
            
        promt+="y devuelvemelo en formato python de lista, en una sola linea, sin informacion extra, sin corchetes, separado por comas, sin comillas y sin puntos."

    else:
        # Genera una respuesta a partir de un prompt
        promt="Recomiendame solo los nombres de " + str(n) +" peliculas distintas sin informacion antes ni despues ,"
        
        if genero:
            promt += "que sean del genero, "
            for i in genero:
                promt = promt + i + ","

        if actor:
            promt += " donde participo el actor, "
            for i in actor:
                promt = promt + i + ","            

        if len(peliculas)!=0:

            promt+="distintas pero sabiendo que me gustaron las peliculas, "   


            for i in peliculas:
                if peliculas[i]==1:
                    peliculas_gustadas.append(i)
                    promt = promt + i + ","

            promt+="y sabiendo que no me gustaron las peliculas,"

            for i in peliculas:
                if peliculas[i]==0:
                    promt = promt + i + ","                 
            
        promt+="y devuelvemelo en formato python de lista, en una sola linea, sin informacion extra, sin corchetes, separado por comas, sin comillas y sin puntos."


    print(promt)

    start_time = time.time()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": promt },
        ],
    temperature=0.7)

    end_time = time.time()

    print("Tiempo de ejecucion: ", end_time - start_time , " segundos")

    # Extraer la respuesta generada por el modelo, la cual se encuentra en el primer elemento de la lista
    respuesta_generada = response['choices'][0]['message']['content']

    # Mostrar la respuesta
    respuesta_generada = respuesta_generada.split(", ")

    respuesta={
        "movies":respuesta_generada,
        "time": end_time - start_time
    }

    return respuesta

    






