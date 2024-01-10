import openai
import os
import firebase_admin 
from firebase_admin import db, credentials
from tabulate import tabulate
import pandas as pd
#funtion to sleep
import time

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://iot-robotarm-default-rtdb.firebaseio.com/"})
ref = db.reference('/IAROB')

i = 1
while i != 0:
    # Inicializa la base de datos de Firebase


    # Configura tu clave de API
    openai.api_key = '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'


    prods = (f"Explicame y dime cuales son los objetos que se encontraron en el video: {ref.get()}basado en sus cordenadas y porcentaje de cercania a la cual se encuentran que deberia hacer? ")
    # Genera texto usando GPT-3
    response = openai.Completion.create(
       engine="text-davinci-003",
      prompt=prods,
      temperature=0.5,
      max_tokens=1000
    )
     #Imprime la respuesta generada por GPT-3
    print(response.choices[0].text)
    objetos = db.reference('/IAROB/objetos detectados')
    objetos1 = []
    objetos1.append(objetos.get())
   
    cordenadas = db.reference('/IAROB/cordenados')
    porcentajes = db.reference('/IAROB/porcentaje_de_cercania')
    cordenadas1 = []
    cordenadas1.append(cordenadas.get())
    porcentajes1 = []
    porcentajes1.append(porcentajes.get())


    # Datos
    datos = {
        "Objeto": objetos1[0],
        "Coordenadas": cordenadas1[0],
        "Cercanía": porcentajes1[0]
    }

    # Crear un DataFrame a partir de los datos
    df = pd.DataFrame(datos)
    df["Cercanía"] = df["Cercanía"].map(lambda x: f"{x}%")


    # Imprimir el DataFrame
    print(tabulate(df, headers="keys", tablefmt="grid"))
    #sleep 10 seconds
    time.sleep(20)
