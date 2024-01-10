import openai
import config
import pandas as pd
import numpy as np
import time 
openai.api_key = config.api_key

## leer archivo libro 1.xlsx
df = pd.read_excel("Libro1.xlsx")
df = df.dropna()

## datos del 1 al 170 

df = df.drop(columns=['Column1']) ## eliminar column1 text-mining no lo necesito 
df = df[1:170]
lista_comentarios = df["Column2"].tolist()
# # comentarios = lista_comentarios[1:25]

comentarios = lista_comentarios[:] # copiar lista

# # print(comentarios)


# comentarios=[
#     "La Doctora zally excelente profesional, muy buena su atención se nota que su dedicación es real y completa. 8",
#     "El médico procedió correctamente.Las Enfermeras y teens muy gentiles. 9",
#     "El tiempo que se tomó el médico que me atendió para explicarme todo con detalle. 10",
#     "atencion mala. 2",
#     "nada. 1",
#     "no sé 10"
# ]

## contexto 
pregunta =[{"role":"system","content":"Eres un asistente que lee los comentarios y responde si un comentario es positivo o negativo o sin contexto, si es positivo resesponde 'positivo' si es negativo responde 'negativo' si no tiene contexto analiza el numero que se muestra al lado del comentario, el mismo va de 0 a 10, donde 0 es 'para nada recomendable' y 10 es 'muy recomendable', y utiliza ese numero para responder si es positivo o negativo. Solo responde con una palabra, sin dar explicaciones. No respondas algo diferente a 'positivo', 'negativo' o 'sin contexto'."}]

# positivos = 0
# negativos = 0
# sin_contexto = 0

for i in range(0, len(comentarios)):
    print(comentarios[i])
    try:
        pregunta.append({"role":"user","content":comentarios[i]})
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                messages=pregunta, 
                                                )
        pregunta.append({"role":"assistant","content":response.choices[0].message.content})

        # print(response.choices[0].message.content + "\n")
        
        ## guardar cada prediccion
        df.loc[i, 'chatgpt'] = response.choices[0].message.content

        ## time sleep 2 segundo por cada respuesta porque sino me bloquea la api
        time.sleep(2)
    except:
        pass
