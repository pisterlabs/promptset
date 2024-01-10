import os
from dotenv import load_dotenv
import openai
import PyPDF2
import sys 

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Ahora puedes acceder a las variables de entorno usando la función `os.getenv()`
api_key = os.getenv("API_KEY")

# Configurar OpenAI
openai.api_key = api_key

ai_models = ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']

# gpt-3.5-turbo-1106
# gpt-4-1106-preview

ai_model = ai_models[0]

print("ai_model:",ai_model)


# Textos disponibles:

    # [x] DNU-2023-70-APN-PTE : Disposiciones que desregulan la economía. (20/12/2023) : DNU-2023-70-APN-PTE_dnu_a_texto.txt
    # [x] MEN-2023-7-APN-PTE :  Ley de Bases y Puntos de Partida para la Libertad de los Argentinos (27/12/2023) : MEN-2023-7-APN-PTE_Proyecto_de_Ley_que.txt

file = "MEN-2023-7-APN-PTE_Proyecto_de_Ley_que"



with open('data/'+file+".txt", 'r', encoding='utf-8') as f:
    texto = f.read()


# Inicializar la lista partes
partes = []


## FIXME: revisar cual es la mejor estrategia.
## Esta solución no utiliza chunks ni embeddings ni vectordbs. esto pasa el texto entero, en n partes, pregunta, y despues unifica respuestas. 
## esta solución DEMORA BASTANTE TIEMPO.
## Se puede preferir usar chunks, embeddings y vectordbs, para eso ir por la solución via langchain o esperar que se suba una versión via scripting .py que use chunks, embeddings y vectordbs.

## estrategia 1 :
# Iterar de 1 a 3 (3 partes del documento recortado por capitulos)

if ai_model == "gpt-4-1106-preview":
    rangopartes = range(1, 4)
else:
    file = file+"-gpt3"
    rangopartes = range(1, 7)


for i in rangopartes:
    print(i)
    # Leer el contenido de cada archivo
    print("file:",f'data/{file}-p{i}.txt')
    
    with open(f'data/{file}-p{i}.txt', 'r', encoding='utf-8') as f:
        partes.append(f.read())

# Imprimir la cantidad de partes o chunks del texto
print("-> Cant partes o chunks del texto:", len(partes))

# estrategia 2 :     
# Dividir el texto en partes usando "\nTÍTULO " como separador
# partes = texto.split("\nTÍTULO ")
# print("-> cant partes o chunks del texto:", len(partes))




def hacer_pregunta(partes, pregunta):
    respuestas = []
    for parte in partes:
        print("hacer_pregunta() parte..")

        respuesta = openai.ChatCompletion.create(
            model=ai_model,
            messages=[
                {
                    "role": "system",
                    "content": "Estás analizando decretos de necesidad de urgencia, proyectos de ley y leyes de la República Argentina. Comportate como el mejor analista económico-político de historia Argentina con un alto IQ. Tenes que ser muy serio y preciso. Si hay algo que no sabes o no tenes la respuesta bien clara, tenés que decir que no lo sabes. Tu mirada sobre el tema debe ser neutra. Siempre que puedas, contextualizá de forma completa tu respuesta. Debes responder siempre en español.",
                },
                {"role": "user", "content": "TÍTULO " + parte},
                {"role": "user", "content": pregunta},
            ],
        )
        respuestas.append(respuesta["choices"][0]["message"]["content"].strip())
    
    # Unir todas las respuestas en una sola cadena
    respuestas_unidas = " ".join(respuestas)
    
    # Hacer una pregunta final para obtener una respuesta coherente basada en las respuestas individuales
    respuesta_final = openai.ChatCompletion.create(
        model=ai_model,
        messages=[
            {
                "role": "system",
                "content": "Estás analizando decretos de necesidad de urgencia, proyectos de ley y leyes de la República Argentina. Comportate como el mejor analista económico-político de historia Argentina con un alto IQ. Tenes que ser muy serio y preciso. Si hay algo que no sabes o no tenes la respuesta bien clara, tenés que decir que no lo sabes. Tu mirada sobre el tema debe ser neutra. Siempre que puedas, contextualizá de forma completa tu respuesta. Debes responder siempre en español.",
            },
            {"role": "user", "content": respuestas_unidas},
            {"role": "user", "content": "Dame una respuesta coherente en base a estas respuestas."},
        ],
    )
    return respuesta_final["choices"][0]["message"]["content"].strip()

pregunta = "¿Cuál es el tema principal del documento?"  # Reemplaza esto con tu pregunta
respuesta = hacer_pregunta(partes, pregunta)
print("Respuesta:", respuesta)
