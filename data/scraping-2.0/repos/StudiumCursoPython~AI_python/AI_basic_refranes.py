""" 
Curso Python empresa de 'Lenguaje de Programación Python'

Autor: José Antonio Calvo López

Fecha: Noviembre 2023

"""

import openai

def completar_frase_gpt3(prompt, api_key):
    openai.api_key = api_key

    response = openai.Completion.create(
        # Se puede elegir y usar un modelo específico de GPT-3
        engine="text-davinci-003",
        
        #promt es una entrada de texto inicial que se proporciona al modelo para iniciar la generación de texto.
        #El prompt es esencialmente la parte del texto que se presenta como contexto o inicio de una solicitud al modelo
        #y a partir de ese punto, el modelo genera texto de manera coherente y relevante.
        
        prompt=prompt,
        # Número máxinmo de palabras generadas
        max_tokens=50  
    )

    return response.choices[0].text.strip()

# Clave API de OpenAI
api_key = "API en el mi Drive"

# El comienzo del refrán o frase que quieres completar

prompt = input("Dime el comienzo del refran: ")

# Obtener la frase completada por GPT-3
completed_phrase = completar_frase_gpt3(prompt, api_key)
print("Frase completada:", completed_phrase)
