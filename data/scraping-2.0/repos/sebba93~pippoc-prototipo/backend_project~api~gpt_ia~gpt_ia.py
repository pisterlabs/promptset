import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
#from openai import OpenAI
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

def consulta_gpt(gpt, query):
    try:
        response = openai.ChatCompletion.create(
            messages=[
                {'role': 'system', 'content': 'Respondes preguntas acerca de la seguridad laboral.'},
                {'role': 'user', 'content': query},
            ],
            model=gpt,
            temperature=0,
        )
        return response['choices'][0]['message']
    except:
        return {'content' : 'Error: Hay un problema con la conexión con OpenAI, verifique sus credenciales e intente nuevamente'}