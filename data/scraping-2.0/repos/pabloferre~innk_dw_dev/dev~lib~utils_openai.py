import openai
from dotenv import load_dotenv
import os
import pandas as pd
import time
from openai.error import APIError

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') 
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
openai.api_key = OPENAI_API_KEY
openai.organization = OPENAI_ORG_ID



def classify_field(title:str, description:str):
    """ Function to classify a field into one of the following categories: 
    Problem, Solution or Idea name, Other. Usinf OpenAI's API.

    Args:
        title (str): field name
        description (str): field description

    Returns:
        response: final classification
    """
    
    message = f"I have a form field with the title: '{title}' \
            and the description: '{description}'. Classify the title into the categories: Problem, Solution, \
                Idea name, Other. Using the title and its description. \
                \n\
                Here are some examples for correct classification:\
                Example 1:\
                Title = '¿Cuál es la oportunidad o el problema sobre el que surge tu idea?'\
                Description = '¿Cuál es la oportunidad o el problema sobre el que surge tu idea? \
                    / In this section you should describe the specific problem you want to solve or \
                        opportunity you want to take advantage of.'\
                Category = 'Problem'\
                Example 2:\
                Title = '¿Qué es lo que busca tu idea?'\
                Description = 'Selecciona que acción apunta a lograr tu idea.'\
                Category = 'Solution'\
                Example 3:\
                Title: 'Tu propuesta de nombre'\
                Description: 'Resume en un par de palabras la acción que busca generar tu idea.\
                    Un buen titulo no debería tener más de 5 palabras.'\
                Category: 'Idea name'\
                Example 4:\
                Title: '¿Cuál es el tiempo estimado para desarrollarla?'\
                Description: 'Si no sabes exactamente puedes poner referencias o escribir \
                    Sin Información. Recuerda que mientras más antecedentes tengamos, podremos evaluar\
                        tu idea de mejor manera.'\
                Category: 'Other'\
                \n\
                Here are some examples of incorrect classification:\
                Example 1:\
                Title = '¿Cómo crees que podrían ser solucionados los problemas abordados para asegurar que tu equipo “vuele”?'\
                Description = 'En esta sección debes describir cuál es tu propuesta para el problema u oportunidad detectados.\
                    Sintetiza como funcionaria la solución que propones y en qué se diferencia de otras formas de abordar el \
                        problema.'\
                Category = 'Problem'\
                Example 2:\
                Title = '¿Qué es lo que busca tu idea?'\
                Description = 'Selecciona que acción apunta a lograr tu idea.'\
                Category =  'Solution'\
                Example 3:\
                Title = 'La necesidad'\
                Description = '¿A qué problema, necesidad, falla o quiebre del mercado se enfoca su proyecto?'\
                Category = 'Idea name'\
                Example 4:\
                Title = 'Contexto'\
                Description = 'Cuéntanos detalles sobre la necesidad, oportunidad o situación'\
                Category = 'Other'\
                \n\
                Give only the category as an answer."
    response = openai.ChatCompletion.create(
            model ="gpt-3.5-turbo",
            messages = [{"role": "system", 
                          "content": "You are a helpful assistant, that makes correct classifications."}, 
                         {"role": "user", "content": message}],
        max_tokens=100,
        temperature=0,
    )
    return response.choices[0].message['content']

