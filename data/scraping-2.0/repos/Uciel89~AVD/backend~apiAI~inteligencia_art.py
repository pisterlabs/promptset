import openai
import os
from googletrans import Translator
import json

"""
INTENTAR CON VOZ
IF VOZ O TECLADO

SE PUEDE ENVIAR EN JSON TAMBIÉN
"""

# translator = Translator()

def gpt3(stext):
    openai.api_key= 'ingresar key'
    response = openai.Completion.create(
    engine="davinci",
    prompt=stext,
    temperature=0.0,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    )
    
    content = response.choices[0].text.split('.')
    print(content[0])
    arreglo = content[0]
    response = arreglo.strip('\n\n')
    content[0] = response
    
    return content