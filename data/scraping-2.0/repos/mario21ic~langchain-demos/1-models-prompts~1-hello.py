"""
Usamos modelos para completar texto
"""

import os

from langchain.llms import LlamaCpp, OpenAI

OPENAI_API_KEY=os.environ['API_KEY']
MODEL="text-davinci-003"

# Demo1
#llm_openai = OpenAI(model_name=MODEL, openai_api_key=OPENAI_API_KEY, temperature=1.5)
llm_openai = OpenAI(model_name=MODEL, openai_api_key=OPENAI_API_KEY)
respuesta_openai = llm_openai("Hola, como estas?")
print(respuesta_openai)

