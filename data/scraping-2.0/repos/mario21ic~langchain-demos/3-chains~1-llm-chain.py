"""
LLMChain es una de las cadenas que más usaras. Lo que hace es unir dos elementos
para que puedas interactuar con las LLMs de manera mas sencilla.

Una un modelo LLM (Puede ser LLama, OpenAI, Cohere etc.) y los templates
de prompts vistos en el cuaderno intro.ipynb.
"""

import os
from langchain import LLMChain, OpenAI, PromptTemplate

prompt = '''Eres un asistente virtual experto en {tema} y respondes 
            con una lista de 3 conceptos clave sobre el mismo
            Solo enumeras los tres conceptos'''
template = PromptTemplate.from_template(prompt)


API = os.environ['OPENAI_API_KEY']

llm = OpenAI(openai_api_key=API)
cadena_LLM = LLMChain(llm=llm, prompt=template)

print(cadena_LLM.predict(tema="inteligencia artificial"))
print(cadena_LLM.predict(tema="computacion cuantica"))
