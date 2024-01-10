"""
Para muchos casos de usos solo enviar un texto para ser procesado 
no es suficiente, por lo que se requiere de una secuencia de procesos 
que se ejecuten en orden. Para esto se puede utilizar las cadenas 
SimpleSequentialChain o SequentialChain que permiten encadenar varios 
procesos de manera secuencial.
"""

import os
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains import SimpleSequentialChain

API = os.environ['OPENAI_API_KEY']
llm = OpenAI(openai_api_key=API)

# PRIMER CHAIN
prompt = '''Eres un asistente virtual experto en {tema} y respondes 
            con una lista de 3 conceptos clave sobre el mismo
            Solo enumeras los tres conceptos'''
template = PromptTemplate.from_template(prompt)

# # Armamos una cadena la cual va a recibir la salida de la cadena cadena_LLM y lo procesa para generar otro texto
cadena_lista = LLMChain(llm=llm, prompt=template, output_key="lista_conceptos")

# # SEGUNDO CHAIN
prompt = '''Eres un asistente virtual que recibe una lista de conceptos 
            de un area de conocimiento y 
            debe devolver cual de esos conceptos es mejor aprender primero.
            Los conceptos son: {lista_conceptos}'''
template = PromptTemplate.from_template(prompt)
cadena_inicio = LLMChain(llm=llm, prompt=template, output_key="donde_iniciar")

# # EJECUTAMOS CHAIN
cadena_simple = SimpleSequentialChain(chains=[cadena_lista, cadena_inicio], verbose=True)
cadena_simple.run("Inteligencia artificial")
