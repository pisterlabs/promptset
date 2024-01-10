"""
Los modelos que utilizamos eran para completar texto, sin embargo podemos tambien usar modelos especificos para chat como ChatGPT
"""

import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import PromptTemplate
from langchain.llms import OpenAI

OPENAI_API_KEY = os.environ['API_KEY']
MODEL="text-davinci-003"
llm_openai = OpenAI(model_name=MODEL, openai_api_key=OPENAI_API_KEY)

from langchain import FewShotPromptTemplate

# Lista de ejemplos
ejemplos = [
    {"pregunta": "¿Cuál es el ingrediente principal de la pizza?", "respuesta": "La masa y salsa de tomate"},
    {"pregunta": "¿Cuál es el ingrediente principal de la hamburguesa?", "respuesta": "La carne y el pan"},
    {"pregunta": "¿Cuál es el ingrediente principal del burrito?", "respuesta": "La tortilla y la carne"}
]

# Ahora armamos un template para el modelo
prompt_temp_ejemplos = PromptTemplate(input_variables=["pregunta", "respuesta"], 
                                     template = "Pregunta: {pregunta}\nRespuesta: {respuesta}")

prompt_ejemplos = FewShotPromptTemplate(example_prompt=prompt_temp_ejemplos, 
                                       examples=ejemplos, 
                                       prefix = "Eres un asistenet virtual culinario que responde preguntas de manera muy breve",
                                       suffix = "Pregunta: {pregunta}\nRespuesta:", 
                                        input_variables=["pregunta"]) 

prompt_value = prompt_ejemplos.format(pregunta="¿Cuál es el ingrediente principal del coctel de camaron?")
print(prompt_value)

respuesta = llm_openai(prompt_value)
# respuesta = llm_openai("¿Cuál es el ingrediente principal del coctel de camaron?")
print(respuesta)


