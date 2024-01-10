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

template_basico = """Eres un asistente virtual culinario que responde a preguntas de manera muy breve.
Pregunta: Cuales son los ingredientes para preparar {platillo} al estilo {estilo}
Respuesta:"""

# Construimos el template, especificandole cuales son las variables de entrada y cual es el texto
prompt_temp = PromptTemplate(input_variables=["platillo", "estilo"], template=template_basico)

# Aqui podemos ver como se reemplaza la variable platillo
#prompt_value = prompt_temp.format(platillo="ceviche", estilo="peruano")
#prompt_value = prompt_temp.format(platillo="ceviche", estilo="ecuatoriano")
prompt_value = prompt_temp.format(platillo="mondongo", estilo="peruano")
#prompt_value = prompt_temp.format(platillo="mondongo", estilo="colombiano")
#prompt_value = prompt_temp.format(platillo="juane", estilo="norteño")
print("prompt:", prompt_value)

# Se puede revisar el nro de tokens de un prompt en especifico
print("tokens:", llm_openai.get_num_tokens(prompt_value))

respuesta_openai = llm_openai(prompt_value)
print("rpta:", respuesta_openai)
