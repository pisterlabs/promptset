"""
Los modelos que utilizamos eran para completar texto, sin embargo podemos tambien usar modelos especificos para chat como ChatGPT
"""

import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

OPENAI_API_KEY = os.environ['API_KEY']

chatgpt = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
respuesta = chatgpt( [ HumanMessage(content="Hola, como estas?") ] )

#print(respuesta)
print(respuesta.content)
