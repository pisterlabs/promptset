"""
Esta memoria va a empezar a generar un grafo de conocimiento en lugar de historico.
De esa forma responde acorde a ese contexto de conocimiento.
El spanglish puede generar issues
"""

import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory
import networkx as nx

os.environ['OPENAI_API_KEY']


llm = ChatOpenAI(stop=["\nHuman"])
#llm_k = OpenAI(stop=["\nHuman"])
memoria = ConversationKGMemory(llm=llm)
chatbot_kgm = ConversationChain(llm=llm, memory=memoria, verbose=True)

print(chatbot_kgm.predict(input="Hola como estas? Me llamo Mario y soy programador"))
print(chatbot_kgm.memory.kg.get_triples())

print(chatbot_kgm.predict(input="Mi perro se llama Blackie"))
print(chatbot_kgm.memory.kg.get_triples())

print(chatbot_kgm.predict(input="Como me llamo?")) # No va a saber responder porque no lo sabe
print(chatbot_kgm.memory.kg.get_triples())


print(chatbot_kgm.predict(input="A que se dedica Mario?"))
print(chatbot_kgm.memory.kg.get_triples())
