"""
Esta memoria es igual a la anterior, con la diferencia que se puede definir una
ventana de mensajes a recordar en vez de recordar todo el historico de interacciones.
"""

import os

from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory


os.environ['OPENAI_API_KEY']

llm = ChatOpenAI()
memoria = ConversationBufferWindowMemory(k=2) # ventana de 2 ultimos mensajes
chatbot = ConversationChain(llm=llm, memory=memoria, verbose=True)

chatbot.predict(input = "Hola, soy Mario soy developer")
# print(memoria.chat_memory.messages)

chatbot.predict(input = "como me llamo?")
chatbot.predict(input = "soy programador?")

print(memoria.chat_memory.messages)

# Veamos como olvida nuestro nombre por solo guardar 2 ultimos mensajes
chatbot.predict(input = "cuanto mide el empire state?")
chatbot.predict(input = "quien era pulgarcito?")

chatbot.predict(input = "como me llamo?")
print(memoria.chat_memory.messages)
