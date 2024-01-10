"""
Conversation buffer
Esta es una de las memorias más básicas, cada prompt y respuesta del modelos se almacenara en la memoria. Cada vez que se le envia un nuevo prompt al modelo se envia todo el historico de las interacciones.
La conversación se salva como pares de mensajes entre "Human" y "AI", por lo cual tambien lo podemos integrar con modelos como GPT3.5 Turbo
"""

import os

from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain


os.environ['OPENAI_API_KEY']

llm = ChatOpenAI()
memoria = ConversationBufferMemory()
chatbot = ConversationChain(llm=llm, memory=memoria, verbose=True)

chatbot.predict(input = "Hola como estas? me llamo Mario, soy Developer y mi lenguaje de programación favorito es C")
# chatbot.predict(input = "Como me llamo, a que me dedico y que lenguaje me gusta mas?")
chatbot.predict(input = "Como me llamo?")
# chatbot.predict(input = "A que me dedico y que lenguaje de programación me gusta mas?")
chatbot.predict(input = "que lenguaje de programación me gusta mas?")
print(memoria.chat_memory.messages)
