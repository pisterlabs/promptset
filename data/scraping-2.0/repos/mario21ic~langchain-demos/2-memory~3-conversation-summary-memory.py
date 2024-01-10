"""
Esta memoria en vez de almacenar un registro detallado de las interacciones,
almacena un resumen de la conversación. Muy util para evitar prompts muy largos
"""

import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory



os.environ['OPENAI_API_KEY']


llm = ChatOpenAI(stop=["\nHuman"])
#llm_k = OpenAI(stop=["\nHuman"])
memoria = ConversationSummaryMemory(llm=llm)
chatbot_resumen = ConversationChain(llm=llm, memory=memoria, verbose=True)

chatbot_resumen.predict(input="Hola como estás? Me llamo Mario y soy programador")
print(memoria.chat_memory.messages)

chatbot_resumen.predict(input="Me gusta mucho la tecnologia, IoT e inteligencia artificial son mis favoritos")
print(memoria.chat_memory.messages)


chatbot_resumen.predict(input="Que te gusta a ti de la tecnologia?")
print(memoria.chat_memory.messages)
