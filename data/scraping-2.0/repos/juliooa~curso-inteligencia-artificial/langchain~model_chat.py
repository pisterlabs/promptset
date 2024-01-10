"""Using the chat model to translate from spanish to english"""
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


chatModel = ChatOpenAI(openai_api_key='TU_API_KEY')

messages = [
    SystemMessage(content="Tu eres un traductor de español a ingles"),
    HumanMessage(
        content='''
Hola, como estas? ayudame a traducir esto: "Quiero aprender inteligencia artificial"
'''),
AIMessage(content="La traducción es: I want to learn artificial intelligence"),
HumanMessage(
        content='''
Gracias, y como se dice "no tengo mucho tiempo para aprender"?
'''),
]

response = chatModel(messages)

print(response)
