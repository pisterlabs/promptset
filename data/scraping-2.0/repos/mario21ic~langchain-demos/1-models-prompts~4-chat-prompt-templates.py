"""
Al igual que tenemos templates para modelos abiertos, LangChain tambien nos brinda templates para modelos de chat.
Estos templates nos ayudan a darle informacion a los modelos de chat en la manera en que lo necesitan.
Elementos de templates son:
- Human: texto que escribimos nosotros
- AI: texto que responde el modelo
- System: texto que se le envia al modelos para darle contexto de su funcionamiento
"""

import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


OPENAI_API_KEY = os.environ['API_KEY']
MODEL="text-davinci-003"
llm_openai = OpenAI(model_name=MODEL, openai_api_key=OPENAI_API_KEY)
chatgpt = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Prompt y template para el sistema
prompt_temp_sistema = PromptTemplate(
    template="Eres un asistente virtual que me recomienda una alternativa {adjetivo} a un producto",
    input_variables=["adjetivo"]
)
template_sistema = SystemMessagePromptTemplate(prompt=prompt_temp_sistema)

# Humano
prompt_temp_humano = PromptTemplate(template="{texto}", input_variables=["texto"])
template_humano = HumanMessagePromptTemplate(prompt=prompt_temp_humano)

chat_prompt = ChatPromptTemplate.from_messages([template_sistema, template_humano])

# Este es el formato del prompt que acabamos de armar
chat_prompt_value = chat_prompt.format_prompt(adjetivo="economica", texto="ipad").to_messages()
print(chat_prompt_value)

chat_rpta = chatgpt(chat_prompt_value)
print(chat_rpta)
