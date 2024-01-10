import streamlit as st
import os
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import messages_from_dict, messages_to_dict

os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

history = ChatMessageHistory()
history.add_user_message("Hello")
history.add_ai_message("Hi, what do you want to know about the video?")
text_to_talk = """We are talking about this: O serviço best seller em utilização. Consiste na preparação do evento e registro de presença em tablets de forma off-line (sem internet no local). 
Você faz a criação do evento, importa ou digita os nomes dos convidados, marca quem é VIP, indica grupos e vários outro recursos.
Para esta versão você baixa o aplicativo ConfirmAki PRO (disponível para Android e iOS) gratuitamente. 

No dia do evento, você transfere o evento para o app nos tablets que serão usados na recepção e está tudo pronto para iniciar o registro de presença dos convidados. A partir deste ponto você não precisa mais da internet. Sua recepção pode ser em um parque, um salão, na praia, em um sítio ou qualquer lugar. Não é mais preciso da internet. 
Terminado o evento, retorne com os tablets a um local com internet e, com um simples toque, transfira para o servidor todos os registros realizados. Você já pode emitir um relatório PDF com os que compareceram ao evento ou baixar uma planilha com todos os dados e usar como desejar.

CONFIRMAKI PREMIUM
Quer mais segurança na recepção de convidados? Então use a versão Premium, que trabalha on-line (com internet de forma constante).
Usa o mesmo conceito da versão PRO, sendo que o registro da presença é sempre marcada no servidor. Assim, uma vez registrada a presença de um convidado, qualquer nova busca por este convidado indica que o mesmo já foi registrado, independente da recepcionista que atenda.
Para esta versão você baixa o aplicativo ConfirmAki Premium (disponível para Android e iOS) gratuitamente. 

Não é preciso transferir antecipadamente o evento para o tablet. Nesta versão, toda informação é buscada no servidor e atualizada imediatamente. Você pode aumentar ou reduzir o número de recepcionistas à vontade, sem interferir no registro de presença. Tudo está centralizado no servidor.
A qualquer momento, ou no final do evento, você tem todas as informações de presença, horário de registro, quem registrou e outras informações de apoio."""

memory = ConversationBufferMemory(chat_memory=history)

history.buffer = memory.load_memory_variables({})

memory = ConversationBufferMemory(chat_memory=history, ai_prefix="AI", user_prefix="User")

chat_gpt = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

conversation = ConversationChain(llm=chat_gpt, 
                                 memory=ConversationBufferMemory(),
                                 verbose=True
                                 )

conversation.prompt.templates = [
    """The following is a friendly conversation between a human and an AI. 
    The AI is a talkative and provides lots of specific details about the text that follows. 
    If the AI does not know the answer to a question say it and does not create any new information. <text>""" + text_to_talk + """</text>"""
]




