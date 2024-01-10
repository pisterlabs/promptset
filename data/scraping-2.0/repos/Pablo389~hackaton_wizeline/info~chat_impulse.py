from dotenv import dotenv_values
env_vars = dotenv_values(".env")
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']

#Librerias para las cadenas y memoria
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory


#Librerias para el embedding y retrieval
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

#Librerias para los prompts de chat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def vector_qa(direc_marca):
    #Directorio donde se guarda toda la info de los embeddings en chroma
    persist_directory=direc_marca
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    #Template que se le va a pasar ael retrieval para que obtenga la info que quiero (A/B test)
    prompt_template = """Usa el siguiente contexto para responder la pregunta al final. Si no la sabes o la encuentras no la trates de inventar.

    {context}

    Pregunta: {question}
    Tu respuesta:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    #qa impulse es la herramienta con la que haremos retrieval de documentos, retorna texto
    qa_marca = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), chain_type= "stuff", retriever = vectordb.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    
    return qa_marca

def chat_prompt():
    #Template del prompt que va a ser enviado al modelo de texto para que me regrese el mejor mensaje
    template="""Eres un chatbot especializado en atención al cliente. Tu trabajo es responder al cliente de la manera mas concisa y segura posible. Este es el historial de mensajes:
    {chat_history}
    Y esta es la información que recabaste para poder responder la pregunta:
    {qa_answer}

    A continuacion esta el mensaje del cliente:
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    cht_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return cht_prompt

def chat(qa_marca):
    #Inicializar la memoria
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="text", return_messages=True)
    chat=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    chain = LLMChain(llm=chat, prompt=chat_prompt(), memory=memory, verbose=True)
    while True:
        pregunta = input()
        if pregunta == "exit":
            break
        respuesta_impulse = qa_marca.run(pregunta)
        resp_ia = chain.run(question = pregunta, text = pregunta, qa_answer = respuesta_impulse)
        print(resp_ia)

if __name__ == "__main__":
    directorio = "wizeline_chroma"
    qa_marca = vector_qa(directorio)
    chat(qa_marca=qa_marca)

