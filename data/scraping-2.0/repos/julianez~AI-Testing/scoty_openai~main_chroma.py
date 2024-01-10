import os

from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.prompts.chat import (
    PromptTemplate
)
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

openaiLLM = AzureOpenAI(
    azure_endpoint="https://openai-jez.openai.azure.com/",
    azure_deployment="GPT-35-turbo",
    model="GPT-35-turbo",
    api_version="2023-05-15"
)

embeddings = AzureOpenAIEmbeddings()

# load the vectorstore
db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
retriever = db.as_retriever()
    
qa = RetrievalQA.from_chain_type(
        llm=openaiLLM,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

while True:
    user_input = input("Hola, soy Scoty en que puedo ayudarte?\n")

    help_request = "Eres un asesor. Ayuda al usuario. Si no sabes la respuesta di que no tienes la información." +\
        f"\nUser:{user_input}"
    result = qa({"query": help_request})
    #print(len(result['source_documents']))
    print(result["result"])
