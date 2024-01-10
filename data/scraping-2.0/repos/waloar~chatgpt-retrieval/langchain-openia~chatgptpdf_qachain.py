import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory

import constants
# REF https://python.langchain.com/docs/use_cases/question_answering/

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]


# Loading documents.
loader = PyPDFDirectoryLoader("data/")
documents = loader.load()
# index = VectorstoreIndexCreator().from_loaders([loader])
print(len(documents))

# Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)

print(all_splits)
# Storage & Retrieval - Vectoriza los documentos
vectordb = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory='persist')
vectordb.persist()

# Generation
# Aplica el modelo de ChatGPT para conversacion

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

llm = OpenAI(temperature=0.8)

# Utiliza la extraccion de datos conversacionales de pregunta y respuesta.

chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None
