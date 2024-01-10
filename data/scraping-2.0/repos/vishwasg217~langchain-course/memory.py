from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import dotenv_values

config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]


llm = ChatOpenAI(temperature=0, openai_api_key=OPEN_AI_API, model_name="gpt-3.5-turbo")

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

loader = PyPDFLoader("example.pdf")
documents = loader.load()
# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# select which embeddings we want to use
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
# create a chain to answer questions 
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
chat_history = []

query = ""

while(query != "exit"):
    query = input("Query: ")
    result = qa({"question": query, "chat_history": ""})
    print("Result: ", result)

    hist = memory.load_memory_variables({})
    print("History: ", hist)
