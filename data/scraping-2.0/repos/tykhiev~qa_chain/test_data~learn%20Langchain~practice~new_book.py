# import document and load it
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
import os
import chromadb
from chromadb.config import Settings
client = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port="8000"
                                  ))

openai_api_key = os.getenv("OPENAI_API_KEY")
path = "C:/Users/USER/OneDrive/Documents/ai-chatbot-3/server/test_data/learn Langchain/pdfcoffee_com_113328813_the_definitive_book_of_chinese_astrologypdf.pdf"

loader = PyPDFLoader(path)
document = loader.load()

# load qa chain
llm = OpenAI(temperature=0.1)

# split document into chunks and state embeddings
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents=document)
# state embedding
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# create vector store to use as index
vectordb = Chroma.from_documents(docs, embeddings)
# expose this index in a retriever interfaces
retriever = vectordb.as_retriever(search_type="similarity", kwargs={"k": 2})
# create a question answering chain
qa = ConversationalRetrievalChain.from_llm(
    retriever=retriever, llm=llm)
chat_history = []
query = input("ask something: ")
# ask a question
while True:
    response = qa({"question": query, "chat_history": chat_history})
    chat_history.append(f"User: {query}\nAI: {response['answer']}\n")
    print(response["answer"])
