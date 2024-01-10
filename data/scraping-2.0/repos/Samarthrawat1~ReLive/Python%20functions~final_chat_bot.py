import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import WhatsAppChatLoader
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv


os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI_API_KEY"

context = "You are ..."

def create_vector_db(path_to_chat):
    loader = WhatsAppChatLoader("example_data/whatsapp_chat.txt")
    data = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', 
                                        chunk_size=1000, 
                                        chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)
    return vectorStore_openAI

def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0",
        # verbose=True
    )

    vector_store= create_vector_db()

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        # return_source_documents=True,
        # verbose=True,
    )


if __name__ == "__main__":
    load_dotenv()

    chain = make_chain()
    chat_history = []

    while True:
        print()
        question = input("Question: ")

        # Generate answer
        response = chain({"question": question, "chat_history": chat_history})

        # Retrieve answer
        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(SystemMessage(content=context))
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        print(f"Answer: {answer}")