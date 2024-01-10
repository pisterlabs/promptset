
from dotenv import load_dotenv
import os
import shutil
import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import time
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
from PIL import Image

def show_notification(message, duration=5):
    # Display notification
    notification_placeholder = st.empty()
    notification_placeholder.markdown(message)

    # Wait for the specified duration
    time.sleep(duration)

    # Clear the notification after the specified duration
    notification_placeholder.empty()

def main():
    openai.api_key = "xxxxxxxxxxxxxx"
    os.environ['OPENAI_API_KEY'] = "xxxxxxxxxxxx"
    img = Image.open(r"images.jpeg")
    st.set_page_config(page_title="Profile Selection.AI", page_icon=img)
    st.header("Select the most Suitable Candidate📄")
    pdf = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)  # , accept_multiple_files = True)
    print(pdf)
    query = st.text_input("Ask your Question about your PDF")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None
    if st.session_state.conversation == None:
        if pdf is not None and len(pdf) > 0:
            docs = []
            for file in pdf:
                print(file.name)
                docs.extend(PyPDFLoader(file.name).load())
                print(docs)
         # Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                length_function = len,
                keep_separator="\n"
            )
            splits = text_splitter.split_documents(docs)
            print("doc_splits", len(splits))
            persist_directory = './docs/chroma/'
            if os.path.exists(persist_directory):
                # remove if exists
                shutil.rmtree(persist_directory)

            embedding_fun = OpenAIEmbeddings()
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embedding_fun,
                persist_directory=persist_directory
            )
            print("vectors", vectordb._collection.count())
            st.session_state.conversation = "Done"
            show_notification("Files Uploaded Successfully and Vector DB Created!👍👍", duration=5)
    else:
        if query:
            chat_history = []
            print(query)
            persist_directory = './docs/chroma/'
            embedding_fun = OpenAIEmbeddings()
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_fun)
            print("vectors", vectordb._collection.count())
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key = "answer"
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=vectordb.as_retriever(),
                # memory = memory,
                return_source_documents=True,
                verbose=False)
            result = qa_chain({"question": query, "chat_history": chat_history})
            # st.session_state.chat_history = result['chat_history']
            print(result)
            chat_history.append((query, result["answer"]))
            st.success(result["answer"])
            st.success("Name of the Best Profile: ")
            st.success(result["source_documents"][0].metadata["source"])

if __name__ == '__main__':
    main()



# Which profile is most suitable for a role of a Data Scientist and Why?
