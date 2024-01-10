import lib.settings as settings

from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from typing import Any, Dict, List

from lib.stcallbackhandler import StreamingStreamlitCallbackHandler

global outputArea

# Load environment variables
load_dotenv()

def main():
    settings.init()
    st.title("Chat with your documents")
    
    st.write(settings.chroma._collection.count(), "documents in the database")

    outputArea = st.empty()
    clearButton = st.button('Clear')

    cbhandler = StreamingStreamlitCallbackHandler()
    cbhandler.set_output_area(outputArea)

    ollama = Ollama(base_url = settings.ollamaBaseURL, 
                    model = settings.ollamaModel,
                    callback_manager = CallbackManager([cbhandler]))

    if clearButton:
        outputArea.empty()

    if query := st.chat_input(placeholder = "Who are you?"):
        with outputArea:
            st.spinner("Running query…")
            docs = settings.chroma.similarity_search(query)
            qachain = RetrievalQA.from_chain_type(llm = ollama, chain_type = "stuff", retriever = settings.chroma.as_retriever())
            result = qachain({"query": query})
#        st.write(result["result"])
            
if __name__ == "__main__":
    main()