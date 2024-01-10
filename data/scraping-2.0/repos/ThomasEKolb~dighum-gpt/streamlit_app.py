"""
# Welcome to this tutorial!

This tutorial code is based on the tutorial apps of streamlit.

Link: https://github.com/langchain-ai/streamlit-agent/tree/main/streamlit_agent
"""

from my_config import *

import os
os.environ["OPENAI_API_KEY"] = open_ai_key_from_config_file # workaround to hide the api key
openai_api_key = os.getenv('OPENAI_API_KEY')

import streamlit as st

st.set_page_config(page_title="DigHum GPT", page_icon="🦜")
st.title("📚 DigHum GPT")

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Setup memory for contextual conversation
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# LLM
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)

# Embedding
from langchain.embeddings import OpenAIEmbeddings
embedding = OpenAIEmbeddings(chunk_size=1,deployment='text-embedding-ada-002')

# Vector store location
persist_directory = './db'

# Retriever
import wget
from langchain.document_loaders import PDFMinerLoader
@st.cache_data
def load_book():
    url_book = "https://owncloud.tuwien.ac.at/index.php/s/FW7Y2GNUOaUtrhf/download" # book as pdf
    wget.download(url_book) # download book
    loader = PDFMinerLoader("./978-3-030-86144-5.pdf")
    book = loader.load()
    return book

from langchain.text_splitter import RecursiveCharacterTextSplitter
@st.cache_data
def split_book_into_chunks(_book):
    # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(_book)
    return texts

from langchain.vectorstores import Chroma
def create_embeddings(book_chunks):
    chroma_db = Chroma.from_documents(documents=book_chunks, embedding=embedding, persist_directory=persist_directory)
    chroma_db.persist() # store embeddings
    return chroma_db.as_retriever()

import zipfile
def load_precomputed_embeddings():
    # download the embeddings (to save some time and cost)
    url_embeddings = "https://owncloud.tuwien.ac.at/index.php/s/xwFts1mQo3VXiCg/download" # embeddings of the book
    embeddings_filename = wget.download(url_embeddings)

    with zipfile.ZipFile('./'+embeddings_filename, 'r') as zip_ref:
        zip_ref.extractall('./')

    # load data from vector store
    chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return chroma_db.as_retriever()

def setup_retriever():
    load_state = st.text('Downloading DigHum book...')
    book = load_book()
    load_state.text('Loading book data...done!')
    load_state.text('Splitting up DigHum book into chunks...')
    book_chunks = split_book_into_chunks(book)
    load_state.text('Splitting up DigHum book data...done!')
    load_state.text('Loading precomputed embeddings...')    
    # chroma_db = create_embeddings(book_chunks)
    chroma_db_as_retriever = load_precomputed_embeddings()
    load_state.text('Loading precomputed embeddings...done!')
    load_state.empty()
    load_state = st.markdown('This tool queries the book [Perspectives on Digital Humanism](https://link.springer.com/book/10.1007/978-3-030-86144-5) please ask me anything about it!')
    return chroma_db_as_retriever

retriever = setup_retriever()

# QA Chain
from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
