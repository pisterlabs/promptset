import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# -------------
import time
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource(show_spinner="Embedding file..")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


model = ChatOpenAI(
    openai_api_key=api_key,
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Answer the question using ONLY the following context. If you don't know the answer just say
        you don't know. DON'T make anything up.

        Context: {context}
    """,
        ),
        ("human", "{question}"),
    ]
)

st.title("PrivateGPT")

st.markdown(
    """
Welcome!

Upload your file!
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
    )

if file:
    retriever = embed_file(file)

    send_message("I'm ready, ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file..")
    if message:
        send_message(message, "human")
        docs = retriever.invoke(message)
        docs = "\n\n".join(document.page_content for document in docs)
        prompt = template.format_messages(context=docs, question=message)
        with st.chat_message("ai"):
            response = model.predict_messages(prompt).content
        # send_message(response, "ai")
else:
    st.session_state["messages"] = []
