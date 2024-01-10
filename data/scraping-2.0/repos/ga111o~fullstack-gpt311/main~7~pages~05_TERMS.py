import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


st.set_page_config(
    page_title="DOCUMENT",
    page_icon="📄",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


ollama = ChatOllama(
    model="llama2:7b",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embedings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append(
            {"messages": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["messages"], message["role"], save=False)


st.title("ga111o! TERMS")

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Just answer the key point using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
    ]
)

st.markdown("""
    ### UPLOAD FILE ON THE SIDEBAR
""")

with st.sidebar:
    file = st.file_uploader("upload file", type=[
                            "pdf", "txt", "docs", "jpg", "png"])


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.status("loading..."):
    if file:
        st.write("embedding...")
        retriever = embed_file(file)
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
            }
            | template
            | ollama
        )
        st.write("chain invoking...")
        send_message(chain.invoke(retriever), "ai")
        st.balloons()

    else:
        st.session_state["messages"] = []
