import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

st.set_page_config(
    page_title="Cosmetics GPT",
    page_icon="🐼",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=150,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a kind, friendly and professional consultant at Chowis Company. 
            You will receive consultation on FAQs about Chowis skin diagnosis services using DERMOPRIME & DERMOSMART device or cosmetic product recommendations.
            Answer the question ONLY using the following context.
            If you don't know the FAQ answer well, just say you don't know and GIVE official FAQ Inquiry Form URL("https://support-bot.chowis.cloud/en/inquiry?app_id=44") as answer. DON'T make anything up.
            If you are recommending a product, You must ONLY recommend product given in the context and Describe the product in as much detail as possible in the given context but .
            You should NEVER recommend other products.
            Also, when recommending products, consider the 'Type' as First, 'Main Effect' as Second.
            If there is no product that satisfies all given conditions, just say you don't know and GIVE official Hompage URL("https://www.chowis.com") as answer. DON'T make anything up.
            
     
            
            Context: {context}
            """,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

st.title("Choice_Cosmetics")

st.markdown(
    """
    Welcome to CHOWIS GPT!
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a file",
        type=["pdf", "txt", "docx", "xlsx"],
    )

if file:
    retriever = embed_file(file)
    send_message(
        "I'm ready! Ask away!",
        "ai",
        save=False,
    )
    paint_history()
    message = st.chat_input("Ask anything about your file")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
