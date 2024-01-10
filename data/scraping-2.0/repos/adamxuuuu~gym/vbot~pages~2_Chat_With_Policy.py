import os
import streamlit as st
from pprint import pprint
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from utils.callbacks import StreamHandler, RetrievalHandler

from config import (
    EMBEDDING,
    PROMPT_TEMPLATE,
    LLM,
    SEARCH_KWARGS,
    SEARCH_TYPE,
    CHUNK_SIZE,
    DB_BASE
)


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)

@st.cache_resource()
def retriever(db_path: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING)

    vector_db = FAISS.load_local(
        folder_path=db_path,
        embeddings=embeddings,
        normalize_L2=True,
    )

    retriever = vector_db.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs=SEARCH_KWARGS
    )

    return retriever

# ======================APP==========================
msgs = StreamlitChatMessageHistory()

llm = LlamaCpp(
    model_path=LLM,
    temperature=0.01,
    n_ctx=3000,
    streaming=True,
    max_tokens=512
)

chain_type_kwargs = {"prompt": PROMPT}
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever(f'{DB_BASE}/cs{CHUNK_SIZE}/{os.path.basename(EMBEDDING)}'),
    chain_type_kwargs=chain_type_kwargs
)

# Sidebar
with st.sidebar:
    if st.sidebar.button("Clear message history") and not msgs:
        msgs.clear()
        msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me about VW policy related question."):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = RetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())

        pprint(f"Start processing query: '{user_query}'")
        response = qa_chain.run(user_query, callbacks=[
                                retrieval_handler, stream_handler])

        pprint(f"Finish generation: {response}")
