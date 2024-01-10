import os
import sys
import tempfile
from typing import Dict, Any, List, Optional
from uuid import UUID

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
# new code
from langchain.memory import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch

sys.path.insert(0, "/usr/lib/x86_64-linux-gnu")

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")


# @st.cache_resource(ttl="1h", experimental_allow_widgets=True)
def configure_retriever(uploaded_file, chunk_size, chunk_overlap):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    # write file to temp dir
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    docs.extend(loader.load())
    

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", "\n", " ", ""], 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)
    with st.expander("See how the documents are split"):
        st.divider()
        st.text_area(label="Raw Content", value=docs[0].page_content, disabled=True)
        st.divider()
        st.title("Split documents")
        # loop item in splits by item and index
        for idx, doc in enumerate(splits):
            st.write("-" * 80)
            st.caption(doc.page_content)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


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
    def __init__(self, container, writer):
        self.writer = writer

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        with self.writer.expander(f"Query"):
            self.writer.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        with self.writer.expander(f"Source documents"):
            for idx, doc in enumerate(documents):
                source = os.path.basename(doc.metadata["source"])
                self.writer.write(f"**Document {idx} from {source}**")
                self.writer.write(doc.page_content)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        with self.writer.expander(f"Rephrased question"):
            for p in prompts:
                self.writer.text(p)


openai_api_key = st.secrets["OPENAI_API_KEY"]
chunk_size = st.sidebar.number_input("Chunk size", min_value=0, max_value=1000, value=1000)

chunk_overlap = st.sidebar.number_input(
    "Chunk overlap", min_value=0, max_value=1000, value=int(chunk_size * 0.1)
)
uploaded_file = st.sidebar.file_uploader(
    label="Upload a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
)

if not uploaded_file:
    st.info("Please upload Markdown documents to continue.")
    st.stop()

st.session_state.file_uploaded = True

retriever = configure_retriever(
    uploaded_file, chunk_size, chunk_overlap
)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True)

# qa_chain = QAGenerationChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm,
                                                           retriever=retriever,
    memory=memory, verbose=True)
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}

for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container(), st)
        stream_handler = StreamHandler(st.empty())
        response = conversation_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        # response = llm(user_query, callbacks=[retrieval_handler, stream_handler])
