import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pinecone
from htmlTemplates import css, bot_template, user_template
import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import shutil
import os
import tempfile
import pypdf
import openai 


from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)

from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List



OPEN_AI_KEY = # insert
openai.api_key = # insert
PINECONE_API_KEY =  # insert
PINECONE_API_ENV =  # insert

def pdfs(pdf):

    # File exists, continue with your code logic
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    temp_file_path = os.path.join(temp_dir, "temp")

    # Copy the file to the temporary directory
    shutil.copyfile(pdf, temp_file_path)

    # Use the temporary directory path with the SimpleDirectoryReader
    documents = SimpleDirectoryReader(temp_dir).load_data()
    service_context = ServiceContext.from_defaults(chunk_size=1024)
    node_parser = service_context.node_parser

    nodes = node_parser.get_nodes_from_documents(documents)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # Cleanup the temporary directory
    shutil.rmtree(temp_dir)

    return nodes


def get_pdf_texts(pdf_docs):
    text_chunks = []  # Initialize as an empty list
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(pdf, temp_file)
            temp_file_path = temp_file.name

        nodes = pdfs(temp_file_path)
        text_chunks.extend(nodes)

    return text_chunks




class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


# store embeddings
def get_vectorstore(text_chunks):
    nodes = text_chunks
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
    return vector_index, keyword_index

# create chain that keeps track of questions & memory
def get_conversation_chain(vectorstore, keywordstore):
    vector_index  = vectorstore
    keyword_index = keywordstore
    # define custom retriever
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

    # define response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    # vector query engine
    vector_query_engine = RetrieverQueryEngine(
        retriever=vector_retriever,
        response_synthesizer=response_synthesizer
    )
    # keyword query engine
    keyword_query_engine = RetrieverQueryEngine(
        retriever=keyword_retriever,
        response_synthesizer=response_synthesizer,
    )
    return custom_query_engine

# have a conversation and handle the user imput
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title = "RETRO")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("RETRO")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process", 
            accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                text_chunks = get_pdf_texts(pdf_docs)

                # create vector store
                vectorstore, keywordstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, keywordstore)

if __name__ == '__main__':
    main()