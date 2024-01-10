from abc import ABC
import ssl
import os
from unstructured.partition.auto import partition
from unstructured.partition.xlsx import partition_xlsx
from unstructured.staging.base import convert_to_dict
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import streamlit as st
import pandas as pd
from loguru import logger
import json
import chromadb
from dotenv import main
from chromadb.utils import embedding_functions
from langchain.callbacks import StreamlitCallbackHandler
import tempfile
import nltk

from intriq.table_representation import table_to_docs

main.load_dotenv()
_PARTITION_STRATEGY = 'hi_res'
_PARTITION_MODEL_NAME = 'yolox'
_OPEN_AI_MODEL_NAME = 'gpt-3.5-turbo-0613'

# Download necessary NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

st.set_page_config(
    page_title="intriq data chatbot", page_icon="🦜")
st.title("🔥🤖🔥 intriq data chatbot 🔥🤖🔥")


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


class DocSplitter(ABC):
    partition_func = None
    partition_args = None

    @classmethod
    def create(cls, file):
        try:
            ext = os.path.splitext(file.name)[1][1:].lower()
        except:
            ext = file.split(".")[-1]

        if ext.lower() in ['xlsx', 'xls']:
            return TableSplitter(file)
        else:
            return GeneralDocSplitter(file)

    def __init__(self, file):
        self._file = file
        self._elements = None

    def __call__(self):
        self._partition()
        return self._chunk()

    # @st.cache_data(ttl="2h")
    def _partition(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self._file.getvalue())
            tmp_file_path = tmp_file.name
        elements = self._partition_mthd(tmp_file_path)
        dict_elements = convert_to_dict(elements)
        self._elements = self._format(dict_elements)

    def _partition_mtd(self, file_name):
        pass

    def _chunk(self):
        pass

    def _format(self, elements):
        pass


class GeneralDocSplitter(DocSplitter):
    def _chunk(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for element in self._elements:
                tmp_file.write((element + "\n\n").encode('utf-8'))

        loader = TextLoader(
            tmp_file.name,
            encoding='utf-8'
        )
        docs = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f'Created chunks of length {len(chunks)}')
        return chunks

    def _format(self, elements):
        extracted_elements = []
        for element in elements:
            if element["type"] == "Table" and element["metadata"].get("text_as_html"):
                logger.info('found a table 🔥')
                # logger.info(element)
                extracted_elements.append(element["metadata"])
            else:
                logger.info(f'found a {element["type"]}')
                extracted_elements.append(element["text"])
        return extracted_elements

    def _partition_mthd(self, file_name):
        return partition(
            filename=file_name,
            strategy=_PARTITION_STRATEGY,
            model_name=_PARTITION_MODEL_NAME,
            pdf_infer_table_structure=True
        )


class TableSplitter(DocSplitter):
    def _chunk(self):
        docs = []
        for element in self._elements:
            if not element.get('text_as_html'):
                logger.info('no html found in table element, skipping')
            docs.extend(
                table_to_docs(
                    element['text_as_html'],
                    doc_name=self._file.name,
                    table_name=element['filename']
                )
            )
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=0,
            length_function=len
        )
        chunks = splitter.split_documents(docs)
        logger.info(f'Created table chunks of length {len(chunks)}')

        return chunks

    def _format(self, elements):
        extracted_elements = []
        for element in elements:
            if element["type"] == "Table" and element["metadata"].get("text_as_html"):
                logger.info('found a table 🔥')
                # logger.info(element)
                extracted_elements.append(element["metadata"])
            else:
                logger.warning(f'found a {element["type"]} in table')
        return extracted_elements

    def _partition_mthd(self, file_name):
        return partition_xlsx(
            filename=file_name,
            strategy=_PARTITION_STRATEGY,
            model_name=_PARTITION_MODEL_NAME,
            pdf_infer_table_structure=True
        )


# def retrieve_and_log_context(query, retriever):
#     """
#     Retrieve context documents and log them.
#     :param query: The user query
#     :param retriever: The document retriever instance
#     :return: Serialized context documents as string
#     """
#     relevant_docs = retriever.get_relevant_documents(query)
#     # serialized_context = format_documents_as_string(relevant_docs)
#     logger.info(f"Retrieved Context: {relevant_docs}")
#     return relevant_docs


# drop files here
uploaded_files = st.file_uploader(
    "Drop all your shit here 💩",
    accept_multiple_files=True,
    help="Various File formats are supported",
    on_change=clear_submit,
)

if not uploaded_files:
    st.warning(
        "What am I a mind reader? Upload some data files if you want to chat with me."
    )

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        embedder = DocSplitter.create(uploaded_file)
        docs.extend(embedder())
    embeddings = OpenAIEmbeddings()
    logger.info(f'about to embed. Len of docs {len(docs)}')
    db = Chroma.from_documents(docs, embeddings)


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Whaddup? Ask me something about something. Or don't, that's fine.."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    docs = db.similarity_search(prompt)
    for doc in docs:
        logger.info(doc.page_content)
        logger.info(doc.metadata)
    llm = ChatOpenAI(
        model_name=_OPEN_AI_MODEL_NAME,
        temperature=0,
        streaming=True
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever()
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)
        response = qa_chain(
            {'query': prompt},
            callbacks=[st_cb]
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
