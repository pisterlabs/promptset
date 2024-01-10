# https://github.com/shahidul034/Chat-with-pdf-using-LLM-langchain-and-streamlit
import streamlit as st
import os
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from enum import Enum

from tools.tool_base import ToolBase, DEMO_PATH
from helper import (
    extract_text_from_uploaded_file,
    extract_text_from_file,
    extract_text_from_url,
    get_var,
)

DEMO_FILE = DEMO_PATH + "documents.csv"


class InputFormat(Enum):
    DEMO = 0
    FILE = 1
    URL = 2


class PdfChat(ToolBase):
    def __init__(self, logger):
        super().__init__(logger)
        self.title = "💬 PDF-Chat"
        self.model = "gpt-3.5-turbo"
        self.formats = ["Demo", "PDF Datei hochladen", "URL"]
        self._input_type = None
        self.text = None
        self._input_file = None
        self.user_prompt = None
        self.default_prompt = "Fasse das Dokument zusammen."
        self.response = None

        self.script_name, _ = os.path.splitext(__file__)
        self.intro = self.get_intro()

    @property
    def input_type(self):
        return self._input_type

    @input_type.setter
    def input_type(self, value):
        if value != self._input_type:
            self.text = None
        self._input_type = value

    @property
    def input_file(self):
        return self._input_file

    @input_file.setter
    def input_file(self, value):
        if value != self._input_file:
            self.text = None
        self._input_file = value

    def process_text(self, text):
        """
        Process the given text by splitting it into chunks, converting the chunks into embeddings,
        and forming a knowledge base using FAISS.

        Args:
            text (str): The input text to be processed.

        Returns:
            knowledgeBase: The knowledge base formed from the input text.
        """
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        knowledgeBase = FAISS.from_texts(chunks, embeddings)
        return knowledgeBase

    def get_demo_documents(self):
        df = pd.read_csv(DEMO_FILE, sep=";")
        documents_dic = dict(zip(df["file_path"], df["title"]))
        return documents_dic

    def show_settings(self):
        self.input_type = st.radio("Input Format", options=self.formats)
        if self.formats.index(self.input_type) == InputFormat.DEMO.value:
            doc_options = self.get_demo_documents()
            self.input_file = st.selectbox(
                "Dokument auswählen",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
            )
            self.text = extract_text_from_file(DEMO_PATH + self.input_file)
        elif self.formats.index(self.input_type) == InputFormat.FILE.value:
            self.input_file = st.file_uploader(
                "PDF Datei",
                type=["pdf"],
                help="Lade die Datei hoch, dessen Text du analysieren möchtest.",
            )
            if self.input_file:
                self.text = extract_text_from_uploaded_file(self.input_file)
        elif self.formats.index(self.input_type) == InputFormat.URL.value:
            self.input_file = st.text_input(
                "URL",
                help="Gib die URL des PDF-Dokuments ein, dessen Text du analysieren möchtest.",
            )
            if self.input_file:
                self.text = extract_text_from_url(self.input_file)

        if self.text is not None:
            with st.expander("Vorschau Text", expanded=True):
                st.markdown(self.text)

    def check_input(self):
        """
        Checks if the text and user_prompt attributes are not None.

        Returns:
            bool: True if both text and user_prompt are not None,
                  False otherwise.
        """
        ok = self.text is not None and self.user_prompt is not None
        return ok

    def run(self):
        self.user_prompt = st.text_area(
            label="Stelle eine Frage zum Dokument",
            value=self.default_prompt,
            height=100,
        )
        ok = self.check_input()
        if st.button("📨 Abschicken", disabled=(ok is False)):
            knowledgeBase = self.process_text(self.text)
            docs = knowledgeBase.similarity_search(self.user_prompt)
            llm = OpenAI(api_key=get_var("OPENAI_API_KEY"))

            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cost:
                self.response = chain.run(
                    input_documents=docs, question=self.user_prompt
                )
                print(cost)
        if self.response is not None:
            st.markdown(f"**🤖 Antwort:**<br>{self.response}", unsafe_allow_html=True)
