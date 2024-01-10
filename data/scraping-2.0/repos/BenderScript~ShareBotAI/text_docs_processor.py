#!/usr/bin/env python
# coding: utf-8
import os

from dotenv import dotenv_values
from langchain.document_loaders.directory import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


class TextDocProcessor:
    def __init__(self, llm=None, embeddings=None, env_file="./.env", temp_dir="./.tempdir"):

        """
        Initialize the TextDocProcessor class. This class packs everything needed to
        process documents, from loading from disk,  through creating embeddings and
        finally conversations.

        :param llm: If None, default is ChatOpenAI instance
        :param embeddings: If None, default is OpenAIEmbeddings instance
        :param env_file: path to the environment file, default is "./.azure.env"
        :param temp_dir: path to the temporary directory, default is "./.tempdir"
        """

        # Check if env_file exists when llm or embeddings are None
        if (llm is None or embeddings is None) and not os.path.exists(env_file):
            raise FileNotFoundError(f"No env_file found at {env_file}")

        self.config = dotenv_values(env_file)
        self.temp_dir = temp_dir

        if llm is None:
            self.llm = ChatOpenAI(openai_api_key=self.config["OPENAI_API_KEY"],
                                  temperature=self.config["OPENAI_API_TEMPERATURE"],
                                  model_name=self.config["OPENAI_API_MODEL_NAME"])
        else:
            self.llm = llm

        if embeddings is None:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.config["OPENAI_API_KEY"],
                chunk_size=1000)
        else:
            self.embeddings = embeddings

        # These are the "loaded" docs or files. Nomenclature changes a lot based on the library
        self.text_docs = []
        self.texts = None
        self.text_retriever = None
        self.qa_prompt = None
        self.text_conv_chain = None
        self.chat_history = []

    def load_text_docs(self, file_path=None):
        """
        Load text documents from a directory.

        :param file_path: path to the directory containing text documents, default is None
        :return: True if successful, False otherwise
        """

        if file_path is None:
            glob = "**/corrected_*.txt"
        else:
            glob = os.path.basename(file_path)
        loader = DirectoryLoader(self.temp_dir, glob=glob, recursive=True, silent_errors=True)

        try:
            self.text_docs = loader.load()
            return True
        except FileNotFoundError as e:
            print(f"Directory not found: '{e}'")
        except ValueError as e:
            print(f"Unsupported type: '{e}'")
        except Exception as e:
            print(f"Unexpected error: '{e}'")
        return False

    def split_text_docs(self):
        """
        Split text documents into chunks.

        :return: True if successful, False otherwise
        """
        # TODO Not quite the correct logic here, but it works for now
        # Calculate total length of all documents
        total_length = sum(len(doc.page_content) for doc in self.text_docs)

        # Maximum number of chunks, Azure AI limitations
        max_chunks = 16

        # Calculate chunk size, max of 4000 characters
        chunk_size = max(total_length // max_chunks, 4000)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50, length_function=len,
                                                       is_separator_regex=False, )
        self.texts = text_splitter.split_documents(self.text_docs)
        return True

    def create_text_retriever(self):
        """
        Create a text retriever using Chroma and the embeddings.

        :return: True if successful, False otherwise
        """

        db = Chroma.from_documents(self.texts, self.embeddings)

        self.text_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        return True

    def create_text_conv_chain(self):
        """
        Create a conversational retrieval chain for having a conversation based on retrieved documents.

        :return: True if successful, False otherwise
        """

        template = """
        Given the following document collection and context, respond to the best of your ability:
        Context:  {context}
        Chat History:  {chat_history}
        Question: {question}
        """

        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        self.text_conv_chain = ConversationalRetrievalChain.from_llm(self.llm, retriever=self.text_retriever,
                                                                     memory=memory, verbose=True,
                                                                     combine_docs_chain_kwargs={"prompt": PROMPT}
                                                                     )

        return True
