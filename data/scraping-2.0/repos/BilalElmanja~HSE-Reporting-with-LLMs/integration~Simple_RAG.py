# setup langchain
import os

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings


class DocumentProcessor:
    def __init__(self, directory, glob_pattern, loader_class):
        self.directory = directory
        self.glob_pattern = glob_pattern
        self.loader_class = loader_class
        self.documents = None
        self.texts = None

    def load_documents(self):
        loader = DirectoryLoader(self.directory, glob=self.glob_pattern, loader_cls=self.loader_class)
        self.documents = loader.load()
        return self.documents

    def split_text(self, chunk_size=200, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.texts = text_splitter.split_documents(self.documents)
        return self.texts


class EmbeddingManager:
    def __init__(self, model_name, normalize_embeddings=True, persist_directory=None):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.persist_directory = persist_directory
        self.embedding = None
        self.vector_db = None

    def create_embedding(self):
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': self.normalize_embeddings}
        )

    def embed_documents(self, documents):
        if not self.embedding:
            self.create_embedding()

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )

    def get_retriever(self, k=1):
        return self.vector_db.as_retriever(search_kwargs={"k": k})



class QuestionAnsweringChain:
    def __init__(self, llm, chain_type, retriever):
        self.llm = llm
        self.chain_type = chain_type
        self.retriever = retriever
        self.qa_chain = None

    def create_qa_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=self.retriever,
            return_source_documents=True
        )

    def answer_question(self, question):
        if not self.qa_chain:
            self.create_qa_chain()

        llm_response = self.qa_chain.answer(question)
        return process_llm_response(llm_response)


# Example usage:
# response = qa_chain.answer_question("Your question here")


## Cite sources

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    text = wrap_text_preserve_newlines(llm_response['result'])
    sources = []
    for source in llm_response["source_documents"]:
        sources.append(source.metadata['source'])

    return text, sources


