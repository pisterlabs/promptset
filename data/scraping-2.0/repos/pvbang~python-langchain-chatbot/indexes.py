import tempfile
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import Pinecone
from PyPDF2 import PdfReader
import pinecone

from config import config

def create_indexes(file: tempfile, pinecone_api_key: str, pinecone_environment: str, pinecone_index_name: str, openai_api_key: str) -> str:
    try:
        file_path = file.name
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        output_file_path = os.path.join(
            config.OUTPUT_DIR,
            'output.txt'
        )
        with open(output_file_path, 'w') as file:
            file.write(text)
        loader = DirectoryLoader(
            f'{config.OUTPUT_DIR}',
            glob='**/*.txt',
            loader_cls=TextLoader
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=0
        )
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key
        )
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        indexes_list = pinecone.list_indexes()
        if pinecone_index_name not in indexes_list:
            pinecone.create_index(
                name=pinecone_index_name,
                dimension=1536
            )
        Pinecone.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=pinecone_index_name
        )
        os.unlink(output_file_path)
        return 'Tài liệu và index upload oke rồi. Chat thôi.'
    except Exception as e:
        return e

def clear_indexes(pinecone_api_key: str, pinecone_environment: str, pinecone_index_name: str) -> str:
    try:
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        indexes_list = pinecone.list_indexes()
        if pinecone_index_name in indexes_list:
            pinecone.delete_index(name=pinecone_index_name)
        return 'Đã xóa indexes.', None
    except Exception as e:
        return e, None
    