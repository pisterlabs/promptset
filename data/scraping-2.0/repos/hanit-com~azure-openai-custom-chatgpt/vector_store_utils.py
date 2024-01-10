import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

storage_path = "faiss_vector_store"


def create_vector_store():
    openai_api_version = "2023-05-15"
    data_path = "context_data/data"

    def get_file_names(path):
        result = []
        for _, _, files in os.walk(path):
            for file in files:
                result.append(file)
        return result

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                   chunk_overlap=50,
                                                   separators=["\n\n", ""])
    file_names = get_file_names(data_path)
    documents = []

    for file_name in file_names:
        file_path = data_path + "/" + file_name

        loader = UnstructuredFileLoader(file_path)
        documents += loader.load_and_split(text_splitter=text_splitter)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",
                                  chunk_size=1,
                                  openai_api_version=openai_api_version)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(storage_path)


def get_retriever():
  embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
  faiss_db = FAISS.load_local(storage_path, embeddings)
  retriever = faiss_db.as_retriever()
  return retriever