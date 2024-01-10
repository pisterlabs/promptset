from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

class DatasetVectorizer:
    """
        A class for vectorizing datasets.
    """
    def __init__(self):
        pass

    def vectorize(self, text_file_paths, chunk_size=1000, chunk_overlap=500, openai_key=""):
        documents = []
        for text_file_path in text_file_paths:
            doc_loader = TextLoader(text_file_path)
            documents.extend(doc_loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=chunk_overlap, chunk_size=chunk_size)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        docsearch = Chroma.from_documents(texts, embeddings)

        return documents, texts, docsearch