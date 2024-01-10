from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LangchainClient:
    def __init__(self):
        pass

    def get_document_from_url(self, url):
        """
        Get a document from a URL.
        """
        loader = WebBaseLoader(url)
        doc = loader.load()
        return doc

    def split_text(self, data, chunk_size=500, chunk_overlap=0):
        """
        Split text into chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_docs = text_splitter.split_documents(data)
        return split_docs
