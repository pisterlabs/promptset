from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.settings import settings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


class LLMChain:
    def __init__(self, document: str, llm_obj):
        self.document = document
        self.pages = None
        self.embedding = None
        self.vector_store = None
        self.retriever = None
        self.llm_obj = llm_obj

    def set_embedding(self, embedding):
        """
        Set the embedding to be used for the document.
        Defaults to OpenAIEmbeddings.
        """
        self.embedding = embedding

    def set_pages(self):
        """
        Split the document into pages.
        """
        if len(self.document) < 100:
            raise Exception("Not enough data.")
        self.pages = self._document_splitter()

    def set_retriever(self, k=3):
        """
        Create a retriever for the document.
        """
        if not self.vector_store:
            raise Exception("Vector store not set, run create_vector_store() first.")
        self.retriever = self._setup_retriever(k=k)

    def _setup_retriever(self, k=3):
        """
        Private function: Create a compression retriever for the document.
        Use set_retriever() to create a retriever.
        """
        compressor = LLMChainExtractor.from_llm(self.llm_obj)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
        return compression_retriever

    def setup_qa_chain(self, chain_type: str):
        """
        Setup a QA chain for the document.
        """
        if not self.retriever:
            raise Exception("Retriever not set, run set_retriever() first.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_obj,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=True,
        )
        return qa_chain

    def create_vector_store(self, db_directory):
        """
        This function creates a vector store for the document.
        vector store is a database of embeddings for the document.
        """
        self.vector_store = self._setup_vector_store(db_directory=db_directory)

    def _document_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n", " "]
        )
        pages = text_splitter.create_documents([self.document])
        return pages

    def _setup_vector_store(self, db_directory, persist=True):
        """
        Private function: Create a vector store for the document.
        """
        if not self.pages:
            raise Exception("Pages not set, run set_pages() first.")

        if not self.embedding:
            raise Exception("Embedding not set, run set_embedding() first.")

        vectordb = Chroma.from_documents(
            documents=self.pages,
            embedding=self.embedding,
            persist_directory=db_directory,
        )
        if persist:
            vectordb.persist()
        return vectordb


def get_llm_chain_obj(documents):
    """
    This is a helper function to create an singleton LLM chain object
    with OpenAI as the LLM.
    """
    llm_obj = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    llm_service = LLMChain(documents, llm_obj=llm_obj)

    llm_service.set_pages()
    llm_service.set_embedding(OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY))
    llm_service.create_vector_store("db")
    llm_service.set_retriever(k=3)
    llm_chain = llm_service.setup_qa_chain("stuff")

    return llm_chain
