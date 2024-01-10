import os
import requests
from omegaconf import DictConfig
from agent.utils.configuration import load_config
from agent.utils.utility import replace_multiple_whitespaces
from loguru import logger
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List, Optional, Tuple
from langchain.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from agent.backend.qdrant_service import get_db_connection

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

class DocumentService():
    def __init__(self):
        self.embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = get_db_connection(embedding_model=self.embedding_model)
        
    def embed_directory(self, dir: str) -> None:
        """Embeds the documents in the given directory in the llama2 database.

        This method uses the Directory Loader for PDFs and the PyPDFLoader to load the documents.
        The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

        Args:
            dir (str): The directory containing the PDFs to embed.

        Returns:
            None
        """
        logger.info(f"Logged directory:  {dir}")
        loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
        length_function = len
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "; ", "! ", "? ", "# "],
            chunk_size=500,
            chunk_overlap=50,
            length_function=length_function,
        )
        docs = loader.load_and_split(splitter)
        logger.info(f"Loaded {len(docs)} documents.")
        text_list = [doc.page_content for doc in docs]
        metadata_list = [doc.metadata for doc in docs]
        self.vector_store.add_texts(texts=text_list, metadatas=metadata_list)
        logger.info("SUCCESS: Texts embedded.")


    def embed_text(self, text: str, file_name: str, seperator: str) -> None:
        """Embeds the given text in the llama2 database.

        Args:
            text (str): The text to be embedded.


        Returns:
            None
        """
        # split the text at the seperator
        text_list: List = text.split(seperator)

        # check if first and last element are empty
        if not text_list[0]:
            text_list.pop(0)
        if not text_list[-1]:
            text_list.pop(-1)

        metadata = file_name
        # add _ and an incrementing number to the metadata
        metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]

        self.vector_store.add_texts(texts=text_list, metadatas=metadata_list)
        logger.info("SUCCESS: Text embedded.")


    def embed_folder(self, folder: str, seperator: str) -> None:
        """Embeds text files in the llama2 database.

        Args:
            folder (str): The folder containing the text files to embed.
            seperator (str): The seperator to use when splitting the text into chunks.

        Returns:
            None
        """
        # iterate over the files in the folder
        for file in os.listdir(folder):
            # check if the file is a .txt or .md file
            if not file.endswith((".txt", ".md")):
                continue

            # read the text from the file
            with open(os.path.join(folder, file)) as f:
                text = f.read()

            text_list: List = text.split(seperator)

            # check if first and last element are empty
            if not text_list[0]:
                text_list.pop(0)
            if not text_list[-1]:
                text_list.pop(-1)

            # ensure that the text is not empty
            if not text_list:
                raise ValueError("Text is empty.")

            logger.info(f"Loaded {len(text_list)} documents.")
            # get the name of the file
            metadata = os.path.splitext(file)[0]
            # add _ and an incrementing number to the metadata
            metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]
            self.vector_store.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Text embedded.")
        
    @load_config(location="config/db.yml")    
    def search_documents(self, cfg: DictConfig, query: str, amount: int, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
            query (str): The question for which documents should be searched.

        Returns:
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.
        """
        docs = self.vector_store.similarity_search_with_score(query, k=amount, score_threshold=.7)

        logger.debug(f"\nNumber of documents: {len(docs)}")

        if docs is not None and len(docs) > 0:
            for element in docs:
                document, score = element
                logger.debug(f"\n Document found with score: {score}")
                logger.debug(replace_multiple_whitespaces(document.page_content))


        logger.debug("SUCCESS: Documents found after similarity_search_with_score.")

        if os.environ.get('ACTIVATE_RERANKER') == "True":
            embedding = self.embedding_model
            filtered_docs = [t[0] for t in docs]
            retriever = self.vector_store.from_documents(filtered_docs, embedding, api_key=os.environ.get('QDRANT_API_KEY'), url=os.environ.get('QDRANT_URL'), collection_name="temp_ollama").as_retriever()

            rerank_compressor = CohereRerank(user_agent="my-app", model="rerank-multilingual-v2.0", top_n=3)
            splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", "; ", "! ", "? ", "# "],chunk_size=120, chunk_overlap=20)
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
            relevant_filter = EmbeddingsFilter(embeddings=embedding)
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[splitter, redundant_filter, relevant_filter, rerank_compressor]
            )
            compression_retriever1 = ContextualCompressionRetriever(base_compressor=rerank_compressor, base_retriever=retriever)
            compressed_docs = compression_retriever1.get_relevant_documents(query)

            for docu in compressed_docs:
                logger.info(f"Context after reranking: {replace_multiple_whitespaces(docu.page_content)}")

            #Delete the temporary qdrant collection which is used for the base retriever
            url = f"{os.environ.get('QDRANT_URL')}:{os.environ.get('QDRANT_PORT')}/collections/temp_ollama"
            headers = {"Content-Type": "application/json", "api-key": os.environ.get('QDRANT_API_KEY')}
            requests.delete(url, headers=headers)

            return compressed_docs
        else:
            #Logic for none-reranking needs to be implemented here
            filtered_docs = [t[0] for t in docs]
            return filtered_docs