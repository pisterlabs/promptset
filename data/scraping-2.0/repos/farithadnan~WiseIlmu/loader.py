import os
import chromadb
from halo import Halo
from omegaconf import DictConfig
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from chromadb.errors import InvalidDimensionException
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

class Loader:
    def __init__(self):
        pass

    def load_documents(self, documents_dir: str):
        '''
        Method to load documents from a directory.

        Args:
            documents_dir (str): The directory containing the documents.
        Returns:
            A list of documents objects.
        '''
        spinner = Halo(text='Fetching Files...\n', spinner='dots')
        spinner.start()     

        documents = []
        for file in os.listdir(documents_dir):
            try:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(documents_dir, file)
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                elif file.endswith(".docx"):
                    docx_path = os.path.join(documents_dir, file)
                    loader =  Docx2txtLoader(docx_path)
                    documents.extend(loader.load())
                elif file.endswith(".txt"):
                    txt_path = os.path.join(documents_dir, file)
                    loader = TextLoader(txt_path)
                    documents.extend(loader.load())
                elif file.endswith(".csv"):
                    csv_path = os.path.join(documents_dir, file)
                    loader = CSVLoader(csv_path)
                    documents.extend(loader.load())
                else:
                    raise ValueError(f"Unsupported file format: {file}")
            except Exception as e:
                raise RuntimeError(f"Error while loading & splitting the documents: {e}")
        
        # Stop the spinner once the response is received
        spinner.stop()
        return documents
    

    def split_documents(self, documents: list, chunk_size=1000, chunk_overlap=20):
        '''
        Method to split documents into smaller chunks.

        Args:
            documents (list): The list of documents.
            chunk_size (int): The size of the chunks. 
            chunk_overlap (int): The overlap between the chunks.

        Returns:
            A list of chunked documents.
        '''
        try:
            # Create a loading spinner
            spinner = Halo(text='Splitting File Into Chunk...\n', spinner='dots')
            spinner.start()  

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(documents=documents)

            # Stop the spinner once the response is received
            spinner.stop()
            return documents
        except Exception as e:
            raise RuntimeError(f"Error while splitting the documents: {e}")
        

    def create_vector_db(self, documents, cfg: DictConfig):
        '''
        Method to get the vector database.

        Args:
            documents (list): The list of documents.
            cfg (DictConfig): The configuration file.
        
        Returns:
            The vector database.
        '''
        yellow = "\033[0;33m"

        print(f"{yellow}\n--------------------------------------------------")
        print(f"{yellow}           Configuring Vector Database                ")
        print(f"{yellow}--------------------------------------------------")

        spinner = Halo(text='\n', spinner='dots')
        spinner.start()  
        # Instantiate SentenceTransformerEmbeddings
        embeddings = SentenceTransformerEmbeddings(model_name=cfg.embeddings.model)
        spinner.stop()

        # Get vector from documents, if the dimension is invalid, delete the collection and try again
        try:
            vector_db = Chroma.from_documents(documents=documents,embedding=embeddings, persist_directory=cfg.vector_db_dir)
        except InvalidDimensionException:
            Chroma().delete_collection()
            vector_db = Chroma.from_documents(documents=documents,embedding=embeddings, persist_directory=cfg.vector_db_dir)

        print(f"{yellow}--------------------------------------------------\n")
        return vector_db
    

    def load_collection(self, vector_db_dir: str, collection_name="conversations"):
        '''
        Method to create or load a collection.

        Args:
            vector_db_dir (str): The directory containing the vector database.

        Return the collection.
        '''
        spinner = Halo(text='Configuring collection...\n', spinner='dots')
        spinner.start()  

        embedding_function = ONNXMiniLM_L6_V2()
        chroma_client = chromadb.PersistentClient(path=vector_db_dir)
        collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

        spinner.stop()
        return collection  