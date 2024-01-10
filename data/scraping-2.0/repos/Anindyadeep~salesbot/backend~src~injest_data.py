import os 
import requests
from tqdm import tqdm 
from bs4 import BeautifulSoup
from typing import List, Any, Optional 

# langchain imports 
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

class DataInjestion:
    """
    A very simple local data injestion pipeline class 
    # if we made more changes then these are the args in constructor during instanciating DataInjestion class
    # - doc_storage_folder_path 
    # - the name of the vector store
    # - the name of the embeddings
    """ 
    print("Initiating loading of embedding...")
    _embedding = SentenceTransformerEmbeddings()
    print("=> Embedding loaded successfully")

    def save_scrapped_urls(self, url_list : List[str], file_names : List[str], folder_path : Optional[str] = None) -> None:
        folder_path = os.path.join(project_root, "storage", "documents") if folder_path is None else folder_path
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print("=> Creating folder for documents as : ", folder_path)
        
        if len(os.listdir(folder_path))  == len(url_list): # skip if there are same number of files present inside the folder 
            print("=> Folder already has documents exiting...")
            return
        
        for url, file_name in tqdm(zip(url_list, file_names), total=len(url_list)):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            scrapped_text = soup.get_text()

            textfile_path = os.path.join(folder_path, f"{file_name}.txt")
            textfile = open(textfile_path, "w")
            textfile.write(scrapped_text)
            textfile.close()
        print("=> Done saving scrapped urls to text files")
    
    def load_docs(self, directory : Optional[str] = None) -> List[Any]:
        directory = os.path.join(project_root, "storage", "documents") if directory is None else directory
        loader = DirectoryLoader(directory)
        documents = loader.load()
        return documents
    
    def split_docs(self, documents : List[str], chunk_size : int = 1000, chunk_overlap : int = 0) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs 

    def get_vector_store(self, documents : List[Any]) -> Chroma:
        vector_store = Chroma.from_documents(
            documents, self._embedding,
            metadatas=[{"source": str(i)} for i in range(len(documents))]
        )
        return vector_store
    
    def initiate_data_pipeline(self, urls_list : List[str], file_names : List[str], folder_path : Optional[str] = None) -> Chroma:
        """
        Steps that it follows: (FIXME: There is some problem while using this pipeline while retrieving the context)
        -----------------------
        -> Scrapes all the data from the website 
        -> Saves the data to text files
        -> Loads the text files
        -> Splits the text files into chunks of 1000 characters with 20 characters overlap
        -> Creates a vector store from the splitted documents
        -> Returns the vector store
        """
        self.save_scrapped_urls(url_list=urls_list, file_names=file_names, folder_path=folder_path)
        documents = self.load_docs()
        documents_splitted = self.split_docs(documents)
        vector_store = self.get_vector_store(documents_splitted)
        return vector_store


    