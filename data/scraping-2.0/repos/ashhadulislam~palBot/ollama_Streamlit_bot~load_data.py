'''
sudo snap remove curl
sudo apt install curl
curl https://ollama.ai/install.sh | sh

conda create -n ollamapy310 python=3.10

conda activate ollamapy310

 ollama pull  zephyr

 Zephyr is 4.1 GB

pip install chromadb
pip install langchain
pip install BeautifulSoup4
pip install gpt4all
pip install langchainhub
pip install pypdf
pip install chainlit


'''


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import PyPDFLoader

from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings  

import os

# we have to loop across all the folders


resource_folders=os.listdir("resources")



company_name="ashhadResearch"
DATA_PATH=f"data/{company_name}/"
DB_PATH = f"vectorstores/{company_name}/db/"




def create_vector_db():
    documents=[]
    processed_htmls=0
    processed_pdfs=0
    for f in os.listdir(DATA_PATH):
        print("File",f)
        try:
            if f.endswith(".pdf"):
                pdf_path = DATA_PATH + f
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                processed_pdfs+=1            
        except:
            print("issue with ",f)
            pass
    print("Processed ",processed_pdfs,"pdf files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(),persist_directory=DB_PATH)      
    
    vectorstore.persist()

if __name__=="__main__":
    create_vector_db()