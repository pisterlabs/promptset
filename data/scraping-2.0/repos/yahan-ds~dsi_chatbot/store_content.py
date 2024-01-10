from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def store_web_content()
    base_url = 'https://www.vanderbilt.edu/datascience/'

    reqs = requests.get(base_url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    urls = []

    for link in soup.find_all('a'):
        url = link.get('href')
        if url not in urls and base_url in url and 'php?' not in url and 'feed/' not in url:
        urls.append(url)

    print("total number of urls:", len(urls))


    loaders = UnstructuredURLLoader(urls = urls)
    data = loaders.load()

    # Split the Documents, Create the Embeddings and Create the VectorStore
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    print("total number of documents:", len(docs))

    # store the embedings
    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    with open("faiss_store_openai.pkl", 'wb') as f:
        pickle.dump(vectorStore_openAI, f)

    print("vector store successfully created in local directory.")

if __name__ == '__main__':
    store_web_content()