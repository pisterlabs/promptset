import os
import re
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_transformers import Html2TextTransformer
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
urls = "https://ucy-linc-lab.github.io/fogify/"

def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )
    
def crawlulr(start_url):
    url_contents=[]
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=0,
                                               separators=[" ", ",", "\n"])
    # Send a GET request to the URL
    response = requests.get(start_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all 'a' tags (links) in the page
        links = soup.find_all('a')
        
        # Extract and print all URLs
        for link in links:
            try:
                # Get the href attribute of each 'a' tag
                href = link.get('href')

                # Join the URL if it's relative
                full_url = urljoin(start_url, href)

                print(f"Crawling url: {full_url}")
                loader = AsyncHtmlLoader(full_url)
                docs = loader.load()
                html2text = Html2TextTransformer()
                docs_transformed = html2text.transform_documents(docs)
                
                documents = text_splitter.split_documents(docs_transformed)
                for doc in documents:
                    url_contents.append(Document(doc))
            except:
                continue

    return url_contents

def clean_html(raw_html):
    # html2text = Html2TextTransformer()
    # docs_transformed = html2text.transform_documents([raw_html])
    # print(docs_transformed)
    # return docs_transformed
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()
vectorstore = Chroma("langchain_store", embeddings, persist_directory="./data/CHROMA_DB_3")
vectorstore.persist()

if is_docker():
    docs_dir = "./data/documentation"
else: 
    docs_dir = "./documentation"

# documents=[]
# Read all documents including subdirectories
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=300,
                                               separators=[" ", ",", "\n"])
html2text = Html2TextTransformer()
# documents=(crawlulr(urls))
# vectorstore.add_documents(documents)

for root, dirs, files in os.walk(docs_dir):
    for file in files:
        if file.endswith(".xml") or file.endswith(".md") or file.endswith(".yaml"):
            # Load the document, split it into chunks, embed each chunk and load it into the vector store.
            raw_document = TextLoader(os.path.join(root, file)).load()
            # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
           

            documents = text_splitter.split_documents(raw_document)
            vectorstore.add_documents(documents)
            
        if file.endswith('.html') :
            # file_path = os.path.join(root, file)
            # with open(file_path, 'r', encoding='utf-8') as file:
            raw_document = TextLoader(os.path.join(root, file)).load()
            documents = text_splitter.split_documents(raw_document)
            documents=html2text.transform_documents(documents)
            # print(documents)
            vectorstore.add_documents((documents))


        
            
vectorstore.persist()
print(vectorstore.similarity_search("What is Fogify?"))


