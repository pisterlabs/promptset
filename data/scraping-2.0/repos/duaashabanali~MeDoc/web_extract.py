import requests
# from bs4 import BeautifulSoup
import langchain
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredURLLoader


#Extract URLs from website
def parse_sitemap(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")

    urls = [element.text for element in soup.find_all("loc")]
    return urls


def web_extract(sitemap):
    sites = parse_sitemap(sitemap)

    sites_filtered = [url for url in sites if '/reference/' not in url and '?hl' not in url]


    #Extract pages from URLs
    loader = UnstructuredURLLoader(urls=sites_filtered)
    documents = loader.load()


    #Extract chunks from pages
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 100)

    document_chunks = text_splitter.split_documents(documents)

    print(f"Number documents {len(documents)}")
    print(f"Number chunks {len(document_chunks)}")

    document_chunks=[f"Context: {chunk.page_content} Source: {chunk.metadata['source']}" for chunk in document_chunks]

    data = pd.DataFrame(document_chunks,columns= ['Text'])
    return data