import os
import requests
import re

from bs4 import BeautifulSoup

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

from dotenv import load_dotenv

load_environment = load_dotenv()

dataset_path = os.environ.get('DEEPLAKE_DATASET_PATH')

embeddings =  OpenAIEmbeddings(model_name="text-embedding-ada-002")

def get_documentation_urls():
    return [
        '/docs/huggingface_hub/guides/overview',
        '/docs/huggingface_hub/guides/download',
    ]

def construct_full_url(base_url, relative_url):
    return base_url + relative_url

def scrape_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    text = soup.body.text.strip()

    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def scrape_all_content(base_url, relative_urls, filename):
    content = []
    for relative_url in relative_urls:
        full_url = construct_full_url(base_url, relative_url)
        scraped_content = scrape_page_content(full_url)
        content.append(scraped_content.rstrip('\n'))

    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n" %item)
    
    return content

def load_docs(root_dir, filename):
    docs = []
    try:
        loader = TextLoader(os.path.join(root_dir, filename), encoding='utf-8')

        docs.extend(loader.load_and_split())

    except Exception as e:
        pass

    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    return text_splitter.split_documents(docs)

def load_vectors_into_deeplake(dataset_path, source_chunks):
    deeplake_db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

    deeplakedb = deeplake_db.add_texts(source_chunks)

    return deeplakedb

def main():
    base_url = 'https://huggingface.co'

    filename = 'content.txt'

    root_dir = './'

    relative_urls = get_documentation_urls()

    content = scrape_all_content(base_url, relative_urls, filename)

    docs = load_docs(root_dir, filename)

    texts = split_docs(docs)

    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

    db.add_documents(texts)

if __name__ == '__main__':
    main()
