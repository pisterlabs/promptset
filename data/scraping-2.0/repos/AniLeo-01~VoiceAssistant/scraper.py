import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import re
import time

def get_documentation_urls():
    return [
    '/docs/huggingface_hub/guides/overview',
    '/docs/huggingface_hub/guides/download',
    '/docs/huggingface_hub/guides/upload',
    '/docs/huggingface_hub/guides/hf_file_system',
    '/docs/huggingface_hub/guides/repository',
    '/docs/huggingface_hub/guides/search',
    ]

def get_full_url(base_url, relative_url):
    return base_url+relative_url

def scrape_page_content(url):
    time.sleep(1)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.body.text.strip()
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_all_content(base_url, relative_urls, filename):
    content = []
    for relative_url in relative_urls:
        full_url = get_full_url(base_url, relative_url)
        scrapped_content = scrape_page_content(full_url)
        content.append(scrapped_content.rstrip('\n'))

    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n"%item)

    return content 