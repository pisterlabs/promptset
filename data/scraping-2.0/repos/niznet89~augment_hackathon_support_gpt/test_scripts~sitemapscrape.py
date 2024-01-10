import openai
import xml.etree.ElementTree as ET
import requests
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings


openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
print('openai_api_key', openai_api_key)

try:
    embeddings = OpenAIEmbeddings(
        max_retries=20, openai_api_key=openai_api_key)
except Exception as e:
    print(f"Failed to create OpenAIEmbeddings: {e}")
    exit()


def extract_urls_from_sitemap(sitemap_urls):
    urls = []
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url)
        except Exception as e:
            print(f"Failed to fetch sitemap: {sitemap_url} due to {e}")
            continue

        if response.status_code == 200:
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                print(f"Failed to parse XML from {sitemap_url} due to {e}")
                continue

            for url in root.findall('ns:url', namespace):
                loc = url.find('ns:loc', namespace)
                if loc is not None and loc.text:
                    urls.append(loc.text)
        else:
            print(
                f"Failed to fetch sitemap: {sitemap_url} with status code: {response.status_code}")

    return urls


sitemap_urls = ['https://docs.oceanprotocol.com/sitemap.xml',
                'https://oceanprotocol.com/sitemap.xml']

try:
    urls = extract_urls_from_sitemap(sitemap_urls)
    print('urls', urls)
    loader = WebBaseLoader(urls)
    docs = loader.load()
except Exception as e:
    print(f"Failed to load documents due to {e}")
    exit()

try:
    text_splitter = CharacterTextSplitter(chunk_size=15000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
except Exception as e:
    print(f"Failed to split documents due to {e}")
    exit()

try:
    db = DeepLake.from_documents(
        texts, embeddings, dataset_path="hub://tali/ocean_protocol_docs", overwrite=False)
except Exception as e:
    print(f"Failed to create DeepLake database due to {e}")
    exit()
