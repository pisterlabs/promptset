import xmltodict
import requests
import langchain
import faiss
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, chroma
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
import getpass


def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

def get_all_sitemaps(mainsite = "https://www.idmod.org"):
    r = requests.get(mainsite + "/sitemap.xml")
    xml = r.text
    raw = xmltodict.parse(xml)

    sitemaps = []
    for info in raw['sitemapindex']['sitemap']:
        url = info['loc']
        if mainsite in url:
            sitemaps.append(url)
    return sitemaps

def get_all_site_contents(sitemaps):
    pages = []
    for sitemap in sitemaps:
        r = requests.get(sitemap)
        xml = r.text
        raw = xmltodict.parse(xml)
        if isinstance(raw['urlset']['url'], dict):
            url = raw['urlset']['url']['loc']
            pages.append({'text': extract_text_from(url), 'source': url})
        else:
            for info in raw['urlset']['url']:
                url = info['loc']
                pages.append({'text': extract_text_from(url), 'source': url})
    return pages

def split_text(pages):
    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")
    return docs, metadatas


if __name__ == "__main__":
    sitemaps= get_all_sitemaps()
    contents = get_all_site_contents(sitemaps)
    docs, metadatas = split_text(contents)

    with open('docs.pkl', 'wb') as f:
        pickle.dump(docs, f)
    with open('metadatas.pkl', 'wb') as f:
        pickle.dump(metadatas, f)

    docs = pickle.load(open('docs.pkl', 'rb'))
    metadatas = pickle.load(open('metadatas.pkl', 'rb'))

    if 'OPENAI_API_KEY' not in os.environ:
        os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter your OpenAI API Key:')
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    store.save_local("idmstore")
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)

    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)



