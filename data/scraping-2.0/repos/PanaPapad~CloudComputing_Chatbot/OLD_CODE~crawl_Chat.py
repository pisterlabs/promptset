from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncHtmlLoader
import pickle

# Initialize the NLP model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient()  # Specify the path to store the database
db = chroma_client.get_collection(name="my_collection")

def vectorize_text(text):
    return model.encode(text)

def crawlulr(start_url):
    url_contents = []
    response = requests.get(start_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')

        for link in links:
            href = link.get('href')
            full_url = urljoin(start_url, href)

            print(f"Crawling url: {full_url}")
            loader = AsyncHtmlLoader(full_url)
            docs = loader.load()
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)

            for index, doc in enumerate(docs_transformed):
                doc_text = str(doc)
                if not doc_text.strip():
                    continue

                url_contents.append(doc_text)

                # Add raw text document to ChromaDB
                db.add(documents=[doc_text], metadatas=[{"source": "fogify_docs", "url": full_url}], ids=[f"doc{index}"])
    else:
        print("Failed to retrieve the page")

    write_list(url_contents)

def read_list():
    with open('/data/urlcontent.pickle', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
    
def write_list(a_list):
    with open('/data/urlcontent.pickle', 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

urls = "https://ucy-linc-lab.github.io/fogify/"
crawlulr(urls)
print(len(read_list()))
