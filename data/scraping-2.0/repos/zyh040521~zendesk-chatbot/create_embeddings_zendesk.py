import pickle
import requests

from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=' ')
    return ' '.join(text.split())


if __name__ == '__main__':
    response = requests.get('your zendesk website') #please insert your zendesk api
    articles = response.json().get('articles', [])
    pages = [{"text": clean_html(article['body']), "source": article['html_url']} for article in articles]

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    store = FAISS.from_texts(docs,
                             OpenAIEmbeddings(),
                             metadatas=metadatas)
    with open("faiss_store_new.pkl", "wb") as f:
        pickle.dump(store, f)
