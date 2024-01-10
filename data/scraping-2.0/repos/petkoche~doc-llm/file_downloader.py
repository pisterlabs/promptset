import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


def text_loader(url, doc_name):
    res = requests.get(url)
    with open(doc_name, "w") as f:
        f.write(res.text)

    loader = TextLoader(f"./{doc_name}")
    documents = loader.load()

    return documents


# Text Splitter
def text_splitter(url, doc_name, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        url, doc_name, chunk_size, chunk_overlap)
    docs = text_splitter.split_documents(text_loader(url, doc_name))

    return docs
