"""
Kode di bawah digunakan untuk mengextract teks dari file pdf yang ada di folder document
"""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import re

from langchain.document_loaders import PyPDFLoader

paths = Path("document/").glob("**/*.pdf")
docs = []
for path in paths:
    path = str(path)
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    pages
    document = []
    for page in pages:
        content = page.page_content
        #Kode di bawah untuk menghilangkan header dan footer
        content = re.sub(r"PRES(.*)\n(.*)\n-.*-\n", " ", content)
        content = re.sub(r"FRES(.*)\n(.*)\n-.*-\n", " ", content)
        content = re.sub(r"PRESIDEN\nREPUBLIK INDONESIA\n.11-\n", " ", content)
        content = re.sub(r"PRESIDEN\nREPUBLIK INOONESIA\n_55_\n", " ", content)
        content = re.sub(r"PRESIDEN\nREPUBLIK INDONESIA\n.20 -\n", " ", content)
        content = re.sub(r"PRESIDEN\nREPUBLIK INDONESIA\n24-", " ", content)
        content = re.sub(r"PRESIOEN\nREPUBLIK INDONESIA\n-39\n", " ", content)
        content = re.sub(r"PRESIDEN\nREPUELIK INDONESIA\nL2-\n", " ", content)
        document.append({"content": content,"metadata":{'source':page.metadata['source']}})
    document

    #join content every 3 pages

    for i in range(0, len(pages), 3):
        docs.append({'content':" ".join([page['content'] for page in document][i:i+3]),
                    'metadata':{'source':document[i]['metadata']['source']}
                    })

# Here we create a vector store from the documents and save it to disk.
list_content = [doc['content'] for doc in docs]
list_metadata = [doc['metadata'] for doc in docs]
store = FAISS.from_texts(list_content, OpenAIEmbeddings(), metadatas=list_metadata)
faiss.write_index(store.index, "docs.index")
store.index = None

with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
