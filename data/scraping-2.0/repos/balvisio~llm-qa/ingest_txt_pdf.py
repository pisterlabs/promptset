"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
import pickle

import faiss
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# PATH_TO_DOCUMENTS = "Notion_DB/"
# INDEX_NAME = "notion.index"
# STORE_NAME = "notion.pkl"

PATH_TO_DOCUMENTS = "trainual/"
INDEX_NAME = "trainual+pdf.index"
STORE_NAME = "trainual+pdf.pkl"

# Here we load in the text data from trainual.
ps = list(Path(PATH_TO_DOCUMENTS).glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

# Load PDFs
loader = DirectoryLoader(PATH_TO_DOCUMENTS, glob="**/*.pdf", loader_cls=PyPDFLoader)

pdf_documents = loader.load()
pdf_docs = text_splitter.split_documents(pdf_documents)

# Here we create a vector store from the documents and save it to disk.
store_txt = FAISS.from_texts(docs, HuggingFaceEmbeddings(), metadatas=metadatas)
store_pdf = FAISS.from_documents(pdf_docs, HuggingFaceEmbeddings())

# merge two FAISS vectorstores
# https://python.langchain.com/docs/integrations/vectorstores/faiss#merging

store_txt.merge_from(store_pdf)
faiss.write_index(store_txt.index, INDEX_NAME)
store_txt.index = None
with open(STORE_NAME, "wb") as f:
    pickle.dump(store_txt, f)


