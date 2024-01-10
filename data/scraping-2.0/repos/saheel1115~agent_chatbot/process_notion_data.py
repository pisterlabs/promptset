"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
import os



OPEN_AI_API_KEY = os.environ["OPENAI_API_KEY"]

# Here we load in the data in the format that Notion exports it in.
# ps = list(Path("/Users/saheel/Downloads/notion_new/").glob("**/*.md"))

loader = TextLoader("/Users/saheel/Downloads/single.md")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n", chunk_overlap=0)
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

