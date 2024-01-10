import pprint
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

ps = list(Path("data/").glob("**/*.txt"))

# raw data
data = []

# sources that correspond to each data document 
sources = []

for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(str(p))

# use this to split text 
text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")

docs = []
metadatas = []
doc_count = 0
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    for split in splits:
        metadatas.extend([{"source": sources[i] + "<split>" + str(doc_count)}])
        doc_count += 1

store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")

store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

