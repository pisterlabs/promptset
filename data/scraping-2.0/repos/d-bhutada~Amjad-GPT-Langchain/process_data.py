# This script takes the Replit documentation and creates the embeddings
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

# Here we load in the Replit documentation data (Markdown files)
ps = list(Path("facts/").glob("**/*.*"))

# Read in the files and store them in a list
data = []
for p in ps:
  with open(p) as f:
    print("added " + f.name + " to data")
    data.append(f.read())

# Split the text into chunks of 2000 characters (because of LLM context limits)
text_splitter = CharacterTextSplitter(chunk_size=2000, separator="\n")
docs = []
for d in data:
  docs.extend(text_splitter.split_text(d))

# Create a vector store from the documents and save it locally
store = FAISS.from_texts(docs, OpenAIEmbeddings())
faiss.write_index(store.index, "amjad.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
  pickle.dump(store, f)

# After running this script, you should see two new files: faiss_store.pkl (vector store) and docs.index
# Now you are ready to run main.py
