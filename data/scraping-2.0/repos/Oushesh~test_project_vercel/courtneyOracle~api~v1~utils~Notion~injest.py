##How to injest things from notion based on the git here: https://github.com/hwchase17/notion-qa

"""This is the logic for ingesting Notion Data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from dotenv import load_dotenv, dotenv_values
#loading the keys from python-dotenv
load_dotenv()
config = dotenv_values(".env.local") #or .env for shared code

def injest_notion():
    # Here we load in the data in the format that Notion exports it in.

    #need an automatic way to get the data directly from notion.
    ps = list(Path("Notion_DB/").glob("**/*.md"))
    print (len(ps))

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

    # Here we create a vector store from the documents and save it to disk.
    openai_api_key=""
    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    faiss.write_index(store.index, "docs.index")
    store.index = None
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
    return None

"""
#Testing the code locally
if __name__ == "__main__":
    injest_notion()
"""