from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import logging
# import faiss
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings

# Here we load in the data in the format that Notion exports it in.
# ps = list(Path("Notion_DB/").glob("**/*.md"))

# data = []
# sources = []
# for p in ps:
#     with open(p) as f:
#         data.append(f.read())
#     sources.append(p)

# # Here we split the documents, as needed, into smaller chunks.
# # We do this due to the context limits of the LLMs.
# text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
# docs = []
# metadatas = []
# for i, d in enumerate(data):
#     splits = text_splitter.split_text(d)
#     docs.extend(splits)
#     metadatas.extend([{"source": sources[i]}] * len(splits))

# print(docs[0])

# Here we create a vector store from the documents and save it to disk.
# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
# faiss.write_index(store.index, "docs.index")
# store.index = None
# with open("faiss_store.pkl", "wb") as f:
    # pickle.dump(store, f)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_string_into_chunks(text, chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        # if len(' '.join(current_chunk + [word])) <= chunk_size:
        if len(current_chunk) <= chunk_size:
            current_chunk.append(word)
        else:
            # print(len(current_chunk))
            chunks.append(' '.join(current_chunk))
            # print("\n\n")
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def chunk_docs(docs, chunk_size=512):
    """
    chunk the docs into smaller pieces where each piece has maximum of chunk_size words
    """
    chunked_docs = []
    for doc in docs:
        chunked_docs.extend(split_string_into_chunks(doc, chunk_size))  
    
    return chunked_docs

def read_md_docs(repo):
    """
    read the markdown docs and return them as a list of strings
    """
    logger.info(f"Reading docs from {repo}")
    docs_dir = Path(f"docs/{repo}")
    # docs_dir = Path("../docs/")
    doc_paths = docs_dir.rglob("*.md")
    text_docs = [] 
    for doc_p in doc_paths:
        with open(doc_p) as f:
            text_docs.append(f.read()) 

    return chunk_docs(text_docs, chunk_size=512)

    
if __name__ == "__main__":
    for doc in read_md_docs():
        print(len(doc.split()))