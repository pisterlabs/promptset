import os
import wandb
import faiss
import pickle
import json
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter

PROJECT = "wandb_docs_bot"

run = wandb.init(project=PROJECT)

def find_md_files(directory):
    md_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                md_files.append(file_path)
    return md_files

def load_documents(files):
    docs = []
    for file in files:
        fname = file.split('/')[-1]
        loader = UnstructuredMarkdownLoader(file)
        markdown_splitter = MarkdownTextSplitter(chunk_size=2048, chunk_overlap=128)
        markdown_docs = loader.load()
        markdown_docs = [x.page_content for x in markdown_docs]
        chunks = markdown_splitter.create_documents(markdown_docs)
        for chunk in chunks: chunk.metadata["source"] = fname # need to add the source to doc
        docs.extend(chunks)
    return docs

def create_and_save_index(documents):
    store = FAISS.from_documents(documents,OpenAIEmbeddings())
    artifact = wandb.Artifact("faiss_store", type="search_index")
    faiss.write_index(store.index, "docs.index")
    artifact.add_file("docs.index")
    store.index = None
    with artifact.new_file("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
    wandb.log_artifact(artifact, "docs_index", type="embeddings_index")
    return store

def main():
    files = find_md_files('../docodile/docs/')
    documents = load_documents(files)    
    store = create_and_save_index(documents)
    
if __name__ == "__main__":
    main()