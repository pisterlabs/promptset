import os
from pathlib import Path

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from utils import save_docs_to_jsonl

os.environ["OPENAI_API_KEY"] = ""

for file in Path("./data/永續報告書_上市_111").iterdir():
    try:
        loader = UnstructuredFileLoader(f"./data/永續報告書_上市_111/{file.name}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) 
        docs = loader.load_and_split(splitter)
        save_docs_to_jsonl(docs, f"data/documents/{file.stem}.jsonl")

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            docs, embeddings, persist_directory=f"./data/databases/{file.stem}"
        )
    except Exception as e:
        print(f"Error processing {file}: {e}")
