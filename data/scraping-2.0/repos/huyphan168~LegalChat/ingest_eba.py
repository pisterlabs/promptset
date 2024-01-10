"""Load data from excel files, clean up, split, ingest into FAISS."""
from __future__ import annotations
import pickle
from typing import Any, Dict, List, Optional
from eba_loader import EBALoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import json

def sorted_values(values: Dict[str, str]) -> List[Any]:
    """Return a list of values in dict sorted by key."""
    return [values[val] for val in sorted(values)]

def ingest_docs():
    """Get documents from EBA documents folder"""
    loader = EBALoader("eba_documents")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=100,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

def ingest_questions_db():
    """Get questions answer from database"""
    questions_db = []
    # Open the JSON file
    with open('questions_db.json', 'r') as f:
        # Read each line in the file
        for line in f:
            try:
                question = json.loads(line)
                questions_db.append({
                    "question": question['Question'],
                    "answer": question['EBA answer']
                })
            except:
                continue

    string_examples = [" ".join(sorted_values(eg)) for eg in questions_db]
    embeddings = OpenAIEmbeddings()
    sample_vectorstore = FAISS.from_texts(string_examples, embeddings, metadatas=questions_db)
    with open("sample_vectorstore.pkl", "wb") as f:
        pickle.dump(sample_vectorstore, f)

if __name__ == "__main__":
    ingest_docs()
    ingest_questions_db()
