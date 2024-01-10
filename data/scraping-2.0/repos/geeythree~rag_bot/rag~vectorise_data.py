"""
Purpose:
This script generates vector database for the 
provided knowledge document
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter

import os
from dotenv import load_dotenv
load_dotenv()

if __name__ == '__main__':
    openai_api_key = os.getenv('OPENAI_API_KEY')
    persist_dir = os.getenv('PERSIST_DIR')
    llm_name = os.getenv('LLM_NAME')

    doc_path = r"rag\knowledge_document.txt"
    with open(doc_path, mode='r', encoding="utf8") as f:
        data = f.read()

    # print(data)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]   

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    docs = markdown_splitter.split_text(data)
    
    # Removing the Headers metadata, 
    # and replacing them with a new metadata key 'context'
    # which will represent the most relevant header level (1, 2, or 3)
    # corresponding to the page content (this will later help 
    # during prompting)
    for doc in docs:
        context = doc.metadata[list(doc.metadata.keys())[-1]]
        doc.metadata.update({"context": context})
        doc.metadata = {key: doc.metadata[key] for key in doc.metadata if not key.startswith("Header")}
        # print(doc.metadata, end='\n\n')


    embedding = HuggingFaceEmbeddings() #(model=llm_name, 
                                #show_progress_bar=True)


    vector_db = Chroma.from_documents(docs, 
                                    embedding=embedding, 
                                    k = 5,
                                    persist_directory=persist_dir)
    vector_db.persist()

    print(f"Vector Stored in {persist_dir}")

