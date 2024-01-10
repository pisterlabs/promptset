#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory="db",
        anonymized_telemetry=False
)

load_dotenv()

# Load environment variables
embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "db"

qa = None
db = None

def main():
    global  embeddings, db
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Initialize Chroma database
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)


if __name__ == "__main__":
    main()
    continue_chat=True
    while continue_chat:
        user_input=input("Query : ")
        print( ";".join(list(map(lambda data:data.page_content,db.similarity_search(user_input,4)))))



    
