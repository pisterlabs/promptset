# file_extract.py
import os
import argparse
import getpass
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
def extract_content(file_path):
    print("Excuting content extraction...")
    raw_documents = TextLoader(file_path).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    vectorstore = Chroma.from_documents(
        documents, OpenAIEmbeddings(), persist_directory="vector_dir"
    )

def main():
    print("Excuting file_extract.py...")
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter your API key: ")

    parser = argparse.ArgumentParser(
        description="Process a text file and generate relevant content."
    )
    parser.add_argument("file_path", help="Path to the input text file")

    args = parser.parse_args()

    # Extract content
    extract_content(args.file_path)
    return "Extraction complete."

if __name__ == "__main__":
    main()
