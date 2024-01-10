import os
from dotenv import load_dotenv
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
import traceback
from tqdm import tqdm


def initialize_environment():
    """Load environment variables and return necessary configurations."""
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
    return openai_api_key, pinecone_api_key, pinecone_env

def load_and_split_pdf(pdf_url):
    """Load a PDF from the given URL and split it into pages with progress bar."""
    loader = PyPDFLoader(pdf_url)
    pages = loader.load_and_split()
    if not pages:
        print("Error: The text file is empty or faulty.")
        exit(1)
    return tqdm(pages, desc="Loading and splitting PDF")

def split_into_segments(pages):
    """Split pages into smaller text segments with progress bar."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return tqdm(text_splitter.split_documents(pages), desc="Splitting into segments")

def create_embeddings(pinecone_api_key, pinecone_env, split_docs):
    """Initialize Pinecone, create index, generate and save embeddings with progress bar."""
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index_name = "langchain-demo"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
    embeddings = OpenAIEmbeddings()
    try:
        # Wrapping split_docs with tqdm for progress bar
        docsearch = Pinecone.from_documents(tqdm(split_docs, desc="Generating embeddings"), embeddings, index_name=index_name)
        print(f"Data saved into Pinecone index: {index_name}")
        return docsearch
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

def main():
    openai_api_key, pinecone_api_key, pinecone_env = initialize_environment()
    # Replace with your PDF URL (yes, it is hardcoded still)
    pdf_url = "https://www.theccc.org.uk/wp-content/uploads/2023/09/230925-PF-MN-ZEV-Mandate-Response.pdf"
    pages = load_and_split_pdf(pdf_url)
    split_docs = split_into_segments(pages)
    docsearch = create_embeddings(pinecone_api_key, pinecone_env, split_docs)
    print(f'Number of pages loaded: {len(pages)}')
    print(f'Number of chunks after splitting: {len(split_docs)}')
    print(f'Result: {docsearch}')

# If the script is executed directly, call the main function
if __name__ == "__main__":
    main()