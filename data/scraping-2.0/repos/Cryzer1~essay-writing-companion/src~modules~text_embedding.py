from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

def generate_document_embeddings(extracted_texts):
    """Generate embeddings for the provided texts."""
    
    # Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    embeddings = OpenAIEmbeddings()
    
    # Split texts into chunks and generate embeddings
    chunks = text_splitter.split_documents(extracted_texts)
    chunk_embeddings = [(chunk, embeddings.generate(chunk['text'])) for chunk in chunks]
    
    return chunk_embeddings