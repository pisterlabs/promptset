import openai
from PyPDF2 import PdfReader
import pinecone
import os
from config import OPENAI_API_KEY, PINECONE_API_KEY

# Initialize OpenAI and Pinecone clients
client = openai.OpenAI(api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
index_name = "qstar"

# Create Pinecone index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)
index = pinecone.Index(index_name)

# Function to extract text from PDF and its metadata
def extract_text_and_metadata_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        metadata = reader.metadata or {}
        title = metadata.get('/Title', 'Unknown Title')
        author = metadata.get('/Author', 'Unknown Author')
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text, title, author

# Function to split text into smaller chunks
def split_text(text, max_length=10000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:  # +1 for space
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to get embeddings from OpenAI
def get_openai_embeddings(text):
    chunks = split_text(text)
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embedding = response.data[0].embedding
        embeddings.append(embedding)

    return embeddings

# Function to upsert embeddings and metadata to Pinecone
def upsert_to_pinecone(base_id, embeddings, text_chunks, title, author):
    for i, (embedding, chunk) in enumerate(zip(embeddings, text_chunks)):
        if embedding is not None:
            unique_id = f"{base_id}-{i}"
            metadata = {
                "title": title,
                "author": author,
                "chunk_index": i,
                "text": chunk  # Include the actual text chunk as metadata
            }
            index.upsert(vectors=[(unique_id, embedding, metadata)])

# Function to process PDF and store embeddings with metadata
def process_pdf_and_store_embeddings(pdf_path):
    text, title, author = extract_text_and_metadata_from_pdf(pdf_path)
    text_chunks = split_text(text)
    embeddings = get_openai_embeddings(text)
    pdf_id = os.path.basename(pdf_path).split('.')[0]
    upsert_to_pinecone(pdf_id, embeddings, text_chunks, title, author)

# Example usage (add your own file path)
pdf_path = r'C:B Lab\Vectors\pdf\THE-SEMINAR-OF-JACQUES-LACAN-VI_desir_et_interp-.pdf'
process_pdf_and_store_embeddings(pdf_path)
