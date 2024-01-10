"""
Document Embedding and Uploading Script

This script processes a collection of text documents, generates embeddings using the OpenAI API, 
and uploads these embeddings to a Pinecone index for efficient similarity searching and retrieval.

Key Components:
- EmbeddingManager: Manages the process of generating embeddings with OpenAI and handling Pinecone operations.
- DocumentProcessor: Reads documents and associated metadata, preparing them for embedding generation.
- generate_and_upload_embeddings: Generates embeddings for a list of documents and uploads them to Pinecone.

Usage:
- Ensure all necessary libraries are installed and the .env file is properly configured.
- Set the 'links_csv_path' and 'folder_path' for your documents.
- Run the script. It will process the documents, generate embeddings, and upload them to Pinecone.

Note:
- The script requires a Pinecone API key and an OpenAI API key to be set in an .env file.
- Proper error handling and logging are implemented for robust operation.
"""

import os
import csv
from openai import OpenAI

client = OpenAI(api_key=self.openai_api_key)
import pinecone
from dotenv import load_dotenv
import logging
import json

# Configuration and initialization
load_dotenv()
logging.basicConfig(level=logging.INFO)

class EmbeddingManager:
    def __init__(self):
        """Initialize EmbeddingManager with Pinecone and OpenAI API keys."""
        self.pinecone_key = os.getenv("PINECONE_KEY")
        self.openai_api_key = os.getenv("OPENAI_KEY")
        
        pinecone.init(api_key=self.pinecone_key, environment='gcp-starter')
        self.index_name = "document-embeddings"
        self.ensure_index_exists()

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model)['data'][0]['embedding']

    def ensure_index_exists(self):
        if self.index_name not in pinecone.list_indexes():
            test_embedding_dim = len(self.get_embedding("Test text"))
            pinecone.create_index(self.index_name, dimension=test_embedding_dim)

    def save_vectors_to_file(self, vectors, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(vectors, file)
        logging.info(f"Vectors saved to {file_path}")

    def load_vectors_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                vectors = json.load(file)
            logging.info(f"Vectors loaded from {file_path}")
            return vectors
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return []

    def upload_embeddings(self, documents, batch_size=100):
        index = pinecone.Index(self.index_name)

        if not documents:
            logging.error("No documents provided for embedding upload.")
            return

        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]
            vectors = []
            for doc in batch:
                try:
                    if 'id' in doc and 'values' in doc and 'metadata' in doc:
                        # Use the existing metadata as is
                        vectors.append(doc)
                    else:
                        logging.warning(f"Document is not in the correct format: {doc}")
                except Exception as e:
                    logging.error(f"Error processing document: {doc}. Error: {e}")

            if vectors:
                try:
                    index.upsert(vectors=vectors)
                    logging.info(f"Batch of {len(vectors)} vectors uploaded.")
                except Exception as e:
                    logging.error(f"Error during batch upsert: {e}")
            else:
                logging.error("No valid vectors in batch for upload.")

class DocumentProcessor:
    def __init__(self, links_csv_path, folder_path):
        """Initialize DocumentProcessor with paths to links CSV and documents folder."""
        self.links_csv_path = links_csv_path
        self.folder_path = folder_path
        self.link_mapping = self.read_link_mapping()

    def read_link_mapping(self):
        link_mapping = {}
        try:
            with open(self.links_csv_path, mode='r', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)  # Skip the header
                for row in csvreader:
                    link_mapping[row[0]] = row[1]
            return link_mapping
        except FileNotFoundError:
            logging.error(f"File not found: {self.links_csv_path}")
            return {}

    def process_documents(self):
        document_metadata = []
        for root, dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.endswith('.txt'):
                    with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        prefix = filename.rpartition('_')[0]
                        link = self.link_mapping.get(prefix, "No link found")
                        full_text = "SOURCE LINK: " + link + " " + "CONTENT: " + content
                        document_metadata.append((full_text, link))
        return document_metadata

def generate_and_upload_embeddings(document_processor, embedding_manager, test_documents):
    """Generate embeddings for test documents and upload them to Pinecone."""
    embeddings = [embedding_manager.get_embedding(doc[0]) for doc in test_documents]
    vectors = [create_vector(i, embedding, test_documents[i]) for i, embedding in enumerate(embeddings)]

    # Save vectors to a file
    vectors_file_path = "vectors.json"
    embedding_manager.save_vectors_to_file(vectors, vectors_file_path)

    # Upload embeddings to Pinecone
    embedding_manager.upload_embeddings(vectors)

def create_vector(index, embedding, document):
    """Create a vector data structure from a document and its embedding."""
    text, link = document
    return {
        'id': str(index),
        'values': [float(value) for value in embedding],
        'metadata': {'text': text, 'link': link}
    }

if __name__ == "__main__":
    # Paths to your links CSV file and document folder
    links_csv_path = "06_Data/Capstone_Data/documentation_qa_datasets/VPC_Documentation_Links.csv"
    folder_path = "06_Data/Capstone_Data/chunks/"

    # Create instances of DocumentProcessor and EmbeddingManager
    document_processor = DocumentProcessor(links_csv_path, folder_path)
    embedding_manager = EmbeddingManager()

    # Process documents to get test documents
    documents = document_processor.process_documents()
    # test_documents = documents[:10]  # Adjust the slice as needed
    test_documents = documents # Embedding all data

    # Check if vectors file already exists
    vectors_file_path = "vectors_temp.json"
    vectors_to_upload = embedding_manager.load_vectors_from_file(vectors_file_path)

    if not vectors_to_upload:
        logging.info("No pre-saved vectors found. Generating new embeddings.")
        generate_and_upload_embeddings(document_processor, embedding_manager, test_documents)
    else:
        logging.info("Uploading pre-saved embeddings.")
        embedding_manager.upload_embeddings(vectors_to_upload)
