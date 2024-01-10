# Import necessary modules and libraries
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nucliadb_sdk import create_knowledge_box
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Function to split documents from a directory
def split_docs(directory, chunk_size=4300, chunk_overlap=300):
    """
    Load and split documents from the given directory.

    Args:
        directory (str): The path to the directory containing the documents.
        chunk_size (int): The size of document chunks to split.
        chunk_overlap (int): The overlap between document chunks.

    Returns:
        list: A list of document chunks.
    """
    # Initialize a directory loader
    loader = DirectoryLoader(directory)
    # Load documents from the directory
    documents = loader.load()

    # Initialize a text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split the documents into chunks
    return text_splitter.split_documents(documents)


# Main function to process documents and upload them to a knowledge base
def embed_and_upload(Knowledge_base_name):
    """
    Process documents from a directory and upload them to a knowledge base.

    Args:
        Knowledge_base_name (str): The name of the knowledge base.
    """
    # Define the directory path based on the knowledge base name
    directory = f"./myFiles/{Knowledge_base_name}"
    # Split the documents into chunks
    files = split_docs(directory)

    # Create or retrieve a knowledge base
    my_kb = create_knowledge_box(Knowledge_base_name)
    # Initialize a sentence embedding model
    model_bge = SentenceTransformer("BAAI/bge-base-en")

    # Process each document chunk
    for i, file in enumerate(tqdm(files, desc="Processing files")):
        # Encode the document content into vectors
        vectors = model_bge.encode([file.page_content])
        # Upload the document to the knowledge base
        my_kb.upload(
            key=f"mykey{i}",
            text=file.page_content,
            labels=[file.metadata["source"]],
            vectors={"bge": vectors[0]},
        )


if __name__ == "__main__":
    # Process documents for Knowledge Base 1
    Knowledge_base_name_1 = "knowledge_base1"
    embed_and_upload(Knowledge_base_name_1)

    # Process documents for Knowledge Base 2
    Knowledge_base_name_2 = "knowledge_base2"
    embed_and_upload(Knowledge_base_name_2)
