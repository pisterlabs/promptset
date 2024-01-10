from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings

import hashlib
import json
import pinecone

import os

from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE, PINECONE_ENVIRONMENT

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = PINECONE_INDEX_NAME
index = pinecone.Index(index)
namespace = PINECONE_NAMESPACE

num = 0

def hash_string(input_string, algorithm='sha256'):
    # Create a hash object with the specified algorithm
    hash_obj = hashlib.new(algorithm)

    # Encode the input string to bytes
    input_bytes = input_string.encode('utf-8')

    # Update the hash object with the bytes of the input string
    hash_obj.update(input_bytes)

    # Get the hexadecimal representation of the hash
    hashed_string = hash_obj.hexdigest()

    return hashed_string

def chunk_text(linked_pages, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    end = chunk_size

    if len(linked_pages) == 1:
        page1 = linked_pages[0]
        while end <= len(page1.page_content):
            chunk = page1.page_content[start:end]
            # The pdf-pages start at 1
            metadata = {"text": chunk, "pdf_page_number": page1.metadata["page"] + 1, "source": page1.metadata["source"]}
            chunks.append({"metadata": metadata})
            start += (chunk_size - chunk_overlap)
            end += (chunk_size - chunk_overlap)

        return chunks

    page1 = linked_pages[0]
    page2 = linked_pages[1] if len(linked_pages) == 2 else None

    # First pdf page only
    while end <= len(page1.page_content):
        chunk = page1.page_content[start:end]
        metadata = {"text": chunk, "pdf_page_number": page1.metadata["page"] + 1, "source": page1.metadata["source"]}
        chunks.append({"metadata": metadata})
        start += (chunk_size - chunk_overlap)
        end += (chunk_size - chunk_overlap)

    # First pdf page and second pdf page
    if page2:
        remaining = chunk_size - (len(page1.page_content) - start)
        chunk = page1.page_content[start:] + page2.page_content[:remaining]
        metadata = {"text": chunk, "pdf_page_number": page1.metadata["page"] + 1, "source": page1.metadata["source"]}
        chunks.append({"metadata": metadata})

        # Second pdf page only
        start = remaining
        end = start + chunk_size
        while end <= len(page2.page_content):
            chunk = page2.page_content[start:end]
            metadata = {"text": chunk, "pdf_page_number": page2.metadata["page"] + 1, "source": page2.metadata["source"]}
            chunks.append({"metadata": metadata})
            start += (chunk_size - chunk_overlap)
            end += (chunk_size - chunk_overlap)

    return chunks

def write_chunks_to_file(chunks, output_file):
    with open(output_file, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk))
            f.write('\n')

def process_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    linked_pages = [] # Changed this line
    result = []

    for i, _ in enumerate(pages):
        if(len(pages) == 1):
            linked_pages.append([pages[i]]) # Changed this line
            break
        if(i == len(pages) - 1):
          break
        linked_pages.append([pages[i], pages[i + 1]]) # Changed this line

    for pages in linked_pages:
        result += chunk_text(pages, chunk_size, chunk_overlap)

    return result

def removeFromPinecone(chunks):
    global namespace
    global num
    ids = []
    for i, chunk in enumerate(chunks):
        text = chunk['metadata']['text']
        ids.append(hash_string(text)) # Changed this line

        if(i % 100 == 0):
            index.delete(ids, namespace=namespace)
            num += len(ids)
            ids = []
              
    if(ids != []):
        index.delete(ids, namespace=namespace)
        num += len(ids)

# specify the folder path
def list_files(folder_path):
    file_names = []

    if os.path.isfile(folder_path):
        file_names.append(folder_path)
    elif os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.doc') or filename.endswith('.docx') or filename.endswith('.pdf'):
                file_names.append(os.path.join(folder_path, filename))

    return file_names

folder_path = "docs"
file_names = list_files(folder_path)

# Delete all documents
for file_path in file_names:
    print("Delete chunk ", file_path)
    chunked_pages = process_pdf(file_path)
    removeFromPinecone(chunked_pages)

