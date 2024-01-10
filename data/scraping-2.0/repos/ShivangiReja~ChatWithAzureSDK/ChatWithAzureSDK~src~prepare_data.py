import tiktoken
import os
import json
from bs4 import BeautifulSoup
from typing import List
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from tqdm.auto import tqdm

def load_documents(html_files_path):
    docs = parse_html_content(html_files_path)

    token_counts = [tiktoken_len(doc.page_content) for doc in docs]
    doc1_chunks = text_splitter.split_text(docs[0].page_content)
    doc2_chunks = text_splitter.split_text(docs[1].page_content)

    formatted_docs = process_documents(docs, html_files_path)

    output_directory = r'C:\Users\shreja\Demo\ChatWithAzureSDK\ChatWithAzureSDK\src\JsonDocument'
    write_documents_to_jsonl(formatted_docs, output_directory)
    return f"\n \n Token count Doc0 - {token_counts[0]} \n Token count Doc1 - {token_counts[1]} \n Doc0 chunks length {len(doc1_chunks)} chunk1 token count {tiktoken_len(doc1_chunks[0])} \n Doc1 chunks length {len(doc2_chunks)} chunk1 token count {tiktoken_len(doc2_chunks[0])} \n\n Formatted docs length: {len(formatted_docs)} \n\n Formatted doc 1: {formatted_docs[0]}"

# Custom class to hold document data
class Document:
    def __init__(self, file_path, page_content):
        self.file_path = file_path
        self.page_content = page_content

# Create the function to parse the html content
def parse_html_content(html_files_path):
    documents: List[Document] = []

    for root, _, files in os.walk(html_files_path):
        for file_name in files:
            if file_name.endswith(".html"):
                file_path = os.path.join(root, file_name)

                with open(file_path, "r", encoding="utf-8") as file:
                    html_content = file.read()

                soup = BeautifulSoup(html_content, "html.parser")

                # Get the whole text content without modifications
                extracted_text = soup.get_text()

                # Remove 3 or more empty lines
                extracted_text = "\n".join([t for t in extracted_text.split("\n\n\n") if t])
                document = Document(file_path, extracted_text)
                documents.append(document)

    return documents

# Create the length function to calculate token length for the given text
tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# To create these chunks we use the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

# Process a list of documents, split them into chunks, and format them as follows:
"""
Each processed document will be restructured into a format that looks like:
    
    [
        {
            "id": "abc-0",
            "content": "some important document text",
            "source": "https://langchain.readthedocs.io/en/latest/glossary.html"
        },
        {
            "id": "abc-1",
            "content": "the next chunk of important document text",
            "source": "https://langchain.readthedocs.io/en/latest/glossary.html"
        }
        ...
    ]
"""
def process_documents(docs, html_files_path):
    documents = []
    m = hashlib.sha256()

    for doc in tqdm(docs):
        url = doc.file_path.replace(html_files_path, 'https:')
        m.update(url.encode('utf-8'))
        uid = m.hexdigest()[:12]
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': f'{uid}-{i}',
                'content': chunk,
                'source': url
            })

    return documents

# Save them to a JSON lines (.json) file like so:
def write_documents_to_jsonl(documents, output_directory):
    output_file = os.path.join(output_directory, 'documents.jsonl')
    with open(output_file, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
