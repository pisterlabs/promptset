import os.path
from rag.index.markdown_to_text import markdown_to_text
from rag.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_RAG_COLLECTION_NAME, CONTENT_DIR
from qdrant_client import QdrantClient
from langchain.text_splitter import MarkdownTextSplitter
from rag.index.markdown import markdown_splitter

def upload(data):
    qdrant_client = QdrantClient(
        QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    collection_name = QDRANT_RAG_COLLECTION_NAME
    print(f"Storing data in the collection: {collection_name} file: {data['metadata'][0]['path']}")
    qdrant_client.add(
        collection_name=collection_name,
        documents=data['documents'],
        metadata=data['metadata'],
    )

def splitter(document_string, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(document_string):
        end = start + chunk_size
        chunk = document_string[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def longchain_splitter(document_string):
    markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = markdown_splitter.create_documents([document_string])
    docsChunks = []
    for doc in docs:
        docsChunks.append(doc.page_content)
    return docsChunks


def get_chunks(fulltext:str,chunk_length =500) -> list:
    text = fulltext

    chunks = []
    while len(text) > chunk_length:
        last_period_index = text[:chunk_length].rfind('.')
        if last_period_index == -1:
            last_period_index = chunk_length
        chunks.append(text[:last_period_index])
        text = text[last_period_index+1:]
    chunks.append(text)

    return chunks

def process_file(root_dir, file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        markdown_array = file.readlines()
        markdown_string='\n'.join(markdown_array)
        documents_string= markdown_to_text(markdown_string)
        # documents = splitter(documents_string, chunk_size = 2000,
        # chunk_overlap = 200)
        documents = longchain_splitter(markdown_string)
        # documents= get_chunks(documents_string)
        relative_path = os.path.relpath(file_path, root_dir)
        return {
            "metadata": [{"path":CONTENT_DIR+relative_path}]*len(documents),
            "documents": documents,
        }
    
def process_file_new(root_dir, file_path):

    chucks=markdown_splitter(
        path=file_path,
        max_chunk_size=1024,
        **{
            "merge_sections": True,
            "skip_first": False,
            "remove_images": True,
            "find_metadata": {"description": "description: "},
        },
    )
    relative_path = os.path.relpath(file_path, root_dir)
    documents = []
    for chunk in chucks:
        documents.append(chunk["text"])
    return {
        "metadata": [{"path":CONTENT_DIR+relative_path}]*len(chucks),
        "documents": documents,
    }
    

def explore_directory(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            if file_path.endswith('.md'):
                data=process_file_new(root_dir, file_path)
                upload(data)
    return "success"

def main():
    folder_path = os.getenv('QDRANT_PATH')+CONTENT_DIR
    res = explore_directory(folder_path)
    print(res)

if __name__ == "__main__":
    main()
