import argparse
import json
from typing import (
    Dict,
    List
)

import marqo
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import SimpleDirectoryReader


def load_documents(folder_path):
    source_chunks = []
    sources = SimpleDirectoryReader(
        input_dir=folder_path, recursive=True).load_data()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=200)
    for source in sources:
        for chunk in splitter.split_text(source.text):
            source_chunks.append(Document(page_content=chunk, metadata={
                "page_label": source.metadata.get("page_label"),
                "file_name": source.metadata.get("file_name"),
                "file_path": source.metadata.get("file_path"),
                "file_type": source.metadata.get("file_type")
            }))
    return source_chunks


def get_formatted_documents(documents: List[Document]):
    docs: List[Dict[str, str]] = []
    for d in documents:
        doc = {
            "text": d.page_content,
            "metadata": json.dumps(d.metadata) if d.metadata else json.dumps({}),
        }
        docs.append(doc)
    return docs


def chunk_list(document, batch_size):
    """Return a list of batch sized chunks from document."""
    return [document[i: i + batch_size] for i in range(0, len(document), batch_size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marqo_url',
                        type=str,
                        required=True,
                        help='Endpoint URL of marqo',
                        )
    parser.add_argument('--index_name',
                        type=str,
                        required=True,
                        help='Name of marqo index',
                        )
    parser.add_argument('--folder_path',
                        type=str,
                        required=True,
                        help='Path to the folder',
                        default="input_data"
                        )

    args = parser.parse_args()

    MARQO_URL = args.marqo_url
    MARQO_INDEX_NAME = args.index_name
    FOLDER_PATH = args.folder_path

    # Initialize Marqo instance
    marqo_client = marqo.Client(url=MARQO_URL)
    try:
        marqo_client.index(MARQO_INDEX_NAME).delete()
        print("Existing Index successfully deleted.")
    except:
        print("Index does not exist. Creating new index")

        index_settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 3,
                    "split_overlap": 1,
                    "split_method": "sentence"
                }
            }
        }

    marqo_client.create_index(
        MARQO_INDEX_NAME, settings_dict=index_settings)
    print(f"Index {MARQO_INDEX_NAME} created.")

    print("Loading documents...")
    documents = load_documents(FOLDER_PATH)

    print("Total Documents ===>", len(documents))

    f = open("indexed_documents.txt", "w")
    f.write(str(documents))
    f.close()

    print(f"Indexing documents...")
    formatted_documents = get_formatted_documents(documents)
    tensor_fields = ['text']
    _document_batch_size = 50
    chunks = list(chunk_list(formatted_documents, _document_batch_size))
    for chunk in chunks:
        marqo_client.index(MARQO_INDEX_NAME).add_documents(
            documents=chunk, client_batch_size=_document_batch_size, tensor_fields=tensor_fields)

    print("============ INDEX DONE =============")


if __name__ == "__main__":
    main()
    
# RUN
# python3 index_documents.py --marqo_url=http://0.0.0.0:8882 --index_name=sakhi_activity --folder_path=input_data
