import openai
from llama_index import Document, GPTVectorStoreIndex, SimpleDirectoryReader, EmptyIndex
import Globals
import os
import re

# Generates the vector index from the text files in the texts directory and persists it to the index directory

globals = Globals.Defaults()
index_path = globals.index_path
texts_path = globals.texts_path
openai.api_key = globals.open_api_key

if not os.path.exists(index_path):
    os.mkdir(index_path)

if not os.path.exists(texts_path):
    os.mkdir(texts_path)


def list_files(texts_path):
    # List to store the full paths
    full_paths = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(texts_path):
        for filename in filenames:
            # Combine the path to the directory with the filename
            full_path = os.path.join(texts_path, filename)
            full_paths.append(full_path)

    return full_paths


def parse_filename(filename):
    pattern = r'^(.*?)_(.*?)_(.*?).txt$'
    match = re.match(pattern, filename)

    # If a match was found, extract the groups
    if match:
        regulation_name = match.group(1)
        section_number = match.group(2)
        section_name = match.group(3)
        return regulation_name, section_number, section_name
    else:
        print('Not found!  ' + filename)
        return None


documents = []
text_files = [list_files(os.path.abspath(texts_path))]
doc_id = 0

for file_list in text_files:
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            text = ' '.join(f.readlines())

        file_name = os.path.basename(file)
        metadata = parse_filename(file_name)
        print(metadata)

        document = Document(
            text,
            doc_id=doc_id,
            extra_info={
                'regulation': metadata[0],
                'section_number': metadata[1],
                'section_name': metadata[2].replace('_', ' ')
            })
        documents.append(document)
        doc_id += 1

index = GPTVectorStoreIndex.from_documents(documents)

if len(index.storage_context.docstore.docs) == 0:
    print("The generated index is empty.")
else:
    # save vector index to persistant storage
    index.storage_context.persist(persist_dir=index_path)
