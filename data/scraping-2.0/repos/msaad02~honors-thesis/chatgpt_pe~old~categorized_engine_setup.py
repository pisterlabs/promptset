import json
import pandas as pd

# Load in the JSON file
with open("/home/msaad/workspace/honors-thesis/data_collection/data/categorized_data.json", "r") as f:
    data = json.load(f)

import os

### SETUP RAW DATASTORE ###
def save_to_txt(data: str, filename: str, category: str, subcategory: str = None):
    base_file_path = "/home/msaad/workspace/honors-thesis/data_collection/data/categorized_datastore/raw_data"
    
    filename = filename.removeprefix("https://www2.brockport.edu/").replace("/", "-")

    path = f"{base_file_path}/{category}{(f'/{subcategory}' if subcategory != None else '')}/"

    print(path)  

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}{filename}.txt", "w") as f:
        f.write(data)

# Pull high level categories (i.e. about, academics, admissions, etc.)
for category, content in data.items():

    # Pull first level of category -- the "subcategory". 
    # This can point to either another category or data directly
    for subcategory, inner_content in content.items():

        # If its pointing to data directly, save it off
        if isinstance(inner_content, str):
            save_to_txt(inner_content, subcategory, category)
            continue
        
        # If it points to another category, pull the data from that category
        elif isinstance(inner_content, dict):
            for url, text in inner_content.items():
                save_to_txt(text, url, category, subcategory)
                continue



###SETUP CHROMADB DATASTORE###
def save_to_txt(data: str, filename: str, category: str, subcategory: str = None):
    base_file_path = "/home/msaad/workspace/honors-thesis/data_collection/data/categorized_datastore/chroma_data"
    
    filename = filename.removeprefix("https://www2.brockport.edu/").replace("/", "-")

    path = f"{base_file_path}/{category}{(f'/{subcategory}' if subcategory != None else '')}/"

    if not os.path.exists(path):
        os.makedirs(path)

    return path

# Pull high level categories (i.e. about, academics, admissions, etc.)
set_of_paths = set()
for category, content in data.items():

    # Pull first level of category -- the "subcategory". 
    # This can point to either another category or data directly
    for subcategory, inner_content in content.items():

        # If its pointing to data directly, save it off
        if isinstance(inner_content, str):
            set_of_paths.add(save_to_txt(inner_content, subcategory, category))
            continue
        
        # If it points to another category, pull the data from that category
        elif isinstance(inner_content, dict):
            for url, text in inner_content.items():
                set_of_paths.add(save_to_txt(text, url, category, subcategory))
                continue


# Now we have the relevant directories, and a set of paths to them.

# -----------------------------------------------------------------
# CREATE EMBEDDINGS FOR EACH CATEGORY

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

model = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-small-en",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
)

# Standardizing the content for each chunk
def standardize_string(input_string):
    # Step 1: Remove '\n' characters and replace them with periods
    standardized_string = input_string.replace('\n', ' ')

    # Step 2: Standardize the number of spaces
    standardized_string = re.sub(r'\s+', ' ', standardized_string)

    # Step 3: Remove non-alphanumeric characters at the start of the string
    standardized_string = re.sub(r'^[^a-zA-Z0-9]+', '', standardized_string)

    return standardized_string.strip()


def universal_make_vector_store(path: str, vector_store_dir: str):
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    # Create new 'texts' with some additional filters
    texts_cleaned = []

    # Iterate over texts page_content category with this cleaning method.
    for id in range(len(texts)):
        texts[id].page_content = standardize_string(texts[id].page_content)

        if len(texts[id].page_content) > 100:
            texts_cleaned.append(texts[id])

    vectordb = Chroma.from_documents(
        documents = texts_cleaned,
        embedding = model,
        persist_directory = vector_store_dir
    )

    vectordb.persist()


# Apply universal make vector store to all paths
from tqdm import tqdm

for path in tqdm(set_of_paths):
    universal_make_vector_store(path.replace("/chroma_data/", "/raw_data/"), path)

# NOTE: This takes a short amount of time (<1 minute) on a GPU equipped system.
# Otherwise, it takes a long time (~15-30 minutes) on a CPU equipped system.



print("Done!")

# At this point, you should be able to run categorized_search_engine.py and get results.