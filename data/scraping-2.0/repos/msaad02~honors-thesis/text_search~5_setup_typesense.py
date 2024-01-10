"""
This script creates the Typesense schema and loads data into it. Currently, it
is setup to support OpenAI embeddings with 'text-embedding-ada-002', but this can
be changed to other embeddings (see Typesense documentation for more information).

An intermediate step here is taking our chunked data from `3_prep_for_embeddings.py`
and putting it into jsonl files that Typesense can read. This is done in the
`create_jsonl_files` function, and is promptly deleted after the data is loaded.


# Typesense Text Search

This is a simple implementation of typesense. It includes a script to index the data
and a class designed to search the data given a question. We have an implementation
that includes the question classifier as well.

Typesense, since I chose not to use the cloud version, is a self-hosted search engine.
It is pretty easy to use, but does require some setup. I would recommend using the
docker installation, since it is the easiest to get up and running. Here is their
official guide to getting started: https://typesense.org/docs/guide/install-typesense.html

After installation, run the following command to start typesense. Just be sure to change
the directory you installed to, api key, or port as needed.

```bash
docker run -p 8108:8108 \
            -v/home/msaad/typesense-data:/data typesense/typesense:0.25.2 \
            --data-dir /data \
            --api-key=xyz \
            --enable-cors
```
"""

import typesense
import subprocess
import os
import sys
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset

typesense_host: str = "localhost"
typesense_port: str = "8108"
typesense_protocol: str = "http"
typesense_collection_name: str = "brockport_data_v1"
typesense_api_key: str = "xyz"
openai_api_key: str = os.getenv("OPENAI_API_KEY")

client = typesense.Client({
    'nodes': [{
        'host': typesense_host,
        'port': typesense_port,
        'protocol': typesense_protocol
    }],
    'api_key': typesense_api_key,
    'connection_timeout_seconds': 60,
    'retry_interval_seconds': 5
})

# Define index schema
index_name = typesense_collection_name
schema = {
  "name": index_name,
  "fields": [
    {"name": "url", "type": "string"},
    {"name": "main_category", "type": "string"},
    {"name": "sub_category", "type": "string"},
    {"name": "context", "type": "string"},
    {
      "name" : "embedding",
      "type" : "float[]",
      "embed": {
        "from": ["context"],
        "model_config": {
            "model_name": "openai/text-embedding-ada-002",
            "api_key": openai_api_key
        }
      }
    }
  ]
}

# Delete existing collection if it exists - but ask first!
try:
    client.collections[index_name].retrieve()

    input = input("Collection already exists! Enter 'y' to delete and recreate, or 'n' to exit >>> ")
    if input == 'y':
        client.collections[index_name].delete()
    else:
        print("Exiting...")
        exit()
except:
    pass

# Create collection
collection = client.collections.create(schema)

# ------ Most of this is from `3_prep_for_embeddings.py`, with slight modifications ------
# Attempt to load in chunked data
categorized_data = load_dataset("msaad02/categorized-data", split="train").to_pandas()
categorized_data = categorized_data.dropna(subset=['category']).reset_index(drop=True)

def filter_out_small_strings(input_string):
    "There are many 'sentences' that are really just titles for sections. These don't add much value to the embeddings so we'll filter them out."
    split_string = pd.Series(input_string.split("\n"))  # Split by newlines (which is how the data is formatted)
    tf_mask = [len(i) > 50 for i in split_string]       # Filter out groups that are less than 50 characters long
    out = split_string[tf_mask].str.strip()             # Apply filter and strip whitespace
    out = out.str.removeprefix("- ").str.removesuffix(".").to_list()    # Remove leading bullets and trailing periods
    return ". ".join(out)       # Join the list back into a string and add periods back in

cleaned = categorized_data['data'].apply(filter_out_small_strings)
cleaned = cleaned[cleaned.str.split(" ").str.len() > 100] # filter out data with less than 100 words

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[".", "?", "!"]
)

raw_chunks = cleaned.apply(text_splitter.split_text)

def clean_chunks(chunks):
    "RecursiveCharacterTextSplitter has weird behavior... It puts the punctuation from previous chunks into the next chunk. This cleans that up."
    for i in range(len(chunks)-1):
        chunks[i] += chunks[i+1][:1]
        chunks[i+1] = chunks[i+1][2:]
    return chunks

chunked_data = pd.Series(raw_chunks.copy()).apply(clean_chunks)
categorized_data['chunked_data'] = chunked_data
categorized_data = categorized_data.explode('chunked_data').loc[:, ['url', 'category', 'subcategory', 'chunked_data']].reset_index(drop=True)

# Filter to only chunks word count between 25 and 100. This is farily arbitrary, but we need consistent chunk sizes for the model.
word_count_per_row = categorized_data['chunked_data'].str.split().str.len()

categorized_data = categorized_data.loc[word_count_per_row < 100]
categorized_data = categorized_data.loc[word_count_per_row >  25].reset_index(drop=True)

# # This is the distribution of the number of words per chunk. Should look nice and pretty (~Normal).
# categorized_data['chunked_data'].str.split().str.len().hist(bins=30)

categorized_data.drop_duplicates(subset=['chunked_data'], inplace=True)
categorized_data.rename({'category': 'main_category', 'subcategory': 'sub_category', 'chunked_data': 'context'}, axis=1, inplace=True)

categorized_data.to_csv("documents.csv", index=False)
# ------ End of `3_prep_for_embeddings.py` ------

# Create jsonl file
if not subprocess.run(f'mlr --icsv --ojsonl cat documents.csv > documents.jsonl', check=True, shell=True):
    print("Failed to create jsonl file. Exiting.")
    print("Make sure you have Miller installed, see documentation and github https://typesense.org/docs/0.18.0/api/documents.html#import-a-csv-file and https://github.com/johnkerl/miller")
    sys.exit(1)

# Check if documents.jsonl exists
if not os.path.exists('documents.jsonl'):
    print("documents.jsonl does not exist. Exiting.")
    sys.exit(1)

print("jsonl file successfully created")

# Read and index JSONL file in batches
print("Indexing documents...")
jsonl_file_path = 'documents.jsonl'
with open(jsonl_file_path) as jsonl_file:
  client.collections[index_name].documents.import_(jsonl_file.read().encode('utf-8'), {'action': 'create'})

# Delete files
os.remove(jsonl_file_path)
os.remove('documents.csv')

# Done!
print("\n\nDone!")