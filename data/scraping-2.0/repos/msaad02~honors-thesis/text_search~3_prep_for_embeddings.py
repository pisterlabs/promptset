"""
This script performs some cleaning on the data and chunking for on the data
before we create the embeddings. 
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset
import pandas as pd

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
    chunk_size=350,
    chunk_overlap=25,
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
categorized_data = categorized_data.dropna().reset_index(drop=True)
categorized_data = categorized_data.explode('chunked_data').loc[:, ['url', 'category', 'subcategory', 'chunked_data']].reset_index(drop=True)

# Filter to only chunks with less than 100 words
# Inspecting the data, mostly chunks with >100 words are just a bunch of bullet points of long lists about profs, programs, etc.
categorized_data = categorized_data[categorized_data['chunked_data'].str.split().str.len() < 100].reset_index(drop=True)

# # This is the distribution of the number of words per chunk. Should look nice and pretty (~Normal).
# categorized_data['chunked_data'].str.split().str.len().hist(bins=30)

try:
    categorized_data.to_csv("../data_collection/data/chunked_data.csv", index=False)
except:
    print("Failed to save chunked data to file. Continuing...")
    categorized_data.to_csv("chunked_data.csv", index=False)