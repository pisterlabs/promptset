# import textract
import shutil

from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS
from pathlib import Path
import faiss
from PyPDF2 import PdfReader
import pickle

def process_txt_folder(txt_input_folder_name, txt_folder_name):
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    data = []
    sources = []

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens
    )

    # Iterate over all files in the folder
    for filename in os.listdir(txt_input_folder_name):
        # Only process PDF files
        if filename.endswith(".txt"):
            # Full path to the file
            filepath = os.path.join(txt_input_folder_name, filename)

            print("Processing TXT:", filename)

            # Write the extracted text to a .txt file
            txt_filename = filename
            txt_filepath = os.path.join(txt_folder_name, txt_filename)
            path = Path(txt_filepath)
            if not path.is_file():
                print("Storing text:", txt_filename)
                shutil.copy(filepath, txt_filepath)

            # Read the .txt file
            with open(txt_filepath, 'r') as f:
                data.append(f.read())
            sources.append(filename)

        # Here we split the documents, as needed, into smaller chunks.
        # We do this due to the context limits of the LLMs.
        docs = []
        metadatas = []
        for i, d in enumerate(data):
            splits = text_splitter.split_text(d)
            docs.extend(splits)
            metadatas.extend([{"source": sources[i]}] * len(splits))

    # Return the array of chunks
    return docs, metadatas


# Store embeddings to vector db
all_chunks, metadatas = process_txt_folder("./text-input", "./text");

# Here we create a vector store from the documents and save it to disk.
db = FAISS.from_texts(all_chunks, OpenAIEmbeddings(), metadatas=metadatas)

faiss.write_index(db.index, "docs.index")
db.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(db, f)


