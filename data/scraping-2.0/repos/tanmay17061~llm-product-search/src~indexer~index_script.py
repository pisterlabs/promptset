from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.fake import DeterministicFakeEmbedding
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import pandas as pd
import logging

def convert_product_features_to_text(features_str: str) -> str:
    return ", ".join(features_str.split("|"))

# Load the documents, embed each document and load them into the vector store.
df = pd.read_csv('data/wayfair_wands_product.csv', sep="\t")
df["text"] = "Product Name: " + df["product_name"] + " " + \
             "Product Class: " + df["product_class"] + " " + \
             "Product Description: " + df["product_description"] + " " + \
             "Product Features: " + df["product_features"].apply(convert_product_features_to_text)

raw_documents = DataFrameLoader(
                    data_frame = df,
                    page_content_column="text"
                ).load()[:5000]
logging.info(f"Indexing {len(raw_documents)} documents.")
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(raw_documents, DeterministicFakeEmbedding(size=100), persist_directory="data/db.chroma")