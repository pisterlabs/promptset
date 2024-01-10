"""
Builds an index for locally stored pdfs using LangChain and Pinecone. 

It involves a simple adaptation of Arize's own documentation here: https://github.com/Arize-ai/phoenix/blob/main/tutorials/build_arize_docs_index_langchain_pinecone.py

To run, you must first create an account with Pinecone and create an index in the UI with the
appropriate embedding dimension (1536 if you are using text-embedding-ada-002 like this script). You
also need an OpenAI API key. This implementation relies on the fact that the Arize documentation is
written and hosted with Gitbook. If your documentation does not use Gitbook, you should use a
different document loader.
"""
import os
import argparse
import logging
import sys
from functools import partial
from typing import Dict, List, Optional
import json
import numpy as np
import openai
import pandas as pd
import pinecone  # type: ignore
import tiktoken
import pyarrow.parquet as pq
from langchain.docstore.document import Document
#from langchain.document_loaders import GitbookLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import LatexTextSplitter
from langchain.vectorstores import Pinecone
from tiktoken import Encoding
from typing_extensions import dataclass_transform
from tqdm import tqdm


def load_pdf_docs(d_path: str) -> List[Document]:
    """
    Loads documentation from three pdf docs.
    """
    print(f"dpath={d_path}")
    loader = PyPDFDirectoryLoader(d_path)

    print("loader = {}".format(loader))
    return loader.load()


def tiktoken_len(text: str, tokenizer: Encoding) -> int:
    """
    Returns the number of tokens in a text.
    """

    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def chunk_docs(documents: List[Document], embedding_model_name: str, chunk_type: str, chunk_size: int = 400, chunk_overlap: int = 20) -> List[Document]:
    """
    Chunks the documents by a specifief chunking strategy

    The original chunking strategy used in this function is from the following notebook and accompanying
    video:

    - https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/
      xx-langchain-chunking.ipynb
    - https://www.youtube.com/watch?v=eqOfr4AGLk8

    Since then we have added multiple chunk types that will be called using argparse, modified pinecone arguments to include specific subs
    """ 

    if chunk_type == 'RecursiveCharacterTextSplitter':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=partial(
                tiktoken_len, tokenizer=tiktoken.encoding_for_model(embedding_model_name)
            ),
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_documents(documents)

    elif chunk_type == 'MarkdownTextSplitter':
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return markdown_splitter.split_documents(documents)

    elif chunk_type == 'NLTKTextSplitter':
        NLTK_text_splitter = NLTKTextSplitter()
        return NLTK_text_splitter.split_text(documents)

    elif chunk_type == 'SpacyTextSplitter':
        Spacy_text_splitter = SpacyTextSplitter()
        return Spacy_text_splitter.split_documents(documents)

    elif chunk_type == 'LatexTextSplitter':
        latex_splitter = LatexTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return latex_splitter.split_documents(documents)        

def build_pinecone_index(
    documents: List[Document], embeddings: Embeddings, index_name: str, name_space: str
) -> None:
    """
    Builds a Pinecone index from a list of documents.
    """

    Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace=namespace)
    #print(f"embeddings post build of pinecone:{embeddings}")

def save_dataframe_to_parquet(dataframe: pd.DataFrame, save_path: str) -> None:
    """
    Saves a dataframe to parquet - ensures as we loop through combinations of namespaces that the parquet is added to instead of overwritten.
    """
    dataframe.to_parquet(save_path)  # Overwrites existing file
    print("File saved successfully.")


def combine_parquet(out_path: str) -> None:
    """
    Combines all parquet files in folder specified. Includes tests for identical columns
    """
    parq_files= [f for f in os.listdir(out_path) if f.endswith('.pq') and f != 'knowledge_db.pq']
    list_dfs=[]


    for file in parq_files:
        file_path=os.path.join(out_path,file)
        df=pd.read_parquet(file_path)
        list_dfs.append(df)
    
    combined_df=pd.concat(list_dfs, ignore_index=True, sort=False)

    combined_df.to_parquet(out_path+"/"+"knowledge_db.pq")

class ColumnConsistencyChecker:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.column_sets = []

    def check_column_consistency(self):
        for file_path in self.file_paths:
            df = pd.read_parquet(file_path)
            self.column_sets.append(set(df.columns))

        # Check if all sets in the list are equal
        return all(columns == self.column_sets[0] for columns in self.column_sets)

class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    Wrapper around OpenAIEmbeddings that stores the query and document embeddings in memory.
    """

    query_text_to_embedding: Dict[str, List[float]] = {}
    document_text_to_embedding: Dict[str, List[float]] = {}

    def embed_query(self, text: str) -> List[float]:
        embedding = super().embed_query(text)
        self.query_text_to_embedding[text] = embedding
        #print(f"embedding:{embedding}")
        return embedding

    def embed_documents(self, texts: List[str], chunk_size: Optional[int] = 0) -> List[List[float]]:
        embeddings = super().embed_documents(texts, chunk_size)
        for text, embedding in zip(texts, embeddings):
            self.document_text_to_embedding[text] = embedding
        #print(f"embeddings2:{embeddings}")
        return embeddings

    @property
    def query_embedding_dataframe(self) -> pd.DataFrame:
        return self._convert_text_to_embedding_map_to_dataframe(self.query_text_to_embedding)

    @property
    def document_embedding_dataframe(self) -> pd.DataFrame:
        return self._convert_text_to_embedding_map_to_dataframe(self.document_text_to_embedding)

    @staticmethod
    def _convert_text_to_embedding_map_to_dataframe(
        text_to_embedding: Dict[str, List[float]]
    ) -> pd.DataFrame:
        #print("text_to_embedding:", text_to_embedding)
        texts, embeddings = map(list, zip(*text_to_embedding.items()))
        embedding_arrays = [np.array(embedding) for embedding in embeddings]
        return pd.DataFrame.from_dict(
            {
                "text": texts,
                "text_vector": embedding_arrays,
            }
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    pinecone_api_key = os.getenv("YOUR_PINECONE_API_KEY")
    pinecone_index_name = os.getenv("YOUR_PINECONE_INDEX_NAME")
    pinecone_environment = os.getenv("YOUR_PINECONE_ENVIRONMENT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    output_parquet_path = os.getenv("OUTPUT_PARQUET_PATH")
    docs_path=os.getenv("DOC_PATH")
    print(f"docs_path={docs_path}")
    chunk_types_text=os.getenv("CHUNK_TYPES")
    chunk_sizes_text=os.getenv("CHUNK_SIZES")
    chunk_overlaps_text=os.getenv("CHUNK_OVERLAPS")

    chunk_types = json.loads(chunk_types_text)
    chunk_sizes = json.loads(chunk_sizes_text)
    chunk_overlaps = json.loads(chunk_overlaps_text)
    

    openai.api_key = openai_api_key
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    p_index = pinecone.Index(pinecone_index_name)
    
    embedding_model_name = "text-embedding-ada-002"
    documents = load_pdf_docs(docs_path)
    
    import time

    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            stat_dict = p_index.describe_index_stats()
            print(f"stat dict: {stat_dict}")
            break  # Break out of the loop if successful
        except Exception as e:
            print(f"Attempt {attempt} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        # If all attempts fail, handle the error or raise an exception
        print("All attempts failed. Exiting.")
    print(f"stat dict: {stat_dict}")
    namespace=""
    doc_iter= None
    i = 0
    old_df=None
    new_df=None
    
    for ct in chunk_types:
        if ct == 'LatexTextSplitter' :
            print("Ignoring LatexTextSplitter as documents are pdfs")
            continue
        elif ct == "RecursiveCharacterTextSplitter":
            for cs in chunk_sizes:
                for co in chunk_overlaps:
                    #create namespace label
                    namespace= f"{ct}_{cs}_{co}"
                    print(f"chunking for namespace {i}:{namespace}")

                    #delete namespace if it already exists
                    nsp_obj=stat_dict["namespaces"]
                    if 'namespaces' in stat_dict and namespace in stat_dict['namespaces']:
                        print(f"{namespace} namespace was deleted and replaced")
                        p_index.delete(delete_all=True, namespace=namespace)                
                    else:
                        print(f"{namespace} namespace did not exist; new one created")
                    
                    #chunk the documents into a list of documents
                    doc_iter = chunk_docs(documents, embedding_model_name, ct, cs, co) 
                    number_of_docs = len(doc_iter)
                    print(f"docs for {namespace} = {number_of_docs}")
                    
                    # new embeddings object
                    embeddings = OpenAIEmbeddingsWrapper(model=embedding_model_name)

                    #Build the pinecone index with the document embeddings dict 
                    build_pinecone_index(doc_iter, embeddings, pinecone_index_name, namespace)
                    
                    #Create df from embeddings
                    new_df = embeddings.document_embedding_dataframe
                    new_df["namespace"] = namespace
                    number_of_rows=new_df.shape[0]
                    print(f"number of embedding items ={number_of_rows}")
                    
                    #Save to parquet file in colab and make old df = new df for next iteration - we will pull things together into one file in the colab and push to gdrive there as well.
                    final_path = output_parquet_path+"/"+namespace+".pq"
                    save_dataframe_to_parquet(new_df, final_path)  
                    # delete or reset embeddings object- issues otherwise
                    embeddings = None
                    new_df = None
        # else:       
    print("Total number of namespaces:",i)
    parq_files = [f for f in os.listdir(output_parquet_path) if f.endswith('.pq')]
    file_paths = [os.path.join(output_parquet_path, file) for file in parq_files]
    checker = ColumnConsistencyChecker(file_paths)

    if checker.check_column_consistency():
        print("All Parquet files have the same column names.")
        combine_parquet(output_parquet_path)
        print("knowledge_db.pq built")
    else:
        print("Parquet files have different column names.")
    

