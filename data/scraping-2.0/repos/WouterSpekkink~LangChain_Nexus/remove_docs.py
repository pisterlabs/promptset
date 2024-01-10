from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import os
import sys
import constants
import openai
from typing import Optional
import numpy as np

# Define removal function
# Source: https://github.com/hwchase17/langchain/issues/2699
def remove(vectorstore: FAISS, docstore_ids: Optional[list[str]]):
    """
    Function to remove documents from the vectorstore.
    
    Parameters
    ----------
    vectorstore : FAISS
        The vectorstore to remove documents from.
    docstore_ids : Optional[List[str]]
        The list of docstore ids to remove. If None, all documents are removed.
    
    Returns
    -------
    n_removed : int
        The number of documents removed.
    n_total : int
        The total number of documents in the vectorstore.
    
    Raises
    ------
    ValueError
        If there are duplicate ids in the list of ids to remove.
    """
    if docstore_ids is None:
        vectorstore.docstore = {}
        vectorstore.index_to_docstore_id = {}
        n_removed = vectorstore.index.ntotal
        n_total = vectorstore.index.ntotal
        vectorstore.index.reset()
        return n_removed, n_total
    set_ids = set(docstore_ids)
    if len(set_ids) != len(docstore_ids):
        raise ValueError("Duplicate ids in list of ids to remove.")
    index_ids = [
        i_id
        for i_id, d_id in vectorstore.index_to_docstore_id.items()
        if d_id in docstore_ids
    ]
    n_removed = len(index_ids)
    n_total = vectorstore.index.ntotal
    vectorstore.index.remove_ids(np.array(index_ids, dtype=np.int64))
    for i_id, d_id in zip(index_ids, docstore_ids):
        del vectorstore.docstore._dict[
            d_id
        ]  # remove the document from the docstore

        del vectorstore.index_to_docstore_id[
            i_id
        ]  # remove the index to docstore id mapping
    vectorstore.index_to_docstore_id = {
        i: d_id
        for i, d_id in enumerate(vectorstore.index_to_docstore_id.values())
    }
    return n_removed, n_total

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load FAISS database
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./vectorstore/", embeddings)

docs = list(db.docstore._dict.items())

# SET THE NAME OF THE DOC TO REMOVE BY BIBTEX ID
doc_to_remove = ""
ids_to_remove = []

for doc in docs:
    if 'ID' in doc[1].metadata and doc[1].metadata['ID'] == to_remove:
        ids_to_remove.append(doc[0])

remove(db, ids_to_remove)

db.save_local('/.vectorstore/")


