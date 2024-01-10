# The MIT License (MIT)
# Copyright © 2023 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
import requests
from bs4 import BeautifulSoup

import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

from typing import List, Sequence
from template.base.validator import BaseValidatorNeuron

# SETUP VECTOR DATABASE for simple example of how a miner could work with incoming data
chroma_client = chromadb.Client()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap  = 30,
    length_function = len,
    is_separator_regex = False,
)

# receive the validator request, put the data into vectorDB, 
# ... query VDB for prompt-relevant context, build citations, return 
def get_relevant_context_and_citations_from_synapse(synapse: BaseValidatorNeuron) -> List:
    urls = synapse.urls
    prompt = synapse.prompt
    datas = synapse.datas
    if not urls and not datas:
        # if urls is empty and datas is empty, we don't have anything to do wrt context/citations
        return []

    collection = __index_data_from_datas(datas)
    results = collection.query(query_texts=[prompt],n_results=4)

    citations = []
    context = ""
    for i,d in enumerate(results['documents'][0]):
        context += d
        citations.append({'context': d, 'source':results['metadatas'][0][i]['source']})

    # after getting the results, we can delete the collection
    chroma_client.delete_collection(collection.name)

    return [context, citations]

# on the fly collection creation
def __index_data_from_datas(datas: List[dict]) -> Sequence:
    collection = chroma_client.create_collection(name=__generate_collection_name())
    for x, data in enumerate(datas):
        source = data['source']
        context = data['context']
        chunks = text_splitter.create_documents([context])
        docs = [c.page_content for c in chunks]
        collection.add(documents=docs, 
                       ids=[f"id_data_{x}_{i}" for i in range(len(docs))],
                       metadatas=[{"source": source} for _ in range(len(docs))])

    return collection

# random hash for colleciton name
def __generate_collection_name() -> str:
    h = random.getrandbits(128)
    return f'bitqna.collection.{h}'
