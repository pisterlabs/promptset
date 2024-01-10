from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from qdrant_client import QdrantClient

from langchain.tools import BaseTool
from langchain.docstore.document import Document
from openai import OpenAI

import httpx
import time

from conf.constants import *

from typing import Any    
    
import numpy as np

from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# ---

def document_from_scored_point(        
        scored_point: Any,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        return Document(
            page_content=scored_point.payload.get(content_payload_key),
            metadata=scored_point.payload.get(metadata_payload_key) or {},
        )

def query_qdrant(embedding, top_k=5, collection_name="rhaetor.github.io_components"):
    
    results = create_qdrant_client().search(
        collection_name=collection_name,
        query_vector=(embedding),
        with_payload=True,
        with_vectors=True,
        limit=top_k,
    )
    
    return results

def get_embedding(text, model="text-embedding-ada-002"):
   start = time.time()
   text = text.replace("\n", " ")
   resp = create_openai_client().embeddings.create(input = [text], model=model)
   print("Embedding ms: ", time.time() - start)
   return resp.data[0].embedding

def create_openai_client():
    client = OpenAI(
        timeout=httpx.Timeout(
            10.0, read=8.0, write=3.0, connect=3.0
            )
    )
    return client

def create_qdrant_client(): 
    client = QdrantClient(
       QDRANT_URL,
        api_key=QDRANT_KEY,
    )
    return client
    
def fetch_and_rerank(entities, collections):
    
    # compute the query embedding
    embedding = get_embedding(text=entities)

    # lookup across multiple vector store
    results = []    
    for name in collections:        
        intermittent_results = query_qdrant(embedding=embedding, collection_name=name, top_k=15)
        results.extend(intermittent_results)

        
    ## The MMR impl used with retriever(search_type='mmr')    
    embeddings = [result.vector for result in results]
    
    mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=5, lambda_mult=0.8
        )
    
    mmr_results = [
        (
            document_from_scored_point(
                scored_point=results[i], 
                content_payload_key="page_content", 
                metadata_payload_key="metadata"
            ),
            results[i].score,
        )
        for i in mmr_selected
    ]
    
    response_documents = []
    for i, article in enumerate(mmr_results):    
        doc = article[0]
        score = article[1]        
        print(str(round(score, 3)), ": ", doc.metadata["page_number"])
        response_documents.append(doc)
    
    return response_documents
        

class QuarkusReferenceTool(BaseTool):
    name = "search_quarkus_reference"
    description = "Useful when you need to answer questions about Camel Components used with Camel Quarkus. Input should be a list of camel components or the names of third-party systems."

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        response_content = []
        docs = fetch_and_rerank(query, ["quarkus_reference_2", "rhaetor.github.io_components_2"])
        response_content = [str(d) for d in docs]
        return ' '.join(response_content)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class CamelCoreTool(BaseTool):
    name = "search_camel_core"
    description = "Useful when you need to answer questions about enterprise integration patterns, languages or data formats in Camel, as well as the framework in general. Input should be a list of terms related to the core Camel framework"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        response_content = []
        docs = fetch_and_rerank(query, ["rhaetor.github.io_2", "rhaetor.github.io_components_2"])
        response_content = [str(d) for d in docs]
        return ' '.join(response_content)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


