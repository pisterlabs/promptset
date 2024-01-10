"""
Functions copied from GCP examples:
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/examples/langchain-intro/intro_langchain_palm_api.ipynb
"""

import time
from typing import List
from pydantic import BaseModel
from pgvector.django import CosineDistance

from langchain.embeddings import VertexAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI


from chat.models import Message, User, Chat, DocumentChunk, Document


# Utility functions for Embeddings API with rate limiting
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]


# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
gcp_embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

text_llm = VertexAI(max_output_tokens=1024)
summarize_chain = load_summarize_chain(text_llm, chain_type="map_reduce")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def get_docs_chunks_by_embedding(request, query, max_distance=None):
    query_embedding = gcp_embeddings.embed_documents([query])[0]
    user_docs = Document.objects.filter(user=request.user)
    # documents_by_mean = user_docs.order_by(
    #     CosineDistance("mean_embedding", query_embedding)
    # )[:3]
    if max_distance is None:
        documents_by_summary = user_docs.order_by(
            CosineDistance("summary_embedding", query_embedding)
        )[:3]
        chunks_by_embedding = (
            DocumentChunk.objects.filter(document__in=user_docs)
            .order_by(CosineDistance("embedding", query_embedding))[:10]
            .prefetch_related("document")
        )
    else:
        documents_by_summary = user_docs.alias(
            distance=CosineDistance("summary_embedding", query_embedding)
        ).filter(distance__lt=max_distance)[:3]
        chunks_by_embedding = (
            DocumentChunk.objects.filter(document__in=user_docs)
            .alias(distance=CosineDistance("embedding", query_embedding))
            .filter(distance__lt=max_distance)
            .prefetch_related("document")
        )[:10]

    return documents_by_summary, chunks_by_embedding


def get_qa_response(query, documents, return_sources=True):
    if return_sources:
        chain = load_qa_with_sources_chain(text_llm, chain_type="stuff")
        response = chain(
            {"input_documents": documents, "question": query}, return_only_outputs=True
        )
        print(response)
        return response["output_text"]
    else:
        chain = load_qa_chain(text_llm, chain_type="stuff")
        response = chain.run(input_documents=documents, question=query)
        return response
