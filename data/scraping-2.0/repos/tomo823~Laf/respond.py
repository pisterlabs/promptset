# llama-index==0.7.0
# File to respond to the query

import sys
import pinecone
import os
import openai
from llama_index.utils import truncate_text
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# API-keyの設定
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if pinecone_api_key is None:
    raise ValueError("Please set your PINECONE_API_KEY environment variable")

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
pinecone_index = pinecone.Index("keyword-search")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(
    similarity_top_k=2,
)


def get_query(query: str):
    response = query_engine.query(query)
    logging.debug(response)

    # reference for response
    reference = truncate_text(
        response.source_nodes[0].node.get_content().strip(), 350
    ).strip("...")
    logging.debug(reference)

    metadata: dict = response.metadata if response.metadata else {}
    logging.debug(metadata)
    urls = [metadata.get(key, {}).get("url", "") for key in metadata.keys()]

    return {
        "urls": urls,
        "responce": str(response),
        "reference": reference,
    }


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "三角形の内角の和は何度？"
    res = get_query(query)
    print(res["urls"])
