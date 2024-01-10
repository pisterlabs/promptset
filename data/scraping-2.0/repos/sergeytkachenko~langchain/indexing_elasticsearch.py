from elasticsearch import Elasticsearch
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.vectorstores.utils import DistanceStrategy
from docs_utils import get_academy_docs, get_demo_docs
from huggingface.embedding import HuggingfaceAwsApiEmbedding

def get_es_retriever(
        index_name,
        embedding_endpoint: str,
        embedding: Embeddings = None,
        hybrid_search: bool = False,
) -> ElasticsearchStore:
    if embedding is None:
        embedding = HuggingfaceAwsApiEmbedding(embedding_endpoint)
    client = Elasticsearch(
        hosts=["http://207.154.243.50:9200"],
        # cloud_id="academy:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGRiM2MzNWE3NDJjNDRhZGE4OGY4YjYwOTUxOWM0YTg4JGFhZThmMWM1NTdhYTRmMTZhYTY5YzlkNzg2YzRmYTM4",
        request_timeout=280,
        retry_on_timeout=True,
        max_retries=2,
        #basic_auth=("elastic", "bbOUCV7nDnK8r05DjtRz5zfj")
    )
    client.options(request_timeout=280).cluster.health( timeout="280s", master_timeout="280s", )
    # store = ElasticsearchStore.from_documents(
    #     documents=splits_documents,
    #     embedding=embedding,
    #     index_name="academy-from-md-latex-text",
    #     vector_query_field="embedding",
    #     strategy=ElasticsearchStore.ApproxRetrievalStrategy(
    #         hybrid=True,
    #     ),
    #     distance_strategy="COSINE",
    #     es_connection=client
    # )
    # store.client.indices.refresh(index="academy-from-md-latex-text")
    return ElasticsearchStore(
        embedding=embedding,
        index_name=index_name.lower(),
        vector_query_field="embedding",
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(
            hybrid=hybrid_search,
        ),
        distance_strategy=DistanceStrategy.COSINE,
        es_connection=client
    )
    # results = store.similarity_search(query)
    # print("start ES hybrid search by query: ", query)
    # print(list(map(lambda x: x.metadata["source"], results)))
    # print("\n")

def es_search(store: ElasticsearchStore, query: str):
    results = store.similarity_search_with_score(query=query)
    data = []
    for (doc, score) in results:
        # print(score, " ", doc.metadata["source"])
        data.append((score, doc.metadata["source"]))
    return data

def es_indexing(
        index_name: str,
        embedding_endpoint: str = None,
        embedding: Embeddings = None,
        load_files: bool = False,
        hybrid_search: bool = False
) -> ElasticsearchStore:
    # retriever1 = get_es_retriever(
    #     index_name=index_name,
    #     embedding=embedding,
    #     embedding_endpoint=embedding_endpoint,
    #     hybrid_search=hybrid_search,
    # )
    retriever2 = get_es_retriever(
        index_name=index_name+"demo",
        embedding=embedding,
        embedding_endpoint=embedding_endpoint,
        hybrid_search=hybrid_search,
    )
    if load_files:
        #docs = get_academy_docs()
        demo_docs = get_demo_docs()
        #retriever1.add_documents(docs)
        retriever2.add_documents(demo_docs)
    return retriever2

