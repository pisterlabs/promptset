from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings.base import Embeddings
from langchain.schema import Document


def create_retriever(
    embeddings: Embeddings, splits: list[Document]
) -> VectorStoreRetriever:
    try:
        vectorstore = Chroma.from_documents(splits, embeddings)
    except (IndexError, ValueError) as e:
        raise Exception(f"Error creating vectorstore: {e}")
    retriever = vectorstore.as_retriever(search_type="mmr")
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10

    return retriever
