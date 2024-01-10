"""Helper functions for Llama Index."""
from llama_index import GPTVectorStoreIndex, GPTListIndex
from llama_index import ServiceContext, LLMPredictor
from llama_index import LangchainEmbedding, Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

CHATGPT_KWARGS = {"temperature": 0, "model_name": "gpt-3.5-turbo"}


def init_index(docs, model="gpt-3.5-turbo", embedding_model=None, index_type="vector"):
    """Initialize each index with a different service context."""
    assert index_type in ("vector", "list")

    kwargs = CHATGPT_KWARGS
    kwargs["model_name"] = model

    llm = LLMPredictor(llm=ChatOpenAI(**kwargs))

    if embedding_model is None or embedding_model == "text-embedding-ada-002":
        service_context = ServiceContext.from_defaults(llm_predictor=llm)
    elif embedding_model == "sentencetransformers":
        embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm,
            embed_model=embed_model,
        )

    if index_type == "vector":
        index = GPTVectorStoreIndex.from_documents(
            docs, service_context=service_context
        )
    elif index_type == "list":
        index = GPTListIndex.from_documents(docs, service_context=service_context)

    return index


def retrieve_segment_of_text(query, text, model=None, embedding_model=None):
    """Retrieves a segment of text given a query and a text."""
    index = init_index(
        [Document(text=text)], model=model, embedding_model=embedding_model
    )
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    source = response.source_nodes[0].node.text
    return source


def summarize_documents(docs, query="Summarize in a few sentences.", model=None, embedding_model=None):
    """Given a list of documents, create a list of their summaries."""
    if model is None:
        model = "gpt-3.5-turbo"
    if embedding_model is None:
        embedding_model = "text-embedding-ada-002"

    summaries = []
    indexes = []
    for doc in docs:
        index = init_index(
            [doc], index_type="vector", model=model, embedding_model=embedding_model
        )
        query_engine = index.as_query_engine(response_mode="tree_summarize")
        response = query_engine.query(query)
        summaries.append(response.response)
        indexes.append(index)
    return {"summaries": summaries, "indexes": indexes}
