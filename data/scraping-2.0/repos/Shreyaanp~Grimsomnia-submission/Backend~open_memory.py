from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_setup import llm

def setup_memory():
    documents = SimpleDirectoryReader("./Data").load_data()
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index.as_query_engine(), embed_model, service_context

query_engine, embed_model, service_context = setup_memory()
