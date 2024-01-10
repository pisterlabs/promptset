from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


def get_service_context():
    return ServiceContext.from_defaults(embed_model=embed_model)
