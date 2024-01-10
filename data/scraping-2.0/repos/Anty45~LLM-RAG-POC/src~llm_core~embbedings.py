from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding


def create_embbeding(model_config):
    return LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=model_config["embbeding"],
        )
    )
