# pyhton
from enum import Enum


# local


# langchain
# Embedding
# from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

# implementation


class EmbeddingType(Enum):
    OpenAI = 1
    HuggingFaceLocal = 2


class EmbeddingManager:

    @staticmethod
    # open ai embedding model: text-embedding-ada-002
    def get_embedding_model(api_key=None, emb_model_type=EmbeddingType.HuggingFaceLocal):
        if not api_key or emb_model_type == EmbeddingType.HuggingFaceLocal:
            return EmbeddingManager.get_hf_embedding_model()
        if api_key and emb_model_type == EmbeddingType.OpenAI:
            return OpenAIEmbeddings(openai_api_key=api_key)
        return None

    def get_hf_embedding_model():
        # MTEB English leaderboard  https://huggingface.co/spaces/mteb/leaderboard
        model_name = "hkunlp/instructor-large"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        iem = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return iem

    def get_embeddings(texts, api_key=None, emb_model_type="huggingface"):
        iem = EmbeddingManager.get_embedding_model(
            api_key=api_key, emb_model_type=emb_model_type)
        embeddings = iem.embed_query(texts)
        return embeddings
