# from nnsplit import NNSplit
from config.openai import openai_config
from langchain.text_splitter import CharacterTextSplitter


class SentenceSplitConfig:
    embedding_chunk_size = openai_config.embedding_model_optimal_input_tokens
    completion_chunk_size = openai_config.completion_model_optimal_input_tokens


sentence_split = SentenceSplitConfig()


def get_embedding_text_splitter():
    return CharacterTextSplitter.from_tiktoken_encoder(chunk_size=sentence_split.embedding_chunk_size, chunk_overlap=0)

def get_completion_text_splitter():
    return CharacterTextSplitter.from_tiktoken_encoder(chunk_size=sentence_split.completion_chunk_size, chunk_overlap=0)
