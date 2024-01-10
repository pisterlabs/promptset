from enum import Enum

from llama_index import OpenAIEmbedding

from configs import openai_api_key


class EmbedModelOption(Enum):
    """默认OpenAI的Embedding模型"""
    DEFAULT = OpenAIEmbedding(openai_api_key=openai_api_key)

if __name__ == '__main__':
    res = EmbedModelOption.DEFAULT.value.get_query_embedding('hi')
    print(res)