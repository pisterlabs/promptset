import openai

import os
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"


def generate_sentence_vector(words):

    sentence = ' '.join(words)

    embedding_vector = OpenAIEmbeddings().embed_query(sentence)
    return embedding_vector

