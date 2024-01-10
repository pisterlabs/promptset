import os
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


def select_embeddings():
    embeddings = None
    if os.environ.get('EMBEDDING_METHOD', 'openai') == 'huggingface':
        embeddings = HuggingFaceEmbeddings()
    elif os.environ.get('EMBEDDING_METHOD', 'openai') == 'openai':
        embeddings = OpenAIEmbeddings(model=os.environ.get(
            'OPENAI_EMBEDDINGS_MODEL', 'text-embedding-ada-002'))

    return embeddings
