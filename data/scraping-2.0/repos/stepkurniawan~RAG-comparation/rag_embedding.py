from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model_id_retriever = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_id_generator = 'sentence-transformers/all-MiniLM-L6-v2'

def get_embed_model(embed_model_id):

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    print(f'device: {device}')
    return embed_model

def get_retriever_embeddings():
    embed_model = get_embed_model(embed_model_id_retriever)
    print(f'success load embed_model: {embed_model}')
    return embed_model


def get_generator_embeddings():
    embed_model = get_embed_model(embed_model_id_generator)
    print(f'success load embed_model: {embed_model}')
    return embed_model

def get_openai_embeddings():
    embed_model = OpenAIEmbeddings()
    print(f'success load embed_model: {embed_model}')
    return embed_model