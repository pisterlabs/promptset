from langchain.embeddings import OpenAIEmbeddings, GPT4AllEmbeddings, HuggingFaceEmbeddings
from enum import Enum

class Embeddings(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    GPT4ALL = "gpt4all"

def get_store_name(embeddings):
    if embeddings == Embeddings.HUGGINGFACE:
        return "vector_hf"
    elif embeddings == Embeddings.GPT4ALL:
        return "vector_g4"
    elif embeddings == Embeddings.OPENAI:
        return "vector_ai"

    raise Exception(f"Embedding '{embeddings}' not available")

def get_embeddings(embeddings):
    if embeddings == Embeddings.HUGGINGFACE:
        return huggingface_embeddings()
    elif embeddings == Embeddings.GPT4ALL:
        return gpt_4_all_embeddings()
    elif embeddings == Embeddings.OPENAI:
        return open_ai_embeddings()

    raise Exception(f"Embedding '{embeddings}' not available")

def open_ai_embeddings():
    return OpenAIEmbeddings()

def gpt_4_all_embeddings():
    return GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8)

def huggingface_embeddings():
    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': True}

    embeddingsModelPath = "BAAI/bge-large-en-v1.5"

    return HuggingFaceEmbeddings(
        model_name=embeddingsModelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs,          # Pass the model configuration options
        encode_kwargs=encode_kwargs         # Pass the encoding options
)