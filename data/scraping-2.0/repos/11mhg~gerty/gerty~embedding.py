import os, sys
from langchain.storage import LocalFileStore
from langchain.embeddings import (
    LlamaCppEmbeddings,
    CacheBackedEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.embeddings import HuggingFaceBgeEmbeddings


def get_embeddings(
    model_path: str = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "models", "nous-hermes-llama-2-7b")
    ),
    n_ctx=2048,
    n_gpu_layers=25,
    n_batch=512,
    verbose=True,
):
    model_norm = HuggingFaceBgeEmbeddings(
        model_name = "BAAI/bge-large-en",
        model_kwargs = {'device': 'cuda' },
        encode_kwargs = { 'normalize_embeddings': True },
        query_instruction = "Represent this sentence for searching relevant passages: "
    )
    return model_norm
    #num_cores: int = os.cpu_count() or 1
    #model = LlamaCppEmbeddings(
    #    model_path=os.path.join(model_path, "ggml-model-q4_k.bin"),
    #    n_ctx=n_ctx,
    #    n_gpu_layers=n_gpu_layers,
    #    n_batch=n_batch,
    #    f16_kv=True,
    #    n_threads=num_cores // 2,
    #)
    #model.client.verbose = verbose

    #local_file_store: LocalFileStore = LocalFileStore(
    #    os.path.join(model_path, ".cache")
    #)
    #cache_embedder = CacheBackedEmbeddings.from_bytes_store(
    #    model, local_file_store, namespace="llama-cpp-embeddings"
    #)

    #return cache_embedder
