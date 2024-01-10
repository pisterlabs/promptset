# 缓存存储
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore

underlying_store = LocalFileStore("cache")
underlying_embeddings = OpenAIEmbeddings()

# 函数内部使用EncoderBackedStore做中间层
embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_embedding_cache=underlying_store,
    namespace=underlying_embeddings.model,
)

embeddings = embedder.embed_documents(['Hello', 'world'])
print(embeddings)
