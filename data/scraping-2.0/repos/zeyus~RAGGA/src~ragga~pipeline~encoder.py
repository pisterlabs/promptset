from types import MappingProxyType

from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.storage import InMemoryByteStore, LocalFileStore

from ragga.core.config import Config, Configurable


class Encoder(Configurable):
    """Encoder to create word embeddings"""
    _config_key = "encoder"

    _store: LocalFileStore | InMemoryByteStore | None = None

    _core_embeddings: HuggingFaceEmbeddings

    _embeddings: CacheBackedEmbeddings | HuggingFaceEmbeddings

    _default_config = MappingProxyType({
        "model_path": "sentence-transformers/all-MiniLM-l6-v2",
        "cache_embeddings": True,
        "persistent_cache": True,
        "cache_dir": "cache",
    })

    _default_model_kwargs = MappingProxyType({
        "device": "cuda",
    })

    _default_encode_kwargs = MappingProxyType({
        "normalize_embeddings": False,
    })

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)

        self._merge_default_kwargs(dict(self._default_model_kwargs), "model_kwargs")
        self._merge_default_kwargs(dict(self._default_encode_kwargs), "encode_kwargs")

        self._core_embeddings = HuggingFaceEmbeddings(
            model_name=self.config[self._config_key]["model_path"],
            model_kwargs=self.config[self._config_key]["model_kwargs"],
            encode_kwargs=self.config[self._config_key]["encode_kwargs"],
        )

        if self.config[self._config_key]["cache_embeddings"]:
            if self.config[self._config_key]["persistent_cache"]:
                self._store = LocalFileStore(self.config[self._config_key]["cache_dir"])
            else:
                self._store = InMemoryByteStore()
            self._embeddings = CacheBackedEmbeddings.from_bytes_store(
                self._core_embeddings,
                self._store,
                namespace=self._core_embeddings.model_name,
            )
        else:
            self._embeddings = self._core_embeddings

    @property
    def embeddings(self) -> CacheBackedEmbeddings | HuggingFaceEmbeddings:
        return self._embeddings
