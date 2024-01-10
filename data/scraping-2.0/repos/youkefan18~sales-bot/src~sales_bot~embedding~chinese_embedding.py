from threading import Lock

from langchain.embeddings import HuggingFaceEmbeddings


class ChineseEmbedding():
    """
        Singleton embedding instance for Chinese embedding.
        Model chosen according to MTEB [benchmarking](https://huggingface.co/spaces/mteb/leaderboard).

        [Issues] No sentence-transformers model found with name sentence_transformers\infgrad_stella-large-zh-v2. Creating a new one with MEAN pooling.
        [Solution](https://huggingface.co/GanymedeNil/text2vec-large-chinese/discussions/10)
    """
    #For thread safe singleton example see [here](https://refactoring.guru/design-patterns/singleton/python/example#example-1)
    _instance = None
    _lock: Lock = Lock()

    _model_name = "infgrad/stella-large-zh-v2"
    _embeddings = HuggingFaceEmbeddings(model_name=_model_name)

    @property
    def embeddings(self):
        return self._embeddings

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

if __name__ == "__main__":
    print(ChineseEmbedding().embeddings.embed_query("Test sentence for embedding"))