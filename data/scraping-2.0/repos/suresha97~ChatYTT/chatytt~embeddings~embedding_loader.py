from langchain.embeddings.openai import OpenAIEmbeddings

_SUPPORTED_EMBEDDING_SOURCES = ["open-ai"]


class EmbeddingLoader:
    def __init__(self, source="open-ai"):
        self.source = _validate_source_type(source)

    def load_embeddings(self, text):
        return self.loader_type_map[self.source](text)

    def load_open_ai_embeddings(self, text):
        embeddings = OpenAIEmbeddings(model_name="ada")
        return embeddings.embed_query(text)

    @property
    def loader_type_map(self):
        return {"open-ai": self.load_open_ai_embeddings}


def _validate_source_type(source: str) -> None | str:
    if source not in _SUPPORTED_EMBEDDING_SOURCES:
        raise ValueError(
            f"This source for embeddings is not supported."
            f"Support types are: {_SUPPORTED_EMBEDDING_SOURCES}"
        )

    return source
