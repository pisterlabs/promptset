from typing import TypeAlias, Sequence, Callable

from langchain.schema import Document
from langchain.schema.embeddings import Embeddings

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
PARAMS: TypeAlias = dict[str, "JSON"]
FACTORY: TypeAlias = str | PARAMS
CLASS: TypeAlias = str | PARAMS

EMBEDDING: TypeAlias = Sequence[float]

DOCUMENT_MEAN_EMBEDDING: TypeAlias = (
    str | Callable[[Embeddings, Sequence[Document]], EMBEDDING]
)
