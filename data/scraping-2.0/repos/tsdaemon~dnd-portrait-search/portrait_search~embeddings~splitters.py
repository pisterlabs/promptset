import abc

from langchain.text_splitter import RecursiveCharacterTextSplitter

from portrait_search.core.config import SplitterType


class TextSplitter(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def splitter_type(cls) -> SplitterType:
        raise NotImplementedError()

    @abc.abstractmethod
    def split(self, text: str) -> list[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def split_query(self, text: str) -> list[str]:
        raise NotImplementedError()


class LangChainRecursiveTextSplitter(TextSplitter, abc.ABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


class LangChainRecursiveTextSplitterChunk120Overlap60(LangChainRecursiveTextSplitter):
    def __init__(self) -> None:
        super().__init__(chunk_size=120, chunk_overlap=60)

        self.query_splitter = RecursiveCharacterTextSplitter(
            chunk_size=30,
            chunk_overlap=10,
            length_function=len,
            is_separator_regex=False,
        )

    @classmethod
    def splitter_type(cls) -> SplitterType:
        return SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60

    def split_query(self, text: str) -> list[str]:
        return self.query_splitter.split_text(text)


SPLITTERS: dict[SplitterType, type[TextSplitter]] = {
    t.splitter_type(): t for t in [LangChainRecursiveTextSplitterChunk120Overlap60]
}
