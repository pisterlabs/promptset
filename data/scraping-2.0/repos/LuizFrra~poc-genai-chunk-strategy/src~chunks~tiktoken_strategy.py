from langchain.text_splitter import TokenTextSplitter
from chunks.chunk_strategy import ChunckStrategy

class TiktokenStrategy(ChunckStrategy):
    def split(self, text: str) -> list:
        text_splitter = TokenTextSplitter(
            model_name="text-embedding-ada-002",
            chunk_size = 512,
            chunk_overlap = 0,
        )
        text_chunks = text_splitter.split_text(text)
        return text_chunks
