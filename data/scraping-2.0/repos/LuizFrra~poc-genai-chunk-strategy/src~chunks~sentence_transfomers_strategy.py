from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from chunks.chunk_strategy import ChunckStrategy


# max of 128 tokens per chunk
class SentenceTransformersStrategy(ChunckStrategy):
    def split(self, text):
        text_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=128, 
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )
        texts = text_splitter.split_text(text)
        return texts