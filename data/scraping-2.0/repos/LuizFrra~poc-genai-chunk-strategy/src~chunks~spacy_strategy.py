import spacy
from langchain.text_splitter import SpacyTextSplitter
from chunks.chunk_strategy import ChunckStrategy

# see: https://spacy.io/models

nlp = spacy.load("xx_sent_ud_sm")

class SpacyStrategy(ChunckStrategy):

    def split(self, text):
        text_splitter = SpacyTextSplitter(chunk_size=512, max_length=len(text), pipeline='xx_sent_ud_sm')
        text_chunks = text_splitter.split_text(text)
        return text_chunks