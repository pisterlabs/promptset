import nltk
from langchain.text_splitter import NLTKTextSplitter
from chunks.chunk_strategy import ChunckStrategy

nltk.download('punkt')

# this tokenizer use a unsupervised algorithm to split the text, it is trained in english
# https://www.nltk.org/_modules/nltk/tokenize/punkt.html
class NltkStrategy(ChunckStrategy):
    def split(self, text):
        text_splitter = NLTKTextSplitter(chunk_size=512)
        texts = text_splitter.split_text(text)
        return texts
    