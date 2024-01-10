from rank_bm25 import BM25Okapi
import string 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.autonotebook import tqdm
import numpy as np
import nltk
nltk.download('rslp')
nltk.download('stopwords')
from nltk.stem import RSLPStemmer, PorterStemmer
from langchain import OpenAI, LLMChain
import pickle
import os

class DocStore:
    def __init__(self, documents: list, language: str):
        self.documents = documents
        if language == "portugese":
          self.stop_words = nltk.corpus.stopwords.words('portuguese')
          self.stemmer = RSLPStemmer()
        elif language == "english":
          self.stop_words = nltk.corpus.stopwords.words()
          self.stemmer = PorterStemmer()
        self.text_splitter = RecursiveCharacterTextSplitter(# Set a really small chunk size, just to show.
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len)
        self.passages = self.chunk_creator()
        self.bm25 = BM25Okapi(self.tokenize_corpus())
        if not os.path.exists('./berri_files/'):
          os.mkdir("./berri_files/")
        with open("./berri_files/encoded_store.pkl", "wb") as f:
          pickle.dump(self.bm25, f)
        with open("./berri_files/original_store.pkl", "wb") as f:
          pickle.dump(self.passages, f)

    def chunk_creator(self):
      passages = []
      for document in self.documents:
        texts = self.text_splitter.split_text(document)
        passages.extend(texts) # create flat list of all chunks (loses context across documents)
      return passages
      
    def tokenize_corpus(self):
      tokenize_corpus = []
      for passage in tqdm(self.passages):
        tokenize_corpus.append(self.bm25_tokenizer(passage))
      return tokenize_corpus

    # We lower case our text and remove stop-words from indexing
    def bm25_tokenizer(self, text):
      tokenized_doc = []
      for token in text.lower().split():
        token = self.stemmer.stem(token) # stem the token
        if len(token) > 0 and token not in self.stop_words: # don't index on stop words 
            tokenized_doc.append(token)
      return tokenized_doc