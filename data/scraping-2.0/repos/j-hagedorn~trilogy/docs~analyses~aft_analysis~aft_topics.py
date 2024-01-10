# import nltk; nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# Get df from R environment
df = r["aft"]

txt = df.text.values.tolist()
txt = [re.sub('\s+', ' ', sent) for sent in txt]
txt = [re.sub("\'", "", sent) for sent in txt]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

words = list(sent_to_words(txt))

# print(words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(
  words, min_count = 10, threshold = 50, 
  connector_words = gensim.models.phrases.ENGLISH_CONNECTOR_WORDS
) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(
  bigram[words], threshold=50,
  connector_words = gensim.models.phrases.ENGLISH_CONNECTOR_WORDS
)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# print(trigram_mod[bigram_mod[words[2]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization

from gensim.parsing.preprocessing import STOPWORDS
# added_stopwords = STOPWORDS.union(set(['likes', 'play']))

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in STOPWORDS] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

words = make_bigrams(words)
words = make_trigrams(words)
words = remove_stopwords(words)

# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# lemmas = lemmatization(words, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV'])
# print(lemmas[1])

# Create Dictionary
id2word = corpora.Dictionary(words)
texts = lemmas
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# print(corpus[:1])
# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

lda_model = gensim.models.ldamodel.LdaModel(
  corpus=corpus, id2word=id2word, num_topics=20, 
  random_state=100, update_every=1, chunksize=100,
  passes=10, alpha='auto', per_word_topics=True
)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

