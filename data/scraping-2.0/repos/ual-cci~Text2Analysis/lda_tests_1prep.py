# following the nicely written https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

DATASET = "_jobsv1_all"
DATASET = "_jobsv1_goodq"

import numpy as np
data = np.load("data/documents"+DATASET+".npz")['a']

import re
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# CUSTOM STOP WORDS?
stoplist = set(', . : / ( ) [ ] - _ ; * & ? ! – a b c d e t i p an us on 000 if it ll to as are then '
               'they our the you we s in if a m I x re to this at ref do and'.split())
stop_words.extend(stoplist)
stoplist = set('experience job ensure able working join key apply strong recruitment work team successful '
               'paid contact email role skills company day good high time required want right success'
               'ideal needs feel send yes no arisen arise title true'.split())
stop_words.extend(stoplist)
stoplist = set('work experience role application process contract interested touch'.split())
stop_words.extend(stoplist)

print(len(data[:1][0]))
pprint(data[:1])

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', doc) for doc in data]
# Remove new line characters
data = [re.sub('\s+', ' ', doc) for doc in data]
# Remove distracting single quotes
data = [re.sub("\'", "", doc) for doc in data]

print(len(data[:1][0]))
pprint(data[:1][0])

def sent_to_words(sentences):
    for sentence in sentences:
        # remove accent, remove too short and too long words
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(len(trigram_mod[bigram_mod[data_words[0]]]))
print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

### Data prepared, save
texts = data_lemmatized
np.savez_compressed("data/texts"+DATASET+".npz", a=texts)
