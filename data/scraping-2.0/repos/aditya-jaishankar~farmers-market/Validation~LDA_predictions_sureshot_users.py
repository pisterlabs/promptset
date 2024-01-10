# %%
# This file is used to make predictions on the sureshot users dataset with the 
# LDA model trained on the followers dataset from before. We will output the
# feature vector matrix as a numpy array, to be used later to calculate a 
# similarity measure. We will have to do the same thing on the random users
# dataset.

# %%
# Imports

# General imports
import json
import glob
import pickle
import collections
import random
from tqdm import tqdm as tqdm
import config
import time
from pprint import pprint

# NLP imports
import nltk
from nltk.corpus import stopwords
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy

# Visualization imports
import matplotlib.pyplot as plt
import warnings

# Other imports
import pandas as pd
import numpy as np
import tweepy
from pprint import pprint

stop_words = stopwords.words('english')
stop_words.extend(['https', 'http'])
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# To check later if a words is in english or not
with open('./words_dictionary.json') as filehandle:
    words_dictionary = json.load(filehandle)
english_words = words_dictionary.keys()


# %%
# Defining functions


def get_augmented_feature_vectors(feature_vectors):
    """
    Takes in the feature vector list of list and augments it. gensim does not
    actually put a 0 for topics that have 0 probability so I need to manually
    add it in to build my feature vector. 
    input: accepts the feature vectors output by gensim. It is a list of 
    tuples - one list entry per document and tuple are (topic, probability)
    pairs.
    returns: Augmented feature vectors as list of list. Each list entry 
    corresponds to one document, with the i-th element in the inner list
    corresponding to the probability that the document was generated with 
    topic i.
    """
    augmented = []
    for i, vector in enumerate(feature_vectors): # each vector is a list of tuples
        topics = [tup[0] for tup in vector]
        for t in range(7):  # I finally settled on 7 topics
            if t not in topics:
                feature_vectors[i].append((t, 0))
        new_feature_vector = sorted(feature_vectors[i], key=lambda tup: tup[0])
        augmented.append([tup[1] for tup in new_feature_vector])
    return augmented


def get_docs(d):
    """
    Accepts a dictionary of the form {user: [cleaned word list]} and then
    returns the a list of lists, where each outer list is the word list for a 
    particular user. This forms our document corpus on which we make LDA
    predictions.
    """
    docs = []
    for user in d:
        text_list = d[user]
        docs.append(text_list)
    return docs


# %%
# Necessary file loads

# Load the tweet text
with open('./data/dataset_tweettext_sureshot_users.data', 'rb') as filehandle:
    dataset_tweettext_sureshot_users = pickle.load(filehandle)

# Load the NYC lda pretrained model
market_index = 1  # We will use the model for NYC
filename_model = './ldamodels/market' + str(market_index) + '/model.model'
lda_model = gensim.models.ldamodel.LdaModel.load(filename_model)
# load the corpus
filename_corpus = './ldamodels/market' + str(market_index) + '/corpus.corpus'

with open(filename_corpus, 'rb') as filehandle:
    corpus = pickle.load(filehandle)


# %%
# 

docs = get_docs(dataset_tweettext_sureshot_users)
old_id2word = lda_model.id2word

# Important! Use the old dictionary here to generate the new corpus. I have a
# total of about 30 feature vectors. 

new_corpus = [old_id2word.doc2bow(doc) for doc in docs]  
topics = lda_model.get_document_topics(new_corpus, per_word_topics=True)
feature_vectors = [doc_topics for doc_topics, word_topics, word_phis in topics]
augmented_feature_vectors = get_augmented_feature_vectors(feature_vectors)

with open ('./data/sureshot_augmented_feature_vectors.data', 'wb') as filehandle:
    pickle.dump(augmented_feature_vectors, filehandle, 
                protocol=pickle.HIGHEST_PROTOCOL)


# %%
