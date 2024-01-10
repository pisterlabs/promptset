import random
import torch
import math
import pickle
import spacy
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
spacy_tokenizer = spacy.load("en_core_web_sm")
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
path_to_mallet_binary = "Mallet/bin/mallet"
from transformers import BertTokenizer, BertForNextSentencePrediction
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import logging
import scipy
from scipy import spatial, special
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): # Taken from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    """https://spacy.io/api/annotation"""
    texts_out = []
    for text in texts:
        tokens = nlp(text)
        lemmatized_tokens = [token.lemma_ for token in tokens if token.pos_ in allowed_postags]
        destopped_lemmatized_lowercased_tokens = [token.lower() for token in lemmatized_tokens if token.lower() not in stop_words]
        texts_out.append(destopped_lemmatized_lowercased_tokens)
    return texts_out


def KL_divergence(x, y):
    return sum(special.kl_div(x, y))


JS_divergence = spatial.distance.jensenshannon 


def compute_ts_joint(data, dataset, k, pickled_corpus=False, pickled_model=False):
	print("Computing Topic Similarity - Joint Topic Model with {} topics".format(k))

	corpus_f = 'dataset={}_jointTM.pickle'.format(dataset)
	model_f = 'dataset={}_k={}_jointTM.pickle'.format(dataset, k)

	stop_words = stopwords.words('english')
	stop_words.extend(['from', 'subject', 're', 'edu', 'use', '>', '<', '/t', '-lrb-', '-rrb-', '/n', '\n'])

	if pickled_corpus:
		corpus, id2word = pickle.load(open(corpus_f, 'rb'))
	else:
		print("Constructing joint topic model's corpus")
		document_data = [example['text'] for example in data]
		summary_data = [example['summary'] for example in data]
		input_data = document_data + summary_data
		input_data = lemmatization(input_data, stop_words)
		
		id2word = corpora.Dictionary(input_data)
		corpus = [id2word.doc2bow(text) for text in input_data]
		pickle.dump((corpus, id2word), open(corpus_f, 'wb'))

	if pickled_model:
		lda_model = pickle.load(open(model_f, 'rb'))
	else:
		print("Learning joint topic model")
		lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
		                                            id2word = id2word,
		                                            num_topics = k, 
		                                            random_state = 100,
		                                            update_every = 1,
		                                            chunksize = 100,
		                                            passes = 10,
		                                            alpha = 'auto',
		                                            per_word_topics = True)
		pickle.dump(lda_model, open(model_f, 'wb'))

	aggregate_JS = 0.0
	aggregate_KL = 0.0
	for example in tqdm(data):
		summary, document = lemmatization([example['summary'], example['text']], stop_words)
		summary_bow, document_bow = [id2word.doc2bow(text) for text in [summary, document]]
		summary_topic_distribution, document_topic_distribution = [lda_model.get_document_topics(text) for text in [summary_bow, document_bow]]
		summary_topic_distribution, document_topic_distribution = [{i : p for i,p in topic_distribution} for topic_distribution in [summary_topic_distribution, document_topic_distribution]]
		summary_topic_distribution, document_topic_distribution = [{i : topic_distribution.get(i, 0.0) for i in range(k)} for topic_distribution in [summary_topic_distribution, document_topic_distribution]]
		summary_topic_distribution, document_topic_distribution = [[topic_distribution[i] for i in range(k)]for topic_distribution in [summary_topic_distribution, document_topic_distribution]]
		aggregate_JS += JS_divergence(summary_topic_distribution, document_topic_distribution)
		# aggregate_KL += KL_divergence(summary_topic_distribution, document_topic_distribution)

	return 1 - (aggregate_JS / len(data))


def compute_ts_document(data, dataset, k, pickled_corpus=False, pickled_model=False):
	print("Computing Topic Similarity - Document-Only Topic Model with {} topics".format(k))

	corpus_f = 'dataset={}_documentTM.pickle'.format(dataset)
	model_f = 'dataset={}_k={}_documentTM.pickle'.format(dataset, k)

	stop_words = stopwords.words('english')
	stop_words.extend(['from', 'subject', 're', 'edu', 'use', '>', '<', '/t', '-lrb-', '-rrb-', '/n', '\n'])

	if pickled_corpus:
		corpus, id2word = pickle.load(open(corpus_f, 'rb'))
	else:
		print("Constructing joint topic model's corpus")
		document_data = [example['text'] for example in data]
		# summary_data = [example['summary'] for example in data]
		# input_data = document_data + summary_data
		input_data = document_data
		input_data = lemmatization(input_data, stop_words)
		
		id2word = corpora.Dictionary(input_data)
		corpus = [id2word.doc2bow(text) for text in input_data]
		pickle.dump((corpus, id2word), open(corpus_f, 'wb'))

	if pickled_model:
		lda_model = pickle.load(open(model_f, 'rb'))
	else:
		print("Learning joint topic model")
		lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
		                                            id2word = id2word,
		                                            num_topics = k, 
		                                            random_state = 100,
		                                            update_every = 1,
		                                            chunksize = 100,
		                                            passes = 10,
		                                            alpha = 'auto',
		                                            per_word_topics = True)
		pickle.dump(lda_model, open(model_f, 'wb'))

	aggregate_JS = 0.0
	aggregate_KL = 0.0
	for example in tqdm(data):
		summary, document = lemmatization([example['summary'], example['text']], stop_words)
		summary_bow, document_bow = [id2word.doc2bow(text) for text in [summary, document]]
		summary_topic_distribution, document_topic_distribution = [lda_model.get_document_topics(text) for text in [summary_bow, document_bow]]
		summary_topic_distribution, document_topic_distribution = [{i : p for i,p in topic_distribution} for topic_distribution in [summary_topic_distribution, document_topic_distribution]]
		summary_topic_distribution, document_topic_distribution = [{i : topic_distribution.get(i, 0.0) for i in range(k)} for topic_distribution in [summary_topic_distribution, document_topic_distribution]]
		summary_topic_distribution, document_topic_distribution = [[topic_distribution[i] for i in range(k)]for topic_distribution in [summary_topic_distribution, document_topic_distribution]]
		aggregate_JS += JS_divergence(summary_topic_distribution, document_topic_distribution)
		# aggregate_KL += KL_divergence(summary_topic_distribution, document_topic_distribution)

	return 1 - (aggregate_JS / len(data))
