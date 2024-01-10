import pandas as pd
import numpy as np
import pickle
import re
import timeit
import spacy
import copy

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel, LdaModel, LdaMulticore
from nltk.corpus import stopwords
import helper as he

# Number of documents to consider as part of existing model in simulation.
# Here it is the no. of documents in Mass Media DB
# INITIAL_DOC_SIZE = 9442
INITIAL_DOC_SIZE = 4000
# STEP_SIZE = 200

# DOC_TEMPORAL_SIZE = [9442, 12710, 14339, 15445, 16287, 16998, 17849]
# DOC_TEMPORAL_SIZE = [6730, 11328, 13466,
#                           14874, 15926, 16765, 17464, 18063, 18587, 19053]
# DOC_TEMPORAL_SIZE = [6730, 9631, 11328, 12532, 13466, 14229, 14874, 15433,
#                           15926, 16366, 16765, 17129, 17464, 17775, 18063, 18333, 18587, 18826, 19053]

# DOC_TEMPORAL_INCREMENT = [3268, 1629, 1106,
#                           842, 711, 851]  # 2 months, 2 months
DOC_TEMPORAL_INCREMENT = [4598, 2730, 2138,
                          1408, 1052, 839, 699, 599, 524, 466, 500, 500]
# DOC_TEMPORAL_INCREMENT = [3268, 1629, 1106,
#                           842, 711, 851, 699, 599, 524, 466]  # 2 weeks, 1 month
# DOC_TEMPORAL_INCREMENT = [4598, 2138, 1408, 1052,
#                           839, 699, 599, 524, 466]  # 2 weeks, 2 weeks


# dummy_lst = [600]*4

# DOC_TEMPORAL_INCREMENT.extend(dummy_lst)

# Load Data - corp.pkl contains data_lemmatized, id2word, corpus
with open('corp.pkl', 'rb') as f:
    data_lemmatized, _, _ = pickle.load(f)

# Initialize Parameters
# total_time = 0
# coherence_arr = []
# time_arr = []

# Set Data State to that of existing model in simulation
data = data_lemmatized[:INITIAL_DOC_SIZE]
id2word = Dictionary(documents=data_lemmatized)
corpus = [id2word.doc2bow(doc) for doc in data]

# Building for the first time - To be considered as the starting/existing model in simulation.
# start = timeit.default_timer()
lda = LdaMulticore(corpus, num_topics=35, id2word=id2word,
                   workers=3, chunksize=2000, passes=10, batch=False)

corpus_baseline = copy.deepcopy(corpus)
lda_baseline = copy.deepcopy(lda)

doc_size = []
positive_arr = []
# end = timeit.default_timer()

# time_taken = end - start
# total_time += time_taken
# time_arr.append(time_taken)

# Obtain C_V Coherence of existing model in simulation
# coherencemodel2 = CoherenceModel(
# model=lda, texts=data, dictionary=id2word, coherence='c_v')
# coherence_arr.append(coherencemodel2.get_coherence())

f2 = open('confusion_temporal_2.pkl', 'wb')

count = 0
# The loop simulates arrival of new documents from Google Alerts in batches of STEP_SIZE
# len(data_lemmatized)-STEP_SIZE

doc_size_counter = INITIAL_DOC_SIZE

for i in DOC_TEMPORAL_INCREMENT:
    # new_docs is the list of STEP_SIZE new documents which have arrived
    new_docs = data_lemmatized[doc_size_counter:doc_size_counter+i]
    doc_size_counter += i
    # pruned_docs = []
    # for doc in new_docs:
    #     pruned_data = []
    #     for x in doc:
    #         if x in id2word.values():
    #             pruned_data.append(x)
    #     pruned_docs.append(pruned_data)

    # new_docs = pruned_docs
    # print('Pruning Done')

    # Updating Dictionary
    # id2word.add_documents(new_docs)
    # id2word.filter_extremes(no_below=5, no_above=0.95,
    #                         keep_n=1800)

    prev_corpus = copy.deepcopy(corpus)

    # Converting Documents to doc2bow format so that they can be fed to models
    corpus = [id2word.doc2bow(doc) for doc in new_docs]
    count += 1

    # prev_lda = copy.deepcopy(lda)

    print('MODEL NO:'+str(count))
    # start = timeit.default_timer()
    lda.update(corpus)
    print('MODEL DONE')
    # end = timeit.default_timer()

    # time_taken = end - start
    # total_time += time_taken
    # time_arr.append(time_taken)

    # updated_corpus = copy.deepcopy(prev_corpus)
    # updated_corpus.extend(corpus)
    # corpus = updated_corpus
    prev_corpus.extend(corpus)
    corpus = copy.deepcopy(prev_corpus)

    doc_size.append(i)
    positive_arr.append(he.calc_confusion_matrix(
        lda_baseline, lda, corpus))
    # doc_size.append(len(corpus))

    # positive_arr.append(he.calc_confusion_matrix(
    #     prev_lda, lda, prev_corpus))

    # Coherence of model updated so far on the entire corpus from start i.e including Mass Media DB
    # coherencemodel2 = CoherenceModel(
    #     model=lda, texts=data_lemmatized[:i+STEP_SIZE], dictionary=id2word, coherence='c_v')
    # coherence_arr.append(coherencemodel2.get_coherence())

    # with open('models_online.pkl', 'ab') as f2:
    pickle.dump((positive_arr, doc_size), f2)
    # pickle.dump(
    # (count, lda, coherence_arr, time_arr, data_lemmatized[:i+STEP_SIZE], id2word, i), f2)

f2.close()
# print(total_time)
