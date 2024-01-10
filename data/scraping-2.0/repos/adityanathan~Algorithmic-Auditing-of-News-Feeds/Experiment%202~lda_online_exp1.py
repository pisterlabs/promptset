import pandas as pd
import numpy as np
import pickle
import re
import timeit
import spacy

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel, LdaModel, LdaMulticore
from nltk.corpus import stopwords
import helper as he

# Number of documents to consider as part of existing model in simulation.
# Here it is the no. of documents in Mass Media DB
INITIAL_DOC_SIZE = 17850
STEP_SIZE = 200

# Load Data - corp.pkl contains data_lemmatized, id2word, corpus
with open('corp.pkl', 'rb') as f:
    data_lemmatized, _, _ = pickle.load(f)

# Initialize Parameters
total_time = 0
coherence_arr = []
time_arr = []

# Set Data State to that of existing model in simulation
data = data_lemmatized[:INITIAL_DOC_SIZE]
# When updating Online LDA, if I use a normal dictionary I keep getting key errors.
# That's why for online lda alone I use Hash Dictionary
id2word = Dictionary(documents=data)
corpus = [id2word.doc2bow(doc) for doc in data]

# Building for the first time - To be considered as the starting/existing model in simulation.
start = timeit.default_timer()
lda = LdaMulticore(corpus, num_topics=35, id2word=id2word,
                   workers=3, chunksize=2000, passes=10, batch=False)
end = timeit.default_timer()

time_taken = end - start
total_time += time_taken
time_arr.append(time_taken)

# Obtain C_V Coherence of existing model in simulation
coherencemodel2 = CoherenceModel(
    model=lda, texts=data, dictionary=id2word, coherence='c_v')
coherence_arr.append(coherencemodel2.get_coherence())
count = 0
# The loop simulates arrival of new documents from Google Alerts in batches of STEP_SIZE
for i in range(INITIAL_DOC_SIZE, len(data_lemmatized)-STEP_SIZE, STEP_SIZE):
    # new_docs is the list of STEP_SIZE new documents which have arrived
    new_docs = data_lemmatized[i:i+STEP_SIZE]
    pruned_docs = []
    for doc in new_docs:
        pruned_data = []
        for x in doc:
            if x in id2word.values():
                pruned_data.append(x)
        pruned_docs.append(pruned_data)

    new_docs = pruned_docs
    print('Pruning Done')
    # Updating Dictionary
    # id2word.add_documents(new_docs)
    # id2word.filter_extremes(no_below=5, no_above=0.95,
    #                         keep_n=1800)

    # Converting Documents to doc2bow format so that they can be fed to models
    corpus = [id2word.doc2bow(doc) for doc in new_docs]
    count += 1
    print('MODEL NO:'+str(count))
    start = timeit.default_timer()
    lda.update(corpus)
    print('MODEL DONE')
    end = timeit.default_timer()

    time_taken = end - start
    total_time += time_taken
    time_arr.append(time_taken)

    # Coherence of model updated so far on the entire corpus from start i.e including Mass Media DB
    coherencemodel2 = CoherenceModel(
        model=lda, texts=data_lemmatized[:i+STEP_SIZE], dictionary=id2word, coherence='c_v')
    coherence_arr.append(coherencemodel2.get_coherence())

    with open('models_online.pkl', 'ab') as f2:
        pickle.dump(
            (count, lda, coherence_arr, time_arr, data_lemmatized[:i+STEP_SIZE], id2word, i), f2)


print(total_time)
