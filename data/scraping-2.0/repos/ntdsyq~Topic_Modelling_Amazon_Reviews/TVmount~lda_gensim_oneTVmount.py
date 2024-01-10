# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:07:39 2019

@author: yanqi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:34:29 2019

@author: yanqi
"""

import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 4 - Capstone\\AmazonReview'
os.chdir(proj_path)
import pandas as pd
pd.set_option('display.max_colwidth', -1)  # to view entire text in any column
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
import string
import gzip
from pprint import pprint

import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore

import pyLDAvis
import pyLDAvis.gensim   # for visualizing found topics
from model_utils import qc_dict, out_topics_docs, check_topic_doc_prob, topn_docs_by_topic, select_k

import warnings
warnings.simplefilter('ignore')  # suppressing deprecation warnings when running gensim LDA, 

# try: https://stackoverflow.com/questions/33572118/stop-jupyter-notebook-from-printing-warnings-status-updates-to-terminal?lq=1

#import logging  # add filename='lda_model.log' for external log file, set level = logging.ERROR or logging.INFO
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG) 

## Load data, prepare corpus and dictionary

select_cat = 'Electronics->Accessories & Supplies->Audio & Video Accessories->TV Accessories & Parts->TV Ceiling & Wall Mounts'.split('->')

# useful attributes & methods: dictionary.token2id to get mapping, dictionary.num_docs 
df = pd.read_csv(select_cat[-1] + "_processed.csv", index_col= 0)
df = df[ df['asin'] == '0972683275']
df.reset_index(drop=True, inplace = True)
reviews = df['review_lemmatized'].copy()
reviews = reviews.apply(lambda x: x.split())

# Dictionary expects a list of list (of tokens)
dictionary = corpora.Dictionary(reviews)
dictionary.filter_extremes(no_below=3)  # remove terms that appear in < 3 documents, memory use estimate: 8 bytes * num_terms * num_topics * 3

# number of terms
nd = dictionary.num_docs
nt = len(dictionary.keys())
print("number of documents", nd)
print("number of terms", nt)

qc_dict(dictionary)

# create document term matrix (corpus), it's a list of nd elements, nd = the number of documents
# each element of DTM (AKA corpus) is a list of tuples (int, int) representing (word_index, frequency)
DTM = [dictionary.doc2bow(doc) for doc in reviews]

## run lda model
#LDA = gensim.models.ldamodel.LdaModel
##LDA = gensim.models.ldamulticore.LdaMulticore
#n_topics = 10
#passes = 10
#iterations = 400
#
#%time lda_model = LDA(corpus=DTM, id2word=dictionary, num_topics=n_topics, alpha = 'auto', eta = 'auto', passes = passes, iterations = iterations, eval_every = 1, chunksize = 20)
##%time lda_model = LDA(corpus=DTM, id2word=dictionary, num_topics=n_topics, eta = 'auto', passes = passes, iterations = iterations, eval_every = 1, workers = 3, chunksize = 2000)
#
### check priors, conherence score, and create topic visualization
#coherence_lda_model = CoherenceModel(model=lda_model, texts=reviews, dictionary=dictionary, coherence='c_v')
#cs = coherence_lda_model.get_coherence()
#print("model coherence score is:", cs)

# trying out select_k function 
limit=20; start=3; step=1;
model_list, coherence_values = select_k(corpus = DTM, dictionary = dictionary, texts = reviews, limit = limit, start= start, step= step)

# plot coherence score as function of number of topics
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

lda_model = model_list[9]  # 12 topics

print(lda_model.alpha)
print(max(lda_model.eta))
print(min(lda_model.eta))
print(np.mean(lda_model.eta))

pprint(lda_model.print_topics())

## A closer look at the document_topics distribution
len(lda_model[DTM]) 
topics_docs_dict = out_topics_docs(lda_model, DTM)

# check doc_topic probability distribution
for t in sorted(topics_docs_dict.keys()):
    test_prob = check_topic_doc_prob(topics_docs_dict, t)
    print(test_prob.describe(),"\n")
    
# examine each topic by topic key words, number of generated documents, document probabilities, docs with top probabilities
topic_num = 2
print(lda_model.show_topic(topicid=topic_num))
print("topic", topic_num, "has", len(topics_docs_dict[topic_num]),"documents")
print("Distribution of probabilities of documents being generated from this topic:")
doc_prob = check_topic_doc_prob(topics_docs_dict, topic_num)
print(doc_prob.describe(),"\n")

top_docprobs = topn_docs_by_topic(topics_docs_dict,topic_num, 10)
idxs = pd.Series([x[0] for x in top_docprobs])
probs = pd.Series([x[1] for x in top_docprobs])
texts = pd.Series([df['review_no_html'][i] for i in idxs])
products = pd.Series([df['title'][i] for i in idxs])
asins = pd.Series([df['asin'][i] for i in idxs])
top_docs_df = pd.concat([asins, products, idxs, probs, texts], axis = 1)
top_docs_df.columns = ['asin','product','doc_id', 'prob_from_topic','reviewText']
pd.set_option('display.max_columns', 500)
display(top_docs_df[['prob_from_topic','reviewText']])


## create topic vis
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, DTM, dictionary)
