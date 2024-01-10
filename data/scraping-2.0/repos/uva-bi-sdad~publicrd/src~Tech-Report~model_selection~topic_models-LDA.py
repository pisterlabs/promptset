import pandas as pd
#import numpy as np
import pickle
import time
#import gc

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models.coherencemodel import CoherenceModel


# data needed for coherence calculation

# import entire dataset
f = open('coherence_vars.sav', 'rb')

[id2word, docs] = pickle.load(f)
f.close()

print("data ingested--------------------------", flush = True)

# corpus - word frequency in docs - not needed for coherence calculation
# id2word - dictionary
# docs - df["final_tokens"]

# input needed for LDA, NMF and LSA (all from Scikit-Learn) is one string per document (not a list of strings)

text = []

for abstract in docs:
    text.append(" ".join(abstract))
        
        
# Function to format topics as a "list of list of strings".
# Needed for topic coherence function in Gensim

# function modified from https://nlpforhackers.io/topic-modeling/

def list_topics(topic_term_dist, vectorizer, top_n=10):

    #input. top_n: how many words to list per topic.  If -1, then list all words.
       
    topic_words = []
    
    for idx, topic in enumerate(topic_term_dist):  # loop through each row of H.  idx = row index.  topic = actual row
            
        if top_n == -1:   
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[::-1]])
        else:
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]])
        
    return topic_words


# create document-term matrix

stop_wds = ['research', 'study', 'project']  # use will be eliminated by max_df

vectorizer = CountVectorizer(max_df=0.6, min_df=20, lowercase=False, stop_words=stop_wds)
doc_term_matrix = vectorizer.fit_transform(text)

print("doc term matrix computed------------", flush = True)


# delete text - no longer needed
#del text
#gc.collect()

# run once so start up time isn't factored into first iteration time
lda_model = LatentDirichletAllocation(n_components=1, doc_topic_prior = 1, 
                                              topic_word_prior=0.1, n_jobs=39, random_state = 0)
lda_model.fit_transform(doc_term_matrix)



print("model loop beginning-----------", flush = True)

# function adapted from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

def lda_models(doc_term_matrix, n_topics, vectorizer, rand_start):
    """
    Compute LDA model & find perplexity, save topics list for coherence calc

    Parameters:
    ----------
    doc_term_matrix
    n_topics : list of number of topics
    """

    perplexity_values = []
    lda_time = []
    topics_list = []
    
    i = rand_start 
    for num_topics in n_topics:
        
        # create model
        t1 = time.time()
        lda_model = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior = 1/num_topics, 
                                              topic_word_prior=0.1, n_jobs=39, random_state = i) 
        lda_model.fit_transform(doc_term_matrix)
        t2 = time.time()
        lda_time.append(t2-t1)
        print(f"  Model time: {t2-t1}", flush = True)
        
        # compute perplexity
        perplexity_values.append(lda_model.bound_)
        
        # create list of topics
        topics = list_topics(lda_model.components_, vectorizer, top_n=10)
        topics_list.append(topics)
        
        # output completion message
        i = i+1
        print('Number of topics =', num_topics, "complete.", flush = True)

    return perplexity_values, lda_time, topics_list


# code copied from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
# minor alterations made
 
n_topics = list(range(5,131,5)) + [140, 150, 175, 200]
num_runs = 2

batch= 8

col_names = [f"iteration {i+batch}" for i in range(num_runs)]
lda_p = pd.DataFrame(index = n_topics, columns = col_names)
lda_t = pd.DataFrame(index = n_topics, columns = col_names)
lda_topics = pd.DataFrame(index = n_topics, columns = col_names)

for i in range(num_runs):
    
    print(f"Iteration {i}", flush = True)
    
    # run models
    [p, t, topic_terms] = lda_models(doc_term_matrix=doc_term_matrix, n_topics=n_topics, vectorizer=vectorizer, 
                         rand_start = (i+batch)*len(n_topics)) 
    
    # save results
    lda_p[f"iteration {i+batch}"] = p
    lda_t[f"iteration {i+batch}"] = t
    lda_topics[f"iteration {i+batch}"] = topic_terms
       
        
# save results 

lda_p.to_pickle("./results/LDA/lda_p8-9.pkl")
lda_t.to_pickle("./results/LDA/lda_t8-9.pkl")
lda_topics.to_pickle("./results/LDA/lda_topics8-9.pkl")
