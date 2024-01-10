import os
import pickle
import numpy as np
import pandas as pd
from algo.normal import calculate_compositions, count_unique_words

from algo.gp_based import mdkp
from algo.normal import greedy
# from algo.cvxpy_based import mdkp, mwbis
import cvxpy as cp


from mlxtend.frequent_patterns import apriori

from octis.evaluation_metrics.coherence_metrics import Coherence
from gensim.models import CoherenceModel
import pdb

def composite_activations(beta, data, vocab, bow_corpus, K):
    mat = np.sort(data, axis=1).mean(axis=0)
    assert len(mat) == K

    # Find a suitable threshold kappa hyper-parameter (currently using 5th largest mean activation)
    threshold = mat[-5]
    

    max_entries = 10000
    num_entries = 99999

    while num_entries > max_entries:
        reduced_data = np.zeros_like(beta)
        for i,j in np.argwhere(beta > threshold):
            reduced_data[i,j] = 1
        num_entries = reduced_data.sum()
        threshold += 0.01

    print('threshold', threshold)
    reduced_data = pd.DataFrame(reduced_data)
    

    min_s = 0.01
    frequent_itemsets = apriori(reduced_data, 
                                min_support = min_s, 
                                max_len = 5, 
                                use_colnames = True,
                                verbose = 1)

    topic_combinations = [list(a) for a in frequent_itemsets['itemsets']]

    # Get possible compositions O(b^{VK})
    output, topic_combinations = calculate_compositions(beta, topic_combinations, add_pairs=True)

    # Get top-10 vocab
    topics = np.argpartition(np.array(output),-10)[:,-10:]
    topics = [[vocab[idx] for idx in topic] for topic in topics]

    coherence = Coherence(texts=bow_corpus, measure='c_npmi')
    c = CoherenceModel(topics=topics, texts=coherence._texts, dictionary=coherence._dictionary,
                                          coherence='c_npmi', processes=4, topn=coherence.topk)
    total_scores = c.get_coherence_per_topic()
    # choices = greedy(topics, total_scores, K, 0)
    try:
        choices = mdkp(topics, total_scores, K, 0.935*K*10, range(K), MIP_gap=0.01, time_limit=1200)
    except:
        return topics[:K]
    # choices = mdkp(topics, total_scores, final_num_topics=K, epsilon=0.935*K*10,  solver=cp.GLPK_MI, solver_options={'reltol':0.02, 'max_iters':100})
    # choices2 = mwbis(topics, total_scores, final_num_topics=50, epsilon=1, solver=cp.GUROBI, solver_options={'MIPGap':0.05, 'TimeLimit':1000})
    
    # choices = greedy(topics, total_scores, K, 0.935*K*10, range(K), MIP_gap=0.01, time_limit=3600)
    optimized_topics = [topics[i] for i in range(len(choices)) if choices[i]]
    # new_scores = np.array(total_scores)[choices]
    # print("NPMI:", new_scores.mean())
    # print('TU:', count_unique_words(np.array(topics)[choices])/(K*10))
    assert len(optimized_topics) == K
    return optimized_topics