from gensim import corpora
import gensim
import os
import numpy
import pickle
from gensim.models.coherencemodel import CoherenceModel


def compute_coherence_values(dictionary, bow_corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    max_coherencemodel = 0
    best_topic = 0
    for num_topics in range(start, limit, step):
        print('Training with ', num_topics, ' Topic')
        lda_model = gensim.models.LdaMulticore(
            bow_corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=15,
            workers=8,
            minimum_probability=0.04,
            random_state=50,
            alpha=1e-2,
            chunksize=4000,
            eta=0.5e-2,
        )
        coherencemodel = CoherenceModel(
            model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')

        print("Num Topics =", num_topics, " coherence = ",
              round(coherencemodel.get_coherence(), 4))

        if max_coherencemodel < round(coherencemodel.get_coherence(), 4) :
            max_coherencemodel = round(coherencemodel.get_coherence(), 4)
            best_model = lda_model
            best_topic = num_topics
        
        coherence_values.append(coherencemodel.get_coherence())

    return best_model, best_topic, coherence_values
