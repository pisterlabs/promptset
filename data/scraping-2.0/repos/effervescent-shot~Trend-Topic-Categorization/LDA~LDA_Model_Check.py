import os
import re
import sys
import numpy as np
from datetime import datetime
import pandas as pd
import multiprocessing as mp

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary, datapath
import gensim

from gensim.corpora import MmCorpus, Dictionary
from gensim.test.utils import get_tmpfile


DATA_DIR = "../Data"
TWEETS_PATH = os.path.join(DATA_DIR, 'tweets')
TREND_PATH = os.path.join(DATA_DIR, 'trends')
SAVE_PATH = os.path.join(DATA_DIR, 'save')
STATS_PATH = os.path.join(DATA_DIR, 'stats')
TOPICS_PATH = os.path.join(DATA_DIR, 'topics')


def load_lda_datasets():

    corps = MmCorpus.load("./ldadata/corpus0")
    print("LOADING CORPUS, LENGTH: ", len(corps))

    dicts = Dictionary.load_from_text("./ldadata/dictionary0")
    print("LOADING DICTIONARY, LENGTH: ", len(dicts))

    dataset = pd.read_pickle("./ldadata/stemmed_data.pkl")
    print("LOADING DATASET, LENGTH: ", len(dataset))

    return dataset, corps, dicts

def models_run_function(topic_num):
    # Model with the best coherence_value
    print('Model start running for ', topic_num, ' topics. Start time: ', datetime.now())
    lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num,
                                        random_state=1, chunksize=100, passes=5,
                                        alpha='auto', per_word_topics=True)
    cwd = os.getcwd()
    temp_file = datapath(os.path.join(cwd, "models/lda_model_"+str(topic_num)))
    print('Finish time: ', datetime.now(), 'Model is saving... at', temp_file, '\n')
    lda_model.save(temp_file)

    return lda_model


def models_run_parallelized():

    pool = mp.Pool(mp.cpu_count()-2)
    print("Parallel Run!")
    for topic_number in range(7, 25, 3):
        pool.apply_async(models_run_function, args=(topic_number))

    pool.close()
    pool.join()


def models_check(topic_num):

    cwd = os.getcwd()
    temp_file = datapath(os.path.join(cwd, "models/lda_model_"+str(topic_num)))
    lda_model = models.ldamodel.LdaModel.load(temp_file)
    print("Topic number = ", topic_num)

    # Compute Perplexity Score
    print('Perplexity Score: ', lda_model.log_perplexity(corpus))

    # Compute Coherence Score
    cohr_val = CoherenceModel(model=lda_model, texts=stemmed_dataset, corpus=corpus, dictionary=dictionary, coherence='c_v').get_coherence()
    print('Coherence Score: ', cohr_val)
    print("========================================")


if __name__ == '__main__':

    print("HERE WE GO!")
    stemmed_dataset, corpus, dictionary = load_lda_datasets()

    for topic_number in range(7, 26, 3):
        models_run_function((topic_number))

    for topic_number in range(7, 26, 1):
        models_check(topic_number)
