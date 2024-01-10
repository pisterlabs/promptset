# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim import similarities

from tqdm import tqdm
import pandas as pd
import numpy as np

# spacy for lemmatization
import spacy

# Plotting tools
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#% matplotlib inline

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import os.path
import re
import glob

import nltk

nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

os.environ['MALLET_HOME'] = '/content/mallet-2.0.8'
mallet_path = '/content/mallet-2.0.8/bin/mallet'

#최적의 토픽 수를 찾기 위해 여러 토픽 수로 일관성을 계산하고 비교
def compute_coherence_values(mallet_path, id2word, corpus, texts, limit, start=8, step=2, early_stop=True):
    coherence_values = []
    model_list = []
    topic_cnt = 0

    for num_topics in tqdm(range(start, limit, step)):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    for idx, value in enumerate(coherence_values[1:]):
        if coherence_values[topic_cnt] < value:
            topic_cnt = idx
        elif (coherence_values[topic_cnt] >= value) and (early_stop):
            break

    return model_list, coherence_values, topic_cnt


def coherence_graph(start, limit, step, coherence_values, path):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Topic Number")
    plt.ylabel("Coherence")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(path)


def mallet_to_lda(mallet_model):
    '''
    :param mallet_model: mallet's LDA model
    :return: gensim's LDA model
    change mallet's LDA model to gensim's LDA model.
    To ensure successful visualization in pyLDAvis.
    '''
    model_gensim = gensim.models.LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0, iterations=1000,
        gamma_threshold=0.001,
        dtype=np.float32
    )
    model_gensim.sync_state()
    model_gensim.state.sstats = mallet_model.wordtopics
    return model_gensim


def coherence_score(model, texts, dictionary, coherence='c_v'):
    coherence_model_ldamallet = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence)
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    return coherence_ldamallet



def summary(model, corpus, texts):
    '''
    :param model: Gensim LDA model
    :param corpus: corpus that input value fo LDA model
    :param texts: texts that input value of LDA model
    :param num_topics: number of topics
    :return: dataframe df
    df.columns = ['Keywords', 'Num_Documents', 'Perc_Documents'], descending sort
    '''
    df = pd.DataFrame()
    df_topic_sents_keywords = pd.DataFrame()
    num_topics = model.num_topics
    # df_topic_sents_keywords = format_topics_sentences(ldamodel=model, corpus=corpus, texts=texts)
    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                df_topic_sents_keywords = df_topic_sents_keywords.append(
                    pd.Series([int(topic_num), topic_keywords]), ignore_index=True)
            else:
                break
    df_topic_sents_keywords.columns = ['Dominant_Topic', 'Topic_Keywords']

    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts / topic_counts.sum(), 4)
    for topic_num in range(num_topics):
        wp = model.show_topic(topic_num)
        topic_keywords = ", ".join([word for word, prop in wp])
        df = df.append(
            pd.Series([topic_num, topic_keywords]), ignore_index=True)

    # change columns name
    df.columns = ['Dominant_Topic', 'Keywords']

    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts / topic_counts.sum(), 4)

    # Concatenate Column wise
    df = pd.concat([df, topic_counts, topic_contribution], axis=1)

    # change columns name
    df.columns = ['Dominant_Topic', 'Keywords', 'Num_Documents', 'Perc_Documents']

    # del unnecessary col
    df = df.drop(['Dominant_Topic'], axis=1)

    # sort by the number of documents belonging to
    df = df.sort_values(by=['Num_Documents'], ascending=False, ignore_index=True)
    return df