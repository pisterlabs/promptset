from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.callbacks import PerplexityMetric
from gensim.models import CoherenceModel
from pprint import pprint
import pandas as pd
import numpy as np
import pickle
import difflib
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def remove_type(ner_list):
    ner_list = eval(ner_list)
    return np.array([entity[0] for entity in ner_list])

def get_docs(year):
    ''' Get named entites and remove types to represent documents as BoW '''
    print('-'*20 + 'Reading Data' + '-'*20)
    df = pd.read_csv('data/news-%d.csv' % year)
    ner = df['ner'].values
    df['ner-no-type'] = df['ner'].apply(lambda x: remove_type(x))
    # df['ner-merged'] = df['ner'].apply(lambda x: ' '.join(x))
    return df['ner-no-type'].values #, df['ner-merged'].values

def get_ids_dates(year):
    df = pd.read_csv('data/news-%d.csv' % year)
    month = df['month'].values
    ids = df['Unnamed: 0'].values
    return ids, month

def get_dict_corpus(year):
    ''' 
    Extract corpus and dictionary, 
    create BoW represenatiton for documents
    '''
    documents = get_docs(year)
    print('Number of Documents = %d' % documents.shape[0])
    print('-'*20 + 'Vectorizing Data' + '-'*20)
    id2word = Dictionary(documents)
    print('Number of words before filtering extremes = %d' % len(id2word))
    id2word.filter_extremes(no_below=10, no_above=0.2)
    print('Number of words after filtering extremes = %d' % len(id2word))
    corpus = [id2word.doc2bow(doc) for doc in documents]
    pickle.dump(corpus, open(f'models/corpus-{year}.p', 'wb'))
    pickle.dump(id2word, open(f'models/dict-{year}.p', 'wb'))
    return id2word, corpus, documents

def lda(id2word, corpus):
    ''' Fit LDA model to the data '''
    print('-'*20 + 'LDA' + '-'*20)
    lda_model = LdaMulticore(
        corpus=corpus, 
        id2word=id2word, 
        num_topics=20, 
        random_state=100, 
        per_word_topics=True,
        workers=4,
        chunksize=10000,
        passes=5
    )
    pickle.dump(lda_model, open(f"models/lda-{year}.p", "wb"))
    return lda_model

def metrics(lda_model, docs, corpus, id2word):
    # Print Topics
    pprint(lda_model.print_topics())

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

def get_topics(lda_model, corpus):
    ''' Get the topics for each document in the corpus '''

    topic_numbers = []
    topic_weights = []
    topic_words = []
    for i, row in enumerate(tqdm(lda_model[corpus])):
        sorted_row = sorted(row[0], key=lambda x:x[1], reverse=True)
        topic_num, weight = sorted_row[0]
        wp = lda_model.show_topic(topic_num)
        topic_numbers.append(topic_num)
        topic_weights.append(weight)
        topic_words.append(wp)
    return topic_numbers, topic_weights, topic_words

def create_topics_df(year):
    ''' Save topics of documents '''
    
    # Load models
    id2word = pickle.load(open(f'models/dict-{year}.p', 'rb'))
    corpus = pickle.load(open(f'models/corpus-{year}.p', 'rb'))
    lda_model = pickle.load(open(f'models/lda-{year}.p', 'rb'))

    # Get ids and months
    doc_ids, doc_months = get_ids_dates(year)
    topic_numbers, topic_weights, topic_words = get_topics(lda_model, corpus)
    df = pd.DataFrame({'index': doc_ids, 'month': doc_months, 'topic': topic_numbers, 
        'weight': topic_weights, 'topic-words': topic_words})
    print(df.head())

    # export df
    df.to_csv(f'data/topics-{year}.csv', index=False)

def filter_words(row):
    ''' Filters the topic words by removing similar words '''

    output = []
    for r in eval(row):
        similar = difflib.get_close_matches(r[0], output)
        if len(similar) == 0:
            output.append(r[0])
    return output[:7]

def average_weight(year, topics):
    ''' Visualize topics of a year '''

    df = pd.read_csv(f'data/topics-{year}.csv')
    df = df[df['topic'].isin(topics)]
    df_group = df.groupby(['month', 'topic-words'])['weight'].mean().reset_index()
    df_group['topic-words'] = df_group['topic-words'].apply(lambda x: ', '.join(filter_words(x)))
    print(df_group.head())
    # fg = group.reset_index().pipe((sns.factorplot, 'data'), x='month', y='weight', hue='topic')
    sns_plot = sns.relplot(x='month', y='weight', hue='topic-words', kind="line", data=df_group)
    sns_plot.savefig(f'Outputs/topic-{year}.png')
    # plt.show()
    
''' Create a topic model for each year and save its outputs '''

# for year in range(2016, 2021):
#     id2word, corpus, _ = get_dict_corpus(year)
#     lda_model = lda(id2word, corpus)
#     metrics(lda_model, corpus, id2word)
#     create_topics_df(year)

''' Visualize topic words'''

for year in range(2016, 2021):
    id2word, corpus, docs = get_dict_corpus(year)
    lda_model = pickle.load(open(f'models/lda-{year}.p', 'rb'))
    print('-'*25 + str(year) + '-'*25)
    metrics(lda_model, docs, corpus, id2word)
    # pprint(lda_model.print_topics())

''' Plot Selected Topics '''

# average_weight(2016, [1, 3, 7, 13, 14])
# average_weight(2017, [5, 11, 15, 16, 17])
# average_weight(2018, [3, 7, 11, 12, 16])
# average_weight(2019, [1, 4, 6, 10, 18])
# average_weight(2020, [2, 5, 12, 13, 14])
