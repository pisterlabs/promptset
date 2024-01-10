import os
import nltk
# !pip install --upgrade gensim
nltk.download('stopwords')
# !pip install pyLDAvis
import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle
import en_core_web_sm

nlp = en_core_web_sm.load()

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.corpus import stopwords

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# def remove_stopwords(texts):
#     return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]


# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def modelTopic(data):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    # data = ["I love to study in my school. The teacher is not that cool though",
    #     "A bigram or digram is a sequence of two adjacent elements from a string of tokens, which are typically letters, syllables, or words.",
    #     "The NBA's draft lottery won't take place Tuesday in Chicago as originally planned, but whenever it does happen, it is likely to look the same as it did last year, league sources told ESPN.",
    #         "Since play was suspended March 11 due to the coronavirus pandemic, teams at the top of the standings have been curious about the league restarting because they are in pursuit of a championship. For teams at the bottom of the standings, the focus has been on what the lottery will look like.",
    #         "I love to code. My teacher is soo cool"
    #     ]
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    print('data is =')
    print(data)
    data_words = list(sent_to_words(data))
    print('data_words is =')
    print(data_words)
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    print('bigram is =')
    print(bigram)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    print('tri gram is =')
    print(trigram)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = data_words
    t1 =[[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    data_words_nostops = t1

    texts = data_words_nostops
    t2=[bigram_mod[doc] for doc in texts]
    data_words_bigrams = t2
    print('data_words_bigrams')
    print(data_words_bigrams)
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print('data_lemmatized')
    print(data_lemmatized)
    id2word = corpora.Dictionary(data_lemmatized)
    print('id2word')
    print(id2word)
    # original_id2word = pickle.load(open("/content/M2", 'rb'))
    texts = data_lemmatized
    # new_id2word = original_id2word.merge_with(id2word)
    corpus = [id2word.doc2bow(text) for text in texts]
    unseen_doc = corpus[0]
    print('Unseen_doc')
    print(unseen_doc)
    # pickle.dump(corpus, open("M1", 'wb'))
    # [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
    # pickle.dump(id2word, open("M2", 'wb'))
    new_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    print('after model')
    # pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(new_lda_model, corpus, id2word,mds='mmds')
    # pyLDAvis.enable_notebook()
    # pyLDAvis.display(vis)
    print('Before html')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(BASE_DIR)
    # p = os.path.join(BASE_DIR,'/ComDisp/templates/comments/LDA.html')
    print('Path is ')
    # print(BASE_DIR)
    # print(p)
    p = BASE_DIR+'/ComDisp/templates/comments/LDA.html'
    print(p)
    pyLDAvis.save_html(vis, p)
    print('last print here')
    return vis






# stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# data = ["I love to study in my school. The teacher is not that cool though",
#        "A bigram or digram is a sequence of two adjacent elements from a string of tokens, which are typically letters, syllables, or words.",
#        "The NBA's draft lottery won't take place Tuesday in Chicago as originally planned, but whenever it does happen, it is likely to look the same as it did last year, league sources told ESPN.",
#         "Since play was suspended March 11 due to the coronavirus pandemic, teams at the top of the standings have been curious about the league restarting because they are in pursuit of a championship. For teams at the bottom of the standings, the focus has been on what the lottery will look like.",
#         "I love to code. My teacher is soo cool"
#        ]
# data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# data = [re.sub('\s+', ' ', sent) for sent in data]

# data = [re.sub("\'", "", sent) for sent in data]


# print('data is =')
# print(data)

# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# data_words = list(sent_to_words(data))


# print('data_words is =')
# print(data_words)

# bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.

# print('bigram is =')
# print(bigram)

# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# print('tri gram is =')
# print(trigram)

# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)


# def remove_stopwords(texts):
#     return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]


# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]


# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent))
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out



# data_words_nostops = remove_stopwords(data_words)

# data_words_bigrams = make_bigrams(data_words_nostops)

# print('data_words_bigrams')
# print(data_words_bigrams)


# data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# print('data_lemmatized')
# print(data_lemmatized)

# id2word = corpora.Dictionary(data_lemmatized)

# print('id2word')
# print(id2word)
# # original_id2word = pickle.load(open("/content/M2", 'rb'))
# texts = data_lemmatized
# # new_id2word = original_id2word.merge_with(id2word)

# corpus = [id2word.doc2bow(text) for text in texts]
# unseen_doc = corpus[0]

# print('Unseen_doc')
# print(unseen_doc)

# # pickle.dump(corpus, open("M1", 'wb'))

# # [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

# # pickle.dump(id2word, open("M2", 'wb'))
# new_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                             id2word=id2word,
#                                             num_topics=20,
#                                             random_state=100,
#                                             update_every=1,
#                                             chunksize=100,
#                                             passes=10,
#                                             alpha='auto',
#                                             per_word_topics=True)


# print('after model')

# pickle.dump(lda_model, open("M", 'wb'))
# lda_model= pickle.load(open("/content/M", 'rb'))
# unseen_doc = corpus[0]
# vec = lda_model[unseen_doc]
# print(vec)

# # lda_model.update(corpus)
# # pprint(lda_model.print_topics())
# original_corpus = pickle.load(open("/content/M1", 'rb'))

# # print(type(id2word))



# # pprint(lda_model.print_topics())
# # 
# num_topics = 10
# # Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# doc_lda = lda_model[corpus]
# top_topics = lda_model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
# avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
# print('Average topic coherence: %.4f.' % avg_topic_coherence)

# from pprint import pprint
# pprint(top_topics)


# Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(new_lda_model, corpus, id2word,mds='mmds')
# # pyLDAvis.enable_notebook()
# pyLDAvis.display(vis)