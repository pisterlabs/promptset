import re
import numpy as np
import pandas as pd
import pickle
import gensim
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def replace_words(text):
    text = re.sub(r'U\.S\.', 'US', text)
    text = re.sub(r'U\.S\.A\.', 'US', text)
    text = re.sub(r'US', 'USA', text)
    text = re.sub(r'U\.K\.', 'UK', text)
    text = re.sub(r'Mr\.', 'MR', text)
    text = re.sub(r'Mrs\.', 'MRS', text)
    text = re.sub(r'Ms\.', 'MS', text)
    text = re.sub(r'\.\.\.', '', text)
    text = re.sub(r'U.S-China', 'US-China', text)
    text = text.replace('Co.', 'Co')
    text = text.replace('\xa0', '')
    text = text.replace('."', '".')
    text = text.replace('immediatelywith', 'immediately with')
    text = text.replace('theOfficeof', 'the Office of')
    text = text.replace('theCommissionerof', 'the Commissioner of')
    return text

# Modified from: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

stop_words = stopwords.words('english')

# Function to preprocess each sentence
def preprocess_sent(texts):
    texts_out = []
    for text in texts:
        simple_text = gensim.utils.simple_preprocess(text)
        no_stop = [ word for word in simple_text if word not in stop_words ]
        texts_out.append(no_stop)
    return texts_out

# Funtion to preprocess entire body
def preprocess_body(text):
    simple_text = gensim.utils.simple_preprocess(text)
    text_out = [ word for word in simple_text if word not in stop_words ]
    return text_out

# Following functions modified from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def _bigram_model(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def make_bigrams_sent(texts):
    bigram_mod = _bigram_model(texts)
    return [ bigram_mod[doc] for doc in texts ]

def make_bigrams(text):
    bigram_mod = _bigram_model(text)
    return bigram_mod[text]

nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(' '.join(text))
    return [ token.lemma_ for token in doc if token.pos_ in allowed_postags ]

def lemmatize_sent(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(' '.join(sent))
        texts_out.append(' '.join([ token.lemma_ for token in doc if token.pos_ in allowed_postags ]))
    return texts_out

mallet_path = '/Users/waynelam/Documents/DevStuff/mallet-2.0.8/bin/mallet'

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                 corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
# End taken functions

def vader_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def polarity(text):
    analysis = TextBlob(text)
    return analysis.sentiment[0]

def subjectivity(text):
    analysis = TextBlob(text)
    return analysis.sentiment[1]

file = open('mallet.p', 'rb')      
model = pickle.load(file)
file.close()

file = open('id2word.p', 'rb')      
dictionary = pickle.load(file)
file.close()

def sentiment_doc(combine):
    topic_0 = 0
    topic_1 = 0
    topic_2 = 0
    topic_3 = 0
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0

    reg = combine[0]
    tok = combine[1]
    num_sentences = len(tok)
    for i in range(num_sentences):
        tokens = tok[i].split()
        vec = dictionary.doc2bow(tokens)
        topic = topic = sorted(model[vec], key=lambda tup: tup[1], reverse=True)[0][0]
        sentiment = round(vader_analysis(reg[i])['compound'], 2)
        if topic == 0:
            topic_0 += sentiment
            count_0 += 1
        elif topic == 1:
            topic_1 += sentiment
            count_1 += 1
        elif topic == 2:
            topic_2 += sentiment
            count_2 += 1
        else:
            topic_3 += sentiment
            count_3 += 1
    return ((round(topic_0, 2), count_0), (round(topic_1, 2), count_1), (round(topic_2, 2), count_2), (round(topic_3, 2), count_3))

# def topic_sent(sentence_tokens):
#     s_tokens = sentence_tokens
#     topic_0 = 0
#     topic_1 = 0
#     topic_2 = 0
#     topic_3 = 0
#     topic_4 = 0
#     topic_5 = 0
#     topic_6 = 0
#     num_sentences = len(s_tokens)
#     for i in range(num_sentences):
#         tokens = s_tokens[i].split()
#         vec = id2word.doc2bow(tokens)
#         # model can be changed
#         topic = sorted(tfidf_lda_model[vec], key=lambda tup: tup[1], reverse=True)[0][0]
#         sentiment = round(100 * vader_analysis(s_tokens[i])['compound'], 2)
#         if topic == 0:
#             topic_0 += sentiment / num_sentences
#         elif topic == 1:
#             topic_1 += sentiment / num_sentences
#         elif topic == 2:
#             topic_2 += sentiment / num_sentences
#         elif topic == 3:
#             topic_3 += sentiment / num_sentences
#         elif topic == 4:
#             topic_4 += sentiment / num_sentences
#         elif topic == 5:
#             topic_5 += sentiment / num_sentences
#         else:
#             topic_6 += sentiment / num_sentences
#     return (round(topic_0, 2), round(topic_1, 2), round(topic_2, 2), round(topic_3, 2), round(topic_4, 2), round(topic_5, 2), round(topic_6, 2))

