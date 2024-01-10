import re
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
spacy.load('en')

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Enable logging for gensim - optional
import logging as log
log.basicConfig(
    level=log.ERROR)

import ml.download_data.fetch_housing_data as url_retriever
import ml.ml_utils as utils

# log.basicConfig(filename="logs/app-logs.log",
# format='%(asctime)s : %(levelname)s : %(message)s', level=log.DEBUG)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

log.debug("Hello")

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def bigram_trigram_model(data_words):
    # Build the bigram and trigram models
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print("Bi Gram model: ", bigram_mod)
    print
    ("Tri gram model: ", trigram_mod)
    print(trigram_mod[bigram_mod[data_words[0]]])
    return bigram_mod, trigram_mod


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def clean_content(content_as_list):
    # Remove Emails
    content_as_list = [re.sub('\S*@\S*\s?', '', sent)
                       for sent in content_as_list]

    # Remove new line characters
    content_as_list = [re.sub('\s+', ' ', sent) for sent in content_as_list]

    # Remove distracting single quotes
    content_as_list = [re.sub("\'", "", sent) for sent in content_as_list]
    print("First index of Content: \n", content_as_list[:1])
    return content_as_list

# t1 = utils.current_time_in_millis()
url = 'https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json'
# df = pd.read_json(url)
# print(df.target_names.unique())
# log.info("DF Head: %s", df.head().to_string())
# t2 = utils.current_time_in_millis()
# s = t2 - t1
# print("Time ( in ms ) for URL %s", (t2 - t1))

t1 = utils.current_time_in_millis()
file_path = url_retriever.url_retrieve(
    url, 'datasets/newsgroup', 'newsgroup.json')
print(file_path)
df = pd.read_json(file_path)
print(df.target_names.unique())
log.info("Df Head : \n%s", df.head())
t2 = utils.current_time_in_millis()
print("Time ( in ms ) for Downloaded", (t2 - t1))

data = df.content.values.tolist()
cleaned_data = clean_content(data)
data_words = sent_to_words(cleaned_data)
list_words = [x for x in data_words]
print("data words: \n", list_words[0])

bigram_mod, trigram_mod = bigram_trigram_model(list_words)

data_words_nostops = remove_stopwords(list_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=[
                                'NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
print(type(id2word), " :len: ", len(id2word))

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print("Corpus 0:\n", corpus[:1])
print("Corpus 1:\n", corpus[1:2])

# Human readable format of corpus (term-frequency)
print("\n Human readable form\n")
[[print((id2word[id], freq)) for id, freq in cp] for cp in corpus[:1]]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
print("Doc Lda\n", doc_lda)

print('\n Perplexity: ', lda_model.log_perplexity(corpus))
coherence_model_lda = CoherenceModel(
    model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print("\n Coherence score: ", coherence_lda)

# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.display(vis)
