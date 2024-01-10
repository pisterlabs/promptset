import re
import numpy as np
import pandas as pd
from pprint import pprint

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

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nlp = spacy.load('en', disable=['parser', 'ner'])

#function to tokenize all lyrics in pd Series (removes punctuation, lowercase everything)
def tokenize_lyrics(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence).lower(), deacc=True))  # deacc=True removes punctuations

#function remove stop words
def remove_stopwords(texts, stop_words):
    return [[word for word in doc if word not in stop_words] for doc in texts]

#make bigrams
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

#make trigrams
def make_trigrams(texts, trigram_mod, bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

#lemmatize
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []

    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#function to return list of additoinal stop_words to remove
def get_stop_words(lyrics):
    stop_words = stopwords.words('english')
    #list of list of tokens
    tokenized = list(tokenize_lyrics(lyrics))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(tokenized, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[tokenized], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops_stop = remove_stopwords(tokenized, stop_words)

    # Form Trigrams
    data_words_trigrams_stop = make_trigrams(data_words_nostops_stop, trigram_mod, bigram_mod)

    #get words into one continuous list
    all_words_one_list = [word for list_of_words in data_words_trigrams_stop for word in list_of_words]

    #count frequency of each word
    from collections import Counter
    counts = Counter(all_words_one_list)

    #add additional stopwords to remove (100 most frequently appeared words)
    stop_words_to_add = []
    for word in counts.most_common(125):
        stop_words_to_add.append(word[0])
    stop_words.extend(stop_words_to_add)

    return stop_words

def generate_corpus(lyrics):
    tokenized = list(tokenize_lyrics(lyrics))
    stop_words = get_stop_words(lyrics)

    # Remove Stop Words Again Including additional ones added
    data_words_nostops = remove_stopwords(tokenized, stop_words)

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(tokenized, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[tokenized], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Form Trigrams Again
    data_words_trigrams = make_trigrams(data_words_nostops, trigram_mod, bigram_mod)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en

    # #lemmatize
    data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    return (corpus, id2word)

def build_model(lyrics, corpus, id2word, num_topics=5, ):

    #build and train LDA Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

    #Compute Perplexity - a measure of how good the model is. lower the better
    perplexity = lda_model.log_perplexity(corpus)
    # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    print('\nPerplexity: ', perplexity)
    # print('\nCoherence Score: ', coherence_lda)
    return lda_model

def visualize_topics(model, corpus, id2word):
    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    return vis

#get list of topics for each song
def get_topicss_for_songs(model, corpus):
    topics = []
    for idx, row in enumerate(lda_model[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        topics.append(row[0][0])
    return topics

#get topics
#pprint(lda_model.print_topics())
#
#pyLDAvis.save_html(vis, '5_topics/5_topics.html')
