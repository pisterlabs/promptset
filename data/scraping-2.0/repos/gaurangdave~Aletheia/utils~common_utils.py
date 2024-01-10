from tqdm.auto import tqdm
from joblib import dump, load
import pickle
import plotly.io as io
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.sklearn
import pyLDAvis
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from gensim import corpora, models
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
from pprint import pprint
import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')


# Plotting tools


# loading library
nlp = spacy.load('en_core_web_md')
# nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('merge_entities')
# nlp.add_pipe("merge_noun_chunks")
tqdm.pandas(desc="processing")


# Utility Functions for Text Cleaning
def sent_to_words(sentences):
    for sentence in tqdm(sentences):
        yield (simple_preprocess(str(sentence), deacc=True))

# function to clean html tags from text


def clean_html(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script', 'code', 'a']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)

# function to convert text to lowercase


def lower_case(text):
    return text.lower()

# function to remove line breaks


def remove_line_breaks(text):
    return re.sub(r'\n', '', text)

# function to remove punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# function to remove numbers


def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# function to remove extra spaces


def remove_extra_spaces(text):
    text = text.replace(u'\xa0', u' ')
    return text
    # return re.sub(' +', ' ', text)

# function to remove stopwords


def remove_stopwords(texts, stop_words = []):   
    preprocess_text = simple_preprocess(str(texts), deacc=True)
    word_list = [word for word in preprocess_text if word not in stop_words]
    return " ".join(word_list)
    # return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# helper function to create pos tags


def create_pos_tag(str_sent):
    return nlp(str_sent)

# function for text lemmatization using spac
##'ADJ', 'VERB'


def lemmatization(texts, allowed_postags=['PROPN', 'NOUN'], stop_words=[]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if (token.pos_ in allowed_postags and token.is_stop == False and token.text not in stop_words)])
    return texts_out


def tokenization(texts, allowed_postags=['PROPN', 'NOUN'], stop_words=[]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append(
            ["_".join(token.text.split(" ")) for token in doc if (token.pos_ in allowed_postags and token.is_stop == False and token.text not in stop_words)])
    return texts_out


def lemmatization_without_pos(texts):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc])
    return texts_out

def simple_tokenization(texts):
    """https://spacy.io/api/annotation"""
    return [nlp(text) for text in tqdm(texts)]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# helper function to create pos tags distribution


def create_pos_tags_distribution(docs=[]):
    token_distribution = {}
    is_alpha = 0
    is_stop = 0
    for doc in tqdm(docs):
        for token in doc:
            token_distribution[token.pos_] = token_distribution.get(
                token.pos_, 0) + 1
            if (token.is_alpha):
                is_alpha += 1
            if (token.is_stop):
                is_stop += 1
    return token_distribution, is_alpha, is_stop


# function to create n-grams from noun chunks
def create_noun_chunk_ngrams(docs):
    n_gram_docs = []
    for doc in docs:
        doc_text = doc.text
        for chunk in doc.noun_chunks:
            chunk_n_gram = "_".join(chunk.text.split(" "))
            doc_text = doc_text.replace(chunk.text, chunk_n_gram)
        n_gram_docs.append(doc_text.split(" "))
    return n_gram_docs


def lemmatization_noun_chunks(texts):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if (
            ("_" in token.text) or  # if the token is a noun chunk allow that
            # if the token is a noun or proper noun allow that
            (token.pos_ in ['NOUN', 'PROPN']
             and token.is_alpha and token.is_stop == False)
        )])
    return texts_out