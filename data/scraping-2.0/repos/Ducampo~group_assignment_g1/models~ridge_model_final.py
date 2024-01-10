# %matplotlib inline # in case of working with jupyter notebook.


# import required libraries

import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
import pyLDAvis
import pyLDAvis.gensim
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error




nltk.download("stopwords")
nltk.download("punkt")


# Declare document variables, make sure to update relative paths to match local file location
r_train_file = "train.json"
r_test_file = "test.json"

# raw data frames
r_train_eda_df = pd.read_json(r_train_file)
r_test_eda_df = pd.read_json(r_test_file)


# Descriptive analysis raw training dataset
print(r_train_eda_df.head())
print(r_train_eda_df.info())

# Missing data
missing_data = (r_train_eda_df.isnull().sum() / len(r_train_eda_df)) * 100
missing_ratio = pd.DataFrame({"Missing Ratio": missing_data})
print(missing_ratio)

# Data distribution based on Year (Target variable)
print(r_train_eda_df["year"].value_counts())
print(r_train_eda_df["year"].hist())


# Data cleaning and preprocessing
train_df = r_train_eda_df
test_df = r_test_eda_df

# Entrytype to categorical
train_df['ENTRYTYPE'] = pd.Categorical(train_df['ENTRYTYPE'])

# add lowercase columns of interest ['title', 'publisher', 'abstract']
train_df["title_low"] = train_df["title"].str.lower()
train_df["publisher_low"] = train_df["publisher"].str.lower()
train_df["abstract_low"] = train_df["abstract"].str.lower()

# Replacing Null values to blank for ['abstract_low'], ['publisher_low'] to be able to check the length
print(f"Original abstract col: {train_df['abstract'].isnull().sum()}\n")
train_df["abstract_low"].fillna("", inplace=True)
train_df["publisher_low"].fillna("", inplace=True)
print(
    f"After removing null values abstract_low col: {train_df['abstract_low'].isnull().sum()}"
)


# Adding columns lengths for title_low and abstract_low
train_df["title_len"] = train_df["title_low"].apply(lambda x: len(x))
train_df["abstract_len"] = train_df["abstract_low"].apply(lambda x: len(x))

# Value range for columns ['title_low','abstract_low']
print(train_df["title_len"].max())
print(train_df["title_len"].min())
print(train_df["abstract_len"].max())
print(train_df["abstract_len"].min())

train_df.info()
train_df.head(2)

"""
Similar steps for test dataset low values and len
"""

# Entrytype to categorical
test_df['ENTRYTYPE'] = pd.Categorical(test_df['ENTRYTYPE'])

# Add lowercase columns of interest ['title', 'publisher', 'abstract']
test_df["title_low"] = test_df["title"].str.lower()
test_df["publisher_low"] = test_df["publisher"].str.lower()
test_df["abstract_low"] = test_df["abstract"].str.lower()

# Replacing Null values to blank for ['abstract_low'], ['publisher_low'] to be able to check the length
print(f"Original abstract col: {test_df['abstract'].isnull().sum()}\n")
test_df["abstract_low"].fillna("", inplace=True)
test_df["publisher_low"].fillna("", inplace=True)
print(
    f"After removing null values abstract_low col: {test_df['abstract_low'].isnull().sum()}"
)

# Adding columns lengths for title_low and abstract_low
test_df["title_len"] = test_df["title_low"].apply(lambda x: len(x))
test_df["abstract_len"] = test_df["abstract_low"].apply(lambda x: len(x))

# Value range for columns ['title_low','abstract_low']
print(test_df["title_len"].max())
print(test_df["title_len"].min())
print(test_df["abstract_len"].max())
print(test_df["abstract_len"].min())


print(test_df.info())
print(test_df.head(2))


"""
Feature extraction:
Title text preprocessing to reduce dimmensionality:
    - Removing punctuation
    - Removing stopwords (using english, we are aware large number of instances are in english though we see as well other languages i.e. fr., it.)

Steps inspired from following blog:
https://www.section.io/engineering-education/using-imbalanced-learn-to-handle-imbalanced-text-data/#prerequisites

"""


# removing punctuation
def drop_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


train_df["title_pre"] = train_df["title_low"].apply(lambda x: drop_punctuation(x))
train_df["title_pre_len"] = train_df["title_pre"].apply(lambda x: len(x))
train_df["abstract_pre"] = train_df["abstract_low"].apply(lambda x: drop_punctuation(x))
train_df["abstract_pre_len"] = train_df["abstract_pre"].apply(lambda x: len(x))
# prin(train_df.head())

"""
Similar steps test dataset preprocess punctuation remove 
"""
test_df["title_pre"] = test_df["title_low"].apply(lambda x: drop_punctuation(x))
test_df["title_pre_len"] = test_df["title_pre"].apply(lambda x: len(x))
test_df["abstract_pre"] = test_df["abstract_low"].apply(lambda x: drop_punctuation(x))
test_df["abstract_pre_len"] = test_df["abstract_pre"].apply(lambda x: len(x))
# print(test_df.head(10))


# Removing stopwords under assumption that most of the text is in english. 
# (Though we are aware there are other languages present in the datasets)
def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)


train_df["title_n_stop_en"] = train_df["title_pre"].apply(lambda x: remove_stopwords(x))
train_df["title_n_stop_en_len"] = train_df["title_n_stop_en"].apply(lambda x: len(x))
train_df["abstract_n_stop_en"] = train_df["abstract_pre"].apply(
    lambda x: remove_stopwords(x)
)
train_df["abstract_n_stop_en_len"] = train_df["abstract_n_stop_en"].apply(
    lambda x: len(x)
)
print(train_df.head())

"""
Similar step test dataset n_stop_en
"""
test_df["title_n_stop_en"] = test_df["title_pre"].apply(lambda x: remove_stopwords(x))
test_df["title_n_stop_en_len"] = test_df["title_n_stop_en"].apply(lambda x: len(x))
test_df["abstract_n_stop_en"] = test_df["abstract_pre"].apply(
    lambda x: remove_stopwords(x)
)
test_df["abstract_n_stop_en_len"] = test_df["abstract_n_stop_en"].apply(
    lambda x: len(x)
)

# Feature extraction: combine lowered columns both train/test dataset
train_df["title_publisher_low"] = (
    train_df["title_low"] + " " + train_df["publisher_low"]
)
train_df["title_publisher_low_len"] = train_df["title_publisher_low"].apply(
    lambda x: len(x)
)

"""
Similar step test dataset n_stop_en
"""
test_df["title_publisher_low"] = (
    test_df["title_low"] + " " + test_df["publisher_low"]
)
test_df["title_publisher_low_len"] = test_df["title_publisher_low"].apply(
    lambda x: len(x)
)

# feature extraction: combine preprocess columns
train_df["title_publisher_pre"] = (
    train_df["title_n_stop_en"] + " " + train_df["publisher_low"]
)
train_df["title_publisher_len"] = train_df["title_publisher_pre"].apply(
    lambda x: len(x)
)

"""
Similar step test dataset n_stop_en
"""
test_df["title_publisher_pre"] = (
    test_df["title_n_stop_en"] + " " + test_df["publisher_low"]
)
test_df["title_publisher_len"] = test_df["title_publisher_pre"].apply(
    lambda x: len(x)
)


# Function to detect language using langdetect library
# In order to validate the detected language function performed on col: 'lang_tit_publ_low', 'title_low', 'abstract_low'
def detect_language(text):
    try:
        return detect(text)
    except:
        # Return 'unknown' if language detection fails
        return 'unknown'
    

# title_publisher_low lang detect both train/test dataset (Best language detection results with this column)
train_df['lang_tit_publ_low'] = train_df['title_publisher_low'].apply(detect_language)
print(f"Languages based on combination of tit+publi_low: \n{train_df['lang_tit_publ_low'].value_counts()}")
test_df['lang_tit_publ_low'] = test_df['title_publisher_low'].apply(detect_language)
print(f"Languages based on combination of tit+publi_low: \n{test_df['lang_tit_publ_low'].value_counts()}")

# title_low lang detect
train_df["languguage_detect"] = train_df['title_low'].apply(detect_language)
print(f"Languages based on title_low: \n{train_df['languguage_detect'].value_counts()}")
test_df["languguage_detect"] = test_df['title_low'].apply(detect_language)
print(f"Languages based on title_low: \n{test_df['languguage_detect'].value_counts()}")

# abstract_low lang detect
train_df["lang"] = train_df['abstract_low'].apply(detect_language)
print(f"Languages based on abstract_low: \n{train_df['lang'].value_counts()}")
test_df["lang"] = test_df['abstract_low'].apply(detect_language)
print(f"Languages based on abstract_low: \n{test_df['lang'].value_counts()}")

# count number of authors
# names in the author column follow same format "surname, name"
def count_author(input_text):
    name_pairs = [name.strip() for name in input_text.split(',')]
    num_names = int(len(name_pairs)/2)

    return num_names

train_df['author_number'] =  train_df['author'].apply(lambda x: count_author(x))
test_df['author_number'] =  test_df['author'].apply(lambda x: count_author(x))


# print(test_df.head())

# print(train_df.info())
# print(test_df.info())

# train_df.info()
train_df.to_json('poc_feature_train_eda_lan_j.json', orient='records', lines=False)

# test_df.info()
test_df.to_json('poc_feature_test_eda_lan_j.json', orient='records', lines=False)


"""
LDA Topic model done only for papers in english 
Used following videos as reference for the baseline model:
https://www.youtube.com/playlist?list=PL2VXyKi-KpYttggRATQVmgFcQst3z6OlX
github repository:
https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/03_03_lda_model_demo_bigrams_trigrams.ipynb
"""

# Baseline model done using english records from training dataset.
train_lda = train_df[train_df["lang_tit_publ_low"] == "en"]
data = train_lda["title_low"].to_list()  # LDA Model requires a list as an input
len(data)


# nlp lematization from the title_low feature (Raw function, finetune may be needed?)
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out


lemmatized_texts = lemmatization(data)
print(lemmatized_texts[0])


def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final


data_words = gen_words(lemmatized_texts)
print(data_words)

# BIGRAMS AND TRIGRAMS
bigram_phrases = gensim.models.Phrases(data_words, min_count=4, threshold=100)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)


def make_bigrams(texts):
    return [bigram[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]


data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)

print(data_bigrams_trigrams[5])


"""
Important step to remove not relevant most frequent words
"""
# TF-IDF REMOVAL


id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]
# print (corpus[0][0:20])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = []  # reinitialize to be safe. You can skip this.
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [
        id for id in bow_ids if id not in tfidf_ids
    ]  # The words with tf-idf score 0 will be missing

    new_bow = [
        b
        for b in bow
        if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf
    ]
    corpus[i] = new_bow


# LDA Baseline model 
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus[:-1],
    id2word=id2word,
    num_topics=10,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
)


"""
Viz results from lda model training
"""
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis

# Save trained model
lda_model.save("train_model.model")

# Load existing model
new_model = gensim.models.ldamodel.LdaModel.load("train_model.model")

# View 10 topics from the baseline lda model
print(new_model.print_topics())
doc_lda = new_model[corpus]
"""
Feature enrichment
"""


# Test example on unseen data
test_doc = corpus[-1]


vector = new_model[test_doc]
print(vector)


# Sort topics based on higher belonging value. (Topics are stored as follow (1, 0.023) where 1 = Topic and 0.023 grade of belonging)
def Sort(sub_li):
    sub_li.sort(key=lambda x: x[1])
    sub_li.reverse()
    return sub_li


# Function that selects the topic with the highest belonging grade
def Select_topic(sort_topic):
    return sort_topic[0][0]


new_vector = Select_topic(Sort(vector))
print(new_vector)


# Adding corpus to english subset from training dataset.
print(len(corpus))
train_lda["lda_base_corpus"] = corpus
print(corpus[0])
train_lda["lda_base_corpus"][0]

# Adding topics from lda baseline model to english subset from training dataset.
train_lda["lda_base_topics"] = doc_lda
train_lda["lda_topic"] = train_lda["lda_base_topics"].apply(
    lambda x: Select_topic(Sort(x))
)

# review topic distribution per paper.

train_lda["lda_topic"].value_counts()
train_lda.head()

# Save sample english lda model dataframe
train_lda.to_json("train_base_lda_en_topics.json", orient="records", lines=False)


"""
Model training Ridge()
"""

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('train_base_lda_en_topics.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('poc_feature_test_eda_lan_j.json'))).fillna("")
    train = train[['ENTRYTYPE',"title_low","publisher_low","abstract_low","lda_topic","lang_tit_publ_low",'author_number','year']]
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    featurizer = ColumnTransformer(
        transformers=[("title_low", TfidfVectorizer(), "title_low"),
         ("publisher_low", TfidfVectorizer(), "publisher_low"),
          ("abstract_low", TfidfVectorizer(), "abstract_low"),
          ('one_hot_en',OneHotEncoder(sparse=False),['ENTRYTYPE',"lda_topic","lang_tit_publ_low"])],
        remainder='passthrough') 

    ridge = make_pipeline(featurizer, Ridge())
    logging.info("Fitting model")
    ridge.fit(train.drop('year', axis=1), train['year'].values)
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, ridge.predict(val.drop('year', axis=1)))
    logging.info(f"Ridge regress MAE: {err}")
    #logging.info(f"Predicting on test")
    #pred = ridge.predict(test)
    #test['year'] = pred
    #logging.info("Writing prediction file")
    #test.to_json("predicted.json", orient='records', indent=2)

main()

# INFO:root:Loading training/test data
# INFO:root:Splitting validation
# INFO:root:Fitting models
# /usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
#   warnings.warn(
# INFO:root:Evaluating on validation data
# INFO:root:Ridge regress MAE: 3.909538658602481
# INFO:root:Predicting on test




