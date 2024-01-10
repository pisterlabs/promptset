#!/usr/bin/env python

import numpy as np
import os
from os.path import splitext
import sys
import logging
import warnings

import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords

import asreview
from asreview.cluster import normalized_cluster_score


def compute_coherence(model, data_lemmatized, id2word):
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
    coherence_values : Coherence values corresponding to the LDA model with
        respective number of topics
    """
    coherencemodel = CoherenceModel(model=model, texts=data_lemmatized,
                                    dictionary=id2word, coherence='c_v')
    return coherencemodel.get_coherence()


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent))
#         texts_out.append(
#             [token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out
def get_one_all_dict(all_prediction, one_idx):
    unique, counts = np.unique(all_prediction, return_counts=True)
    all_dict = {unique[i]: counts[i] for i in range(len(unique))}

    prediction = all_prediction[one_idx, ]
    unique, counts = np.unique(prediction, return_counts=True)
    one_dict = {unique[i]: counts[i] for i in range(len(unique))}
    return one_dict, all_dict


def create_corpus(texts, use_bigrams=True):
    nltk.download('stopwords')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.ERROR)

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # NLTK Stop words
    stop_words = stopwords.words('english')

    # Import Dataset
    data = texts.tolist()
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
#     trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
#     trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words, stop_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    # Do lemmatization keeping only noun, adj, vb, adv
    if use_bigrams:
        data_lemmatized = data_words_bigrams
    else:
        data_lemmatized = data_words_nostops

    # print(data_lemmatized[:2])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    lemm_texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in lemm_texts]

    return corpus, id2word, data_lemmatized


def lda_clusters(data_fp, n_clusters=8, n_run=1):
    _, texts, labels = asreview.ASReviewData.from_file(data_fp).get_data()
    one_idx = np.where(labels == 1)[0]
    corpus, id2word, data_lemmatized = create_corpus(texts)
    data_name = splitext(os.path.basename(data_fp))[0] + "_lda"
    file_out = os.path.join("cluster_data", data_name, f"embed_k{n_clusters}")
    os.makedirs(os.path.join("cluster_data", data_name), exist_ok=True)

    with open(file_out, "w") as fp:
        for _ in range(n_run):
            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus, id2word=id2word, num_topics=n_clusters,
                update_every=1, chunksize=100, passes=10, alpha='auto',
                per_word_topics=True)

            clusters = []
            for doc in lda_model.get_document_topics(corpus):
                clusters.append(sorted(doc, key=lambda x: -x[1])[0][0])

            clusters = np.array(clusters, dtype=int)
            one_dict, all_dict = get_one_all_dict(clusters, one_idx)
            score = normalized_cluster_score(one_dict, all_dict)
            coherence_score = compute_coherence(lda_model, data_lemmatized,
                                                id2word)
            fp.write(f"{coherence_score} {score}\n")
            fp.flush()


if __name__ == "__main__":
    for n_clusters in [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 30]:
        lda_clusters(sys.argv[1], n_clusters=n_clusters, n_run=10)
