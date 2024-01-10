#!/usr/bin/env python3
# coding: utf-8

__author__ = "David Pacheco Aznar"
__email__ = "david.marketmodels@gmail.com"

# The aim of this script is to build a topic modeller using HDP and BERT.

from src.nlp_utils.preprocessing import install

# Data manipulation
import numpy as np
from collections import Counter

# sklearn 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score  # get sense of topic separation

# umap
install('umap-learn')
import umap.umap_ as umap

# plotting imports
import matplotlib.pyplot as plt

# gensim
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

# Hierarchical Dirichlet Process model to infer best number of LDA clusters
from gensim.models import HdpModel
from gensim.models import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary

# ### Transformers with BERT
# install('sentence-transformers')
# from sentence_transformers import SentenceTransformer
# model_bert = SentenceTransformer('bert-base-nli-max-tokens')


# ##############################################################################
# ####################### TOPIC MODELLING FUNCTIONS ############################
# ##############################################################################

# ################################# HDP ########################################

def train_hdp(corpus):
    id2word = corpora.Dictionary(corpus)
    new_corpus = [id2word.doc2bow(text) for text in corpus]
    hdp = HdpModel(corpus=new_corpus, id2word=id2word)
    return hdp, new_corpus


# ############################# DOCUMENT TOPIC #################################

def get_document_topic_lda(model, corpus, k):
    n_doc = len(corpus)
    doc_topic_mapping = np.zeros((n_doc, k))
    for i in range(n_doc):
        for topic, prob in model.get_document_topics(corpus[i]):
            doc_topic_mapping[i, topic] = prob
    return doc_topic_mapping


def lda_main_topic(lda, corpus):
    labels_lda = []
    for line in corpus:
        line_labels = sorted(lda.get_document_topics(line), key=lambda x: x[1], reverse=True)
        top_topic = line_labels[0][0]
        labels_lda.append(top_topic)
    return labels_lda

# ############################## CLUSTERING ####################################

def predict_topics_with_kmeans(embeddings,num_topics):
  kmeans_model = KMeans(num_topics)
  kmeans_model.fit(embeddings)
  topics_labels = kmeans_model.predict(embeddings)
  return topics_labels


def reduce_umap(embedding):
  reducer = umap.UMAP() #umap.UMAP()
  embedding_umap = reducer.fit_transform( embedding  )
  return embedding_umap


def reduce_pca(embedding):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform( embedding )
    print( "pca explained_variance_ ",pca.explained_variance_)
    print( "pca explained_variance_ratio_ ",pca.explained_variance_ratio_)
    
    return reduced


def reduce_tsne(embedding):
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform( embedding )
    
    return reduced


# ############################## PLOTTING ######################################

def plot_embeddings(embedding, labels,title):

    labels = np.array( labels )
    distinct_labels =  set( labels )
    
    n = len(embedding)
    counter = Counter(labels)
    for i in range(len( distinct_labels )):
        ratio = (counter[i] / n )* 100
        cluster_label = f"cluster {i}: { round(ratio,2)}"
        x = embedding[:, 0][labels == i]
        y = embedding[:, 1][labels == i]
        plt.plot(x, y, '.', alpha=0.4, label= cluster_label)
    # plt.legend(title="Topic",loc = 'upper left', bbox_to_anchor=(1.01,1))
    plt.title(title)
    


