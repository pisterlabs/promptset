from .base import BaseTopicSystem
from models.model.smtopic import SMTopic
import pandas as pd
# from simcse import SimCSE
import gensim.corpora as corpora
from sklearn.cluster import KMeans
import numpy as np
from evaluation.recall_at_k import *

import umap
import hdbscan

# from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel


class SMTopicTM(BaseTopicSystem):
    def __init__(self, dataset, topic_model, num_topics, dim_size, word_select_method, embedding, seed, test_path):
        super().__init__(dataset, topic_model, num_topics)
        print(f'Initialize SMTopicTM with num_topics={num_topics}, embedding={embedding}')
        self.dim_size = dim_size
        self.word_select_method = word_select_method
        self.embedding = embedding
        self.seed = seed
        if test_path is not None:
            self.test_data = pd.read_csv(test_path)
        # make sentences and token_lists
        token_lists = self.dataset.get_corpus()
        self.sentences = [' '.join(text_list) for text_list in token_lists]
        
        # embedding_model = TransformerDocumentEmbeddings(embedding)
        self.model_topic = SMTopic(embedding_model=self.embedding,
                             nr_topics=num_topics, 
                             dim_size=self.dim_size, 
                             word_select_method=self.word_select_method, 
                             seed=self.seed)
        
    
    
    
    def train_cluster(self):
        self.topics = self.model_topic.fit_transform(documents=self.sentences, embeddings=None, cluster=True)
    
    def train_embeddings(self):
        self.model_topic.fit_transform(documents=self.sentences, embeddings=None, cluster=False)
    
    def get_embed_matrix(self):
        return self.model_topic._get_embeddings()
    
    def evaluate_embedding_model(self, cluster='kmeans', size_test=2000):
        embed_matrix = self.get_embed_matrix()
        up = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', n_jobs=-1).fit(embed_matrix)
        umap_embeddings = up.transform(embed_matrix)

        if cluster =='kmeans':
            cluster_model = KMeans(n_clusters=10, random_state=42)  # You can set the number of clusters as per your requirement
            cluster_model.fit(umap_embeddings)
        elif cluster =='hdbscan':
            cluster_model = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
        
        results = get_recall_at_k_parallel(self.test_data, cluster_model.labels_, embed_matrix, size=size_test, k_list=[5,10,50])
        
        return results


    def evaluate_topic_model(self):
        td_score = self._calculate_topic_diversity()
        cv_score, npmi_score = self._calculate_cv_npmi(self.sentences, self.topics)
        
        return td_score, cv_score, npmi_score
    
    
    def get_topics(self):
        return self.model_topic.get_topics()
    
    
    def _calculate_topic_diversity(self):
        topic_keywords = self.model_topic.get_topics()

        bertopic_topics = []
        for k,v in topic_keywords.items():
            temp = []
            for tup in v:
                temp.append(tup[0])
            bertopic_topics.append(temp)  

        unique_words = set()
        for topic in bertopic_topics:
            unique_words = unique_words.union(set(topic[:10]))
        td = len(unique_words) / (10 * len(bertopic_topics))

        return td


    def _calculate_cv_npmi(self, docs, topics): 

        doc = pd.DataFrame({"Document": docs,
                        "ID": range(len(docs)),
                        "Topic": topics})
        documents_per_topic = doc.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.model_topic._preprocess_text(documents_per_topic.Document.values)

        vectorizer = self.model_topic.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in self.model_topic.get_topic(topic)] 
                    for topic in range(len(set(topics))-1)]

        coherence_model = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_v')
        cv_coherence = coherence_model.get_coherence()

        coherence_model_npmi = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_npmi')
        npmi_coherence = coherence_model_npmi.get_coherence()

        return cv_coherence, npmi_coherence 
