
import os
import requests
import openai
import json
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize.punkt import PunktSentenceTokenizer
from utils.lexrank import degree_centrality_scores
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from keybert import KeyBERT
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import pandas as pd

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import streamlit as st

st.session_state['OPENAI_API_BASE'] ="https://api.endpoints.anyscale.com/v1"

prompt = lambda TITLES, KEYWORDS: f"""
I have topic that contains documents with the following titles: {TITLES}.
The topic is described by the following keywords: {KEYWORDS}.
Based on the above information, can you give a short label of the topic? Use the following format ONLY: "Label: <label>".
"""     
class ResearchCorpus():
    """
    Class to hold a corpus of research papers
    
    Attributes:
        file_dict (dict): A dictionary containing information about the research papers.
        is_seed (list): A list indicating whether each paper is a seed or not.
        corpus_embeddings (numpy.ndarray): An array of embeddings for the title and abstract of each paper.
        corpus_embeddings_norm (numpy.ndarray): An array of normalized embeddings for the title and abstract of each paper.
        clustering_model (AgglomerativeClustering): A clustering model used to cluster the papers based on their embeddings.
        cluster_assignment (numpy.ndarray): An array indicating the cluster assignment for each paper.
        clustered_sentences (dict): A dictionary containing the papers grouped by cluster.
        size_cluster (list): A list containing the number of papers in each cluster.
        labels (dict_keys): The unique labels of the clusters.
        titles (list): A list of the titles of the papers in each cluster.
        label_topics (dict): A dictionary containing the topics generated for each cluster.
        prompt (str): The prompt used for generating topics.
    """
    def __init__(self, title_abstracts, is_seed, file_dict):
        self.file_dict = file_dict
        self.is_seed = is_seed
        
        model = SentenceTransformer('allenai-specter')
        self.corpus_embeddings = model.encode(title_abstracts, convert_to_numpy=True)
        self.corpus_embeddings_norm = self.corpus_embeddings / np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6) #, affinity='cosine', linkage='average', distance_threshold=0.4)
        self.clustering_model.fit(self.corpus_embeddings_norm)
        self.cluster_assignment = self.clustering_model.labels_
        self.clustered_sentences = {}
        pbar2 = tqdm(total=len(self.cluster_assignment), desc='Assign clusters')
        for sentence_id, cluster_id in enumerate(self.cluster_assignment):
            if cluster_id not in self.clustered_sentences:
                self.clustered_sentences[cluster_id] = []
            pap = title_abstracts[sentence_id].split('[SEP]')
            abstract = pap[1]
            title = pap[0]
            full = {'title': title, 'abstract': abstract, 'full': title_abstracts[sentence_id], 'is_seed': is_seed[sentence_id],
                    'paperId': file_dict[sentence_id]['paperId'], 'authors': file_dict[sentence_id]['authors'], 'year': file_dict[sentence_id]['year'], 'url': file_dict[sentence_id]['url']}
            
            
            self.clustered_sentences[cluster_id].append(full)
            pbar2.update(1)
        pbar2.close()
        self.size_cluster = [len(self.clustered_sentences[c]) for c in self.clustered_sentences]
        self.labels = self.clustered_sentences.keys()
        self.titles = [self.clustered_sentences[c][0]['title'] for c in self.clustered_sentences]
        self.label_topics = None
        self.prompt = prompt
        self._get_topics()

    def _get_topics(self):
        """
        Generate keywords and topics for each cluster.
        """
        label_keywords = {}
        kw_model = KeyBERT()
        pbar3 = tqdm(total=len(self.labels), desc='Get Keywords for {} clusters'.format(len(self.labels)))
        for label in self.labels:
            docs ="\t".join([self.clustered_sentences[label][i]['full'] for i in range(len(self.clustered_sentences[label]))])
            titles = [self.clustered_sentences[label][i]['title'] for i in range(len(self.clustered_sentences[label]))]
            keywords = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 2), stop_words='english', nr_candidates=30, top_n=15)
            label_keywords[label] = {'keywords': keywords, 'titles': titles}
            pbar3.update(1)
        pbar3.close()
        # print keywords
        for label in self.labels:
            print('Label: ', label)
            print('Keywords: ', label_keywords[label]['keywords'])
            print('')
        self.label_topics = {}
        pbar4 = tqdm(total=len(self.labels), desc='Get Topics for {} clusters'.format(len(self.labels)))
        for label in self.labels:
            t_prompt = self.prompt(label_keywords[label]['titles'], label_keywords[label]['keywords'])
            self.label_topics[label] = self._llma_chat(t_prompt)
            pbar4.update(1)
        pbar4.close()
        
    @staticmethod         
    def _llma_chat(prompt):
        """
        Perform a chat completion using the Llama-2-70b model.
        
        Args:
            prompt (str): The prompt for the chat completion.
            
        Returns:
            str: The response from the chat completion.
        """
        s = requests.Session()

        api_base = st.session_state['OPENAI_API_BASE']
        token = st.secrets['OPENAI_API_KEY']
        url = f"{api_base}/chat/completions"
        body = {
        "model": "meta-llama/Llama-2-70b-chat-hf",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        "temperature": 0
        }

        with s.post(url, headers={"Authorization": f"Bearer {token}"}, json=body) as resp:
            response = resp.json()
        return response['choices'][0]['message']['content']
    def _make_df_cluster_topics(self):
        """
        make a dataframe with title, abstract, cluster_id, topic
        """
        frame = []
        id = 0
        for label in self.labels:
            for i in range(len(self.clustered_sentences[label])):
                frame.append({'title': self.clustered_sentences[label][i]['title'], 'abstract': self.clustered_sentences[label][i]['abstract'], 'title_abstract': self.clustered_sentences[label][i]['full'],
                              'cluster_id': label, 'topic': self.label_topics[label], 'ID': id, 'is_seed': self.clustered_sentences[label][i]['is_seed'],
                              'paperId': self.clustered_sentences[label][i]['paperId'], 'authors': self.clustered_sentences[label][i]['authors'], 'year': self.clustered_sentences[label][i]['year'], 'url': self.clustered_sentences[label][i]['url']})
                id += 1
        

        df = pd.DataFrame(frame)
        df['Topic'] = df['cluster_id'].apply(lambda x: self.label_topics[x].strip().split('Label: ')[-1])
        return df
    def make_graphs(self, df, t=0):
        """
        Create graphs based on the dataframe of papers.
        
        Args:
            df (pandas.DataFrame): The dataframe containing the paper information.
            t (int, optional): The threshold for centrality scores. Defaults to 0.
            
        Returns:
            dict, pandas.DataFrame: The graphs and a copy of the dataframe with additional information.
        """
        clusters = df['Topic'].tolist()
        articles = df['title_abstract'].tolist()
        IDs = df['ID'].tolist()
        titles = df['title'].tolist()
        is_seed = df['is_seed'].tolist()
        edges, nodes, col_dict = self._get_centrality_graph(articles, IDs, titles, clusters, is_seed, t=t)
        graphs= {'nodes': nodes, 'edges': edges}
        df['topic_color'] = df['ID'].map(col_dict)
        return graphs, df.copy()

    @staticmethod
    def _get_centrality_graph(articles, IDs, titles, topics, is_seed, t=0):
        """
        Compute centrality scores and create a graph based on the scores.
        
        Args:
            articles (list): A list of the title and abstract of each paper.
            IDs (list): A list of IDs for each paper.
            titles (list): A list of the titles of the papers.
            topics (list): A list of the topics for each paper.
            is_seed (list): A list indicating whether each paper is a seed or not.
            t (int, optional): The threshold for centrality scores. Defaults to 0.
            
        Returns:
            list, list, dict: The edges, nodes, and color dictionary for the graph.
        """
        #Compute the sentence embeddings
        cos_scores = get_sim_matrix(articles, model_name='allenai-specter')
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
        for i in range(len(articles)): # get relevant scores(diagonal matrix)
            for j in range(len(articles)):
                if i<=j:
                    cos_scores[i, j] = False
        indices = np.argwhere(cos_scores >= 0.75)
        q1 = np.quantile(centrality_scores, 0.25)
        q3 = np.quantile(centrality_scores, 0.75)
        colors = [map_to_color(topics, t, iss) for iss, t in zip(is_seed, topics)]
        cent_dict = {t: str(round(s, 3)) for t, s in zip(IDs, centrality_scores)}
        title_dict = {t: i for t, i in zip(IDs, titles)}
        is_seed_dict = {t: i for t, i in zip(IDs, is_seed)}
        topics_dict = {t: i for t, i in zip(IDs, topics)}
        col_dict = {t: s for t, s in zip(IDs, colors)}

        G = nx.Graph()
        for index in indices:
            if index[0] != index[1]:
                G.add_edge(IDs[index[0]], IDs[index[1]], weight=float(round(cos_scores[index[0], index[1]], 3)))
            
        # positions = nx.kamada_kawai_layout(G, dim=2, center=[-5, 5])
        # nx.set_node_attributes(G, name='position', values=positions)
        nx.set_node_attributes(G, name='centrality', values=cent_dict)
        nx.set_node_attributes(G, name='faveColor', values=col_dict)
        nx.set_node_attributes(G, name='title', values=title_dict)
        nx.set_node_attributes(G, name='is_seed', values=is_seed_dict)
        nx.set_node_attributes(G, name='topic', values=topics_dict)
        cyto = nx.readwrite.json_graph.cytoscape_data(G)
        elements = cyto['elements']
        
        edges  = elements['edges']
        
        nodes = [
            {
                "data": {"id": int(data['data']['id']), "label": data['data']['title'], "centrality": data['data']['centrality'], "faveColor": data['data']['faveColor'], "is_seed": data['data']['is_seed'], "topic": data['data']['topic'].replace(',', '-').strip()},
                # "position": {"x": int(round(abs(data['data']['position'][0]), 4)), "y": int(round(abs(data['data']['position'][1]), 4))}
            }
            for data in elements['nodes']
        ]
        elements = {'nodes': nodes, 'edges': edges}
        return edges, nodes, col_dict
         
# helper function to get the top n most similar articles
def map_to_color(topicslist, topic, is_seed):
    color_options = ['#eb3dd6', '#32f1e7', '#FF7F50', '#90bbd4', '#b7edca', '#f5917d',
                     '#8FBC8F', '#778899', '#FF69B4', '#FF6347']
    seed_color = '#e8da15'
    if is_seed:
        return seed_color
    # check if len(topicslist) > len(color_options)
    if len(topicslist) > len(color_options):
        # add more colors
        import random
        for i in range(len(topicslist) - len(color_options)):
            color_options.append('#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    # get index of topic
    idx = topicslist.index(topic)
    # return color
    return color_options[idx]
    
def get_sim_matrix(articles, model_name='allenai-specter'):
    model = SentenceTransformer(model_name)
    print("Compute the sentence embeddings")
    embeddings = model.encode(articles, convert_to_tensor=True, show_progress_bar=True)

    #Compute the pair-wise cosine similarities
    cos_scores = util.cos_sim(embeddings, embeddings).numpy()
    return cos_scores