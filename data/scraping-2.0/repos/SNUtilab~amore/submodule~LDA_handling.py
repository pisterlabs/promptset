# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 02:59:32 2021

@author: tkdgu
"""

from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
import numpy as np
import pandas as pd


def get_topic_doc(lda_model, corpus) :
    
    topic_doc_df = pd.DataFrame(columns = range(0, lda_model.num_topics))
    
    for corp in corpus :
        
        temp = lda_model.get_document_topics(corp)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
        
        topic_doc_df = topic_doc_df.append(DICT, ignore_index=1)
    topic_doc_df = np.array(topic_doc_df)
    topic_doc_df = np.nan_to_num(topic_doc_df)
    
    
    return(topic_doc_df)


def get_topic_word_matrix(lda_model) :
    
    topic_word_df = pd.DataFrame()
    
    for i in range(0, lda_model.num_topics) :
        temp = lda_model.show_topic(i, 1000)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
            
        topic_word_df = topic_word_df.append(DICT, ignore_index =1)
        
    topic_word_df = topic_word_df.transpose()
    
    return(topic_word_df)


def get_topic_topword_matrix(lda_model, num_word) :
    
    topic_word_df = pd.DataFrame()
    
    for i in range(0, lda_model.num_topics) :
        temp = lda_model.show_topic(i, num_word)
        temp = [i[0] for i in temp]
        DICT = dict(enumerate(temp))
        
        topic_word_df = topic_word_df.append(DICT, ignore_index =1)
        
    topic_word_df = topic_word_df.transpose()
    
    return(topic_word_df)


def cosine(u, v):
    return (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def get_CPC_topic_matrix(encoded_CPC, encoded_topic) :
    
    CPC_topic_matrix = pd.DataFrame(columns = range(0, encoded_topic.shape[0]), index = encoded_CPC.keys())
    
    for topic in range(0, encoded_topic.shape[0]) :
        
        for cpc in encoded_CPC.keys() :
            cpc_embedding = encoded_CPC[cpc]
            sim = cosine(encoded_topic[topic], cpc_embedding)
            
            CPC_topic_matrix[topic][cpc] =  sim
        
    
    return CPC_topic_matrix

def get_topic_novelty(CPC_topic_matrix) :
    
    result_dict = {}
    
    for topic, max_value in enumerate(CPC_topic_matrix.max()) :
        
        result_dict[topic] = 1/max_value
        
    return(result_dict)
        
    
def classifying_topic(CPC_topic_matrix, standard) :
    
    result_dict = {}
    
    for topic, max_value in enumerate(CPC_topic_matrix.max()) :
        
        if max_value <= standard :
            result_dict[topic] = 'Novel'
        else : 
            result_dict[topic] = 'Common'
            
    return(result_dict)
        
def get_topic_vol(lda_model, corpus) :

    topic_doc_df = pd.DataFrame(columns = range(0, lda_model.num_topics))
    
    for corp in corpus :
        
        temp = lda_model.get_document_topics(corp)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
        
        topic_doc_df = topic_doc_df.append(DICT, ignore_index=1)
    
    result = topic_doc_df.apply(np.sum).to_dict()
    
    return(result)
    
def get_topic_vol_time(lda_model, topic_doc_df, data_sample, time) :
    
    topic_doc_df = pd.DataFrame(topic_doc_df)
    topic_doc_df['time'] = data_sample[time]
    
    topic_time_df = pd.DataFrame()
    
    for col in range(0, lda_model.num_topics) :
        grouped = topic_doc_df[col].groupby(topic_doc_df['time'])
        print(grouped)
        DICT = grouped.sum()
        topic_time_df = topic_time_df.append(DICT, ignore_index=1)
    
    topic_time_df = topic_time_df.transpose()
    topic_time_df.index = topic_time_df.index.astype(int)
    topic_time_df = topic_time_df.sort_index()
     
    return(topic_time_df)


def get_topic_weight_time(lda_model, topic_doc_df, data_sample, time, weight, by = 'sum') :
    
    topic_doc_df = pd.DataFrame(topic_doc_df)
    topic_doc_df = topic_doc_df * data_sample[weight]
    
    topic_doc_df['time'] = data_sample[time]
    
    topic_time_df = pd.DataFrame()
    
    for col in range(0, lda_model.num_topics) :
        grouped = topic_doc_df[col].groupby(topic_doc_df[time])
        if by == 'sum' : 
            DICT = grouped.sum()
        if by == 'mean'  : 
            DICT = grouped.mean()
        topic_time_df = topic_time_df.append(DICT, ignore_index=1)
    
    topic_time_df = topic_time_df.transpose()
    topic_time_df.index = topic_time_df.index.astype(int)
    topic_time_df = topic_time_df.sort_index()
     
    return(topic_time_df)


def get_topic_CAGR(topic_time_df) :
    
    st_time = min(topic_time_df.index)
    ed_time = 2021 # 2020 fix
    
    duration = int(ed_time) - int(st_time)
    
    result = {}
    
    for col in topic_time_df :
        st_val = topic_time_df[col][0]
        ed_val = topic_time_df[col][duration]
        CAGR = (ed_val/st_val)**(1/duration) -1
        result[col] = CAGR
        
    return(result)
    

def get_topic2CPC(CPC_topic_matrix) :
    
    result_dict = {}
    
    for col in CPC_topic_matrix.columns :
        
        result_dict[col] = pd.to_numeric(CPC_topic_matrix[col]).idxmax()
    
    return(result_dict)

def get_most_similar_doc2topic(data_sample, topic_doc_df, top_n = 5, title = 'title', date = 'date') :
    
    result_df = pd.DataFrame()
    title = title
    
    for col in range(topic_doc_df.shape[1]) :
        
        DICT = {}
        
        # idx = np.argmax(topic_doc_df[:,col])
        # value = np.max(topic_doc_df[:,col])
        for n in range(1, top_n+1) :
            idx = topic_doc_df.argsort(axis = 0)[-n][col]
            DICT['topic'] = col
            DICT['rank'] = n
            DICT['title'] = data_sample[title][idx]
            DICT['date'] = data_sample[date][idx]
            DICT['similarity'] = topic_doc_df[idx,col]
        
            result_df = result_df.append(DICT, ignore_index=1)
    
    return(result_df)