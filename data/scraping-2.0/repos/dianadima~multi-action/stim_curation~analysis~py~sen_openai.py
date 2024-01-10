#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:01:52 2023

@author: dianadima
"""

import openai, numpy as np
from pandas import read_csv
from scipy.io import savemat
from parse_sentences import parse_sentences

#CHANGE ME: set API key to use for OpenAI embeddings
openai.api_key = []; 

sentences = read_csv("./sentences.csv",header=None)
s_list = parse_sentences(sentences)

#get ada embeddings - current best

emb_ada = []

for s in s_list:

    resp = openai.Embedding.create(
        input=[s],
        engine="text-embedding-ada-002")
        
    emb = resp['data'][0]['embedding']
    emb = np.array(emb)
    emb_ada.append(emb)
    
emb_ada = np.array(emb_ada)
savemat('./gpt_ada.mat',{'emb_ada':emb_ada})

#get davinci embeddings - deprecated

emb_davinci = []

for s in s_list:

    resp = openai.Embedding.create(
        input=[s],
        engine="text-similarity-davinci-001")
        
    emb = resp['data'][0]['embedding']
    emb = np.array(emb)
    emb_davinci.append(emb)
    
emb_davinci = np.array(emb_davinci)
savemat('./gpt_davinci.mat',{'emb_davinci':emb_davinci})


