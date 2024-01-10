#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:28:38 2023

@author: dianadima
"""

import openai, numpy as np
from pandas import read_csv
from scipy.io import savemat
from parse_sentence_parts import parse_sentence_parts

#CHANGE ME: set API key to use for OpenAI embeddings
openai.api_key = []; 

sentences = read_csv("./sentences.csv",header=None)
s_age,s_act,s_con = parse_sentence_parts(sentences)

#get ada embeddings - current best

emb_ada = []
for s in s_age:

    resp = openai.Embedding.create(
        input=[s],
        engine="text-embedding-ada-002")
        
    emb = resp['data'][0]['embedding']
    emb = np.array(emb)
    emb_ada.append(emb)
    
emb_ada = np.array(emb_ada)
savemat('./gpt_ada_agent.mat',{'emb_ada':emb_ada})

emb_ada = []
for s in s_act:

    resp = openai.Embedding.create(
        input=[s],
        engine="text-embedding-ada-002")
        
    emb = resp['data'][0]['embedding']
    emb = np.array(emb)
    emb_ada.append(emb)
    
emb_ada = np.array(emb_ada)
savemat('./gpt_ada_action.mat',{'emb_ada':emb_ada})

emb_ada = []
for s in s_con:

    resp = openai.Embedding.create(
        input=[s],
        engine="text-embedding-ada-002")
        
    emb = resp['data'][0]['embedding']
    emb = np.array(emb)
    emb_ada.append(emb)
    
emb_ada = np.array(emb_ada)
savemat('./gpt_ada_context.mat',{'emb_ada':emb_ada})


