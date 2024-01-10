#python -m spacy download en_core_web_lg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import en_core_web_md
import csv
import regex as re
import os
import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from gensim import corpora

path = './DataUCSB/'
list_of_files = []

for root, dirs, files in os.walk(path):
    for file in files:
        list_of_files.append(os.path.join(root,file))

len(list_of_files)


filepath = './DataUCSB/address-before-joint-session-the-congress-the-state-the-union-16.csv'
speeches = []
for file in list_of_files:
    with open(filepath, 'r') as read_obj: # read csv file as a list of lists
      csv_reader = csv.reader(read_obj) # pass the file object to reader() to get the reader object
      speechList = sum(list(csv_reader), []) # Pass reader object to list() to get a list of lists (matrix)
                                            # sum(list, []) flattens 2D matrix into a vector
      speech = ''.join(speechList)
    speeches.append(speech)
    
# sm - 12MB, md - 33MB, lg - 400MB
#Load SpaCy English Model
nlp = en_core_web_md.load()

#Tags to remove
extags = ['PRON','CCONJ','PUNCT','PART','DET','ADP','NUM','SYM','SPACE']

tokens=[]
#SpaCy tokenization + lemmatization + lowercase
for speech in nlp.pipe(speeches):
    scr_tok = [token.lemma_.lower() for token in speech if token.pos_ not in extags and not token.is_stop and token.is_alpha]
    tokens.append(scr_tok)

dictionary = Dictionary(tokens)
corpus = [dictionary.doc2bow(speech) for speech in speeches]
