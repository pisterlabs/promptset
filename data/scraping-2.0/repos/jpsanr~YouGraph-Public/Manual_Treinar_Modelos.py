from flask import Flask, render_template,  request, jsonify
from acessos import read, get_conn, persistir_uma_linha, persistir_multiplas_linhas, replace_df
from DAOs.dao_Canal import Canal
from models.youtube_extractor import Youtube_Extractor 
from connections.mysql_connector import MySQL_Connector
from models.topic_modeling import Topic_Modeling
from connections.mongodb_connector import Mongo_Connector
from connections.neo4j_connector import Neo4j_Connector
import os
from datetime import datetime
from gensim import corpora, models, similarities
from models.graph_generator import Graph_Generator
from models.tuple_extractor import Tuple_Extractor
import pickle
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

import gensim
from gensim import corpora, models, similarities
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities

if __name__ == "__main__":
    start = 50
    limit = 80
    step= 10

    folder_name = "30-10-2020 14_39_27"


    path = os.getcwd()
    path_modelo = "{}\modelos_lda\{}".format(path, folder_name)

    print("-> Lendo Pickle")
    with (open("{}\\dict_corpus.pickle".format(path_modelo), "rb")) as openfile:
        while True:
            try:
                dict_corpus = pickle.load(openfile)
            except EOFError:
                break
            

    corpus = dict_corpus['corpus']
    id2word= dict_corpus['id2word']

    print("Start: {}".format(start)) 
    print("limit: {}".format(limit))
    print("Step: {}".format(step))       

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("Gerando novo modelo...")
    
    
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                    id2word=id2word,
                    random_state=100, 
                    num_topics=num_topics,
                    per_word_topics=True,
                    workers=3
                    )

        # model = LdaMulticore(corpus=corpus, 
        #                     id2word=id2word,
        #                     random_state=100, 
        #                     num_topics=num_topics,
        #                     per_word_topics=True,
        #                     workers=3)
        print("Novo modelo Gerado..")
        print("Calculando coerencia")
        coherencemodel = CoherenceModel(model=model, texts=corpus, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print(" --> Pronto")

        model.save("{}\#_{}".format(path_modelo,num_topics))
        
        model_list = model_list
        coherence_values = coherence_values
        print("num topics: {}".format(num_topics))
        print("*********")
        print(coherence_values)