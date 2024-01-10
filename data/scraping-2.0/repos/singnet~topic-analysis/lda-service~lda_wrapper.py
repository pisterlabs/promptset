__author__ = 'eyob'
# Tested on python3.6


import psutil
print('===================ram used at program start:',float(list(psutil.virtual_memory())[3])/1073741824.0,'GB')

import os
import sys
import pathlib
import csv
import random
import datetime
import time
import json
import logging

import re
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/plsa-service/plsa')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[0])+'/plsa-service/preprocessing')
# sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[1])+'/plsa-service/plsa')
sys.path.append(str(pathlib.Path(os.path.abspath('')).parents[1])+'/topic-analysis/plsa-service/preprocessing')

# import example_plsa as pplsa
import cleansing as pclean
import porter_dictionary

class LDA_wrapper:

    def __init__(self, docs,local=False):

        self.docs = docs
        if not local:
            self.root_path = str(pathlib.Path(os.path.abspath('')).parents[0]) + '/appData/lda/'
        else:
            self.root_path = str(pathlib.Path(os.path.abspath('')).parents[1]) + '/appData/lda/'
        print('>>>>>>>>>>>>>self.root_path>>>>>>>>>>>')
        print(self.root_path)
        self.extracted_folder = self.root_path + 'extracted/'
        self.file_dict = self.root_path + 'dict/'
        self.source_texts = self.root_path + 'extracted/'
        self.output_dir = self.root_path + 'cleaned/'
        print(self.output_dir)
        self.folder = self.root_path + 'cleaned/'
        self.dict_path = self.root_path + 'dict/'
        self.lda_parameters_path = self.root_path + 'lda-parameters/'
        self.LDA_PARAMETERS_PATH = ''

        # self.messages
        self.unique_folder_naming = None
        self.num_topics = None
        self.topic_divider = None
        self.max_iter = None

    def __del__(self):

        # Close db connections
        pass



    def write_to_json(self):



        # self.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
        print(self.unique_folder_naming)

        os.mkdir(self.extracted_folder+self.unique_folder_naming)

        contents_dict = {}

        file = self.extracted_folder + self.unique_folder_naming + 'extracted' + '.json'

        for i in range(len(self.docs)):
            contents_dict[str(i)] = self.docs[i]

        with open(file, "w") as f:
            json.dump(contents_dict, f, indent=4)

        print("len(contents_dict):",len(contents_dict))



    def generate_topics_gensim(self,num_topics, passes, chunksize,
                               update_every=0, alpha='auto', eta='auto', decay=0.5, offset=1.0, eval_every=1,
                               iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None,
                               minimum_phi_value=0.01, per_word_topics=True, callbacks=None):

        start_time_1 = time.time()

        pclean.file_dict = self.file_dict + self.unique_folder_naming[:-1] + '_dict'
        pclean.source_texts = self.source_texts + self.unique_folder_naming + 'extracted.json'
        pclean.output_dir = self.output_dir + self.unique_folder_naming

        os.mkdir(pclean.output_dir)

        # Do cleansing on the data and turing it to bad-of-words model

        with open(self.lda_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
            f.write('Preprocessing started.')

        pclean.pre_pro()

        with open(self.lda_parameters_path + self.unique_folder_naming + 'status.txt', 'w') as f:
            f.write('Preprocessing finished. Topic analysis started.')

        with open(pclean.output_dir+'cleaned.json', "r") as read_file:
            ret = json.load(read_file)

        data_lemmatized = []

        for k in ret:
            data_lemmatized.append(ret[k].splitlines())

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # View
        # print(corpus[0:1])
        # print(id2word[1])

        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=random_state,
                                                    update_every=update_every,
                                                    chunksize=chunksize,
                                                    passes=passes,
                                                    alpha=alpha,
                                                    eta=eta,
                                                    per_word_topics=per_word_topics,
                                                    decay=decay,
                                                    offset=offset,
                                                    eval_every=eval_every,
                                                    iterations=iterations,
                                                    gamma_threshold=gamma_threshold,
                                                    minimum_probability=minimum_probability,
                                                    minimum_phi_value=minimum_phi_value,
                                                    callbacks=callbacks)

        port_dict = porter_dictionary.porter_dictionary()

        topics = self.lda_model.show_topics(num_topics=num_topics,num_words=300,formatted=False)

        extracted_topics = []

        for topic in topics:
            a_topic = []
            for item in topic[1]:
                a_topic.append(item[0])
            extracted_topics.append(a_topic)

        port_dict.load_dict(self.dict_path + self.unique_folder_naming[:-1] + '_dict')


        self.topics_destemmed = []

        for i in extracted_topics:
            destemmed = []
            for j in i:
                try:
                    destemmed.append(port_dict.dictionary[j][0])
                except:
                    logging.exception('message')
            self.topics_destemmed.append(destemmed)

        '''
        Seems remaining code is to extract any produced parameters from the resulting lda model, like the weights. We need to define the proto formats of course
        for all the returned parameters
        
        also code that writes the final status that shows total running time that elapsed
        
        in general, compare the outputs of plsa and as much as possible try to apply it to the results that are returned by lda
        
        max 300 words returned in p(w|z) and in list of topics
        
        p(w|z) should be sorted by each topic independently
        
        the list of topics should appear in the order of  the p(w|z) mentioned above
        
        topic by doc matric should be joint
        
        word by topic matrix should be joint if possible
        '''








def run_lda():

    docs = []
    s = LDA_wrapper(docs, local=True)

    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_2.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_singnet_all.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_bio_all.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_hersheys_all.json'
    # path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_hr_all.json'
    path = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/extracted_all.json'

    docs = []


    with open(path, "r") as read_file:
        fileList = json.load(read_file)

    for k in fileList:
        docs.append(fileList[k])

    s = LDA_wrapper(docs,local=True)
    # s.topic_divider = 0
    # s.num_topics = 2
    # s.max_iter = 22
    # s.beta = 1
    s.unique_folder_naming = str(datetime.datetime.now()).replace(':','-').replace('.','-') + '^' + str(random.randint(100000000000, 999999999999)) + '/'
    os.mkdir(str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/lda/lda-parameters/'+s.unique_folder_naming)
    s.write_to_json()
    # s.generate_topics_gensim(num_topics=3,passes=22,chunksize=200)
    s.generate_topics_gensim(num_topics=70,passes=22,chunksize=20000)
    # s.generate_topics_gensim(num_topics=2,passes=22,chunksize=200)
    # s.generate_topics_gensim(num_topics=2,passes=100,chunksize=200,random_state=2)


    # pprint(s.lda_model.print_topics(3,50))
    # topics = s.lda_model.show_topics(2,5,formatted=False)
    # print(topics)
    print_two_d(s.topics_destemmed)


    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/singnet_all_plsa_topics_2.txt'
    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/hersheys_all_plsa_topics.txt'
    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/bio_all_plsa_topics.txt'

    # topics_snet_all_plsa_file = str(pathlib.Path(os.path.abspath('')).parents[1])+'/appData/misc/topics/hr_all_plsa_topics.txt'
    # with open(topics_snet_all_plsa_file,'r') as f:
    #     temp_list = f.readlines()
    #     topics_snet_all_plsa = []
    #     for l in temp_list:
    #         topics_snet_all_plsa.append(l.split(','))
    #
    #     for i in range(len(topics_snet_all_plsa)):
    #         for j in range(len(topics_snet_all_plsa[0])):
    #             topics_snet_all_plsa[i][j] = topics_snet_all_plsa[i][j].strip()
    #
    # topics_snet_all_plsa_file_2 = str(pathlib.Path(os.path.abspath('')).parents[1]) + '/appData/misc/topics/hr_all_plsa_topics_2.txt'
    # with open(topics_snet_all_plsa_file_2, 'r') as f:
    #     temp_list = f.readlines()
    #     topics_snet_all_plsa_2 = []
    #     for l in temp_list:
    #         topics_snet_all_plsa_2.append(l.split(','))
    #
    #     for i in range(len(topics_snet_all_plsa_2)):
    #         for j in range(len(topics_snet_all_plsa_2[0])):
    #             topics_snet_all_plsa_2[i][j] = topics_snet_all_plsa_2[i][j].strip()



    # two topics
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[0],depth=30))
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[1],depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[1],s.topics_destemmed[0],depth=30))
    # print(dot_product(topics_snet_all_plsa[1],s.topics_destemmed[1],depth=30))
    # two topics

    # three topics
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[0],depth=30))
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[1],depth=30))
    # print(dot_product(topics_snet_all_plsa[0],s.topics_destemmed[2],depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[1], s.topics_destemmed[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], s.topics_destemmed[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], s.topics_destemmed[2], depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[2], s.topics_destemmed[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], s.topics_destemmed[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], s.topics_destemmed[2], depth=30))
    # print('=========================')
    # three topics

    # plsa self
    # print(dot_product(topics_snet_all_plsa[0], topics_snet_all_plsa_2[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[0], topics_snet_all_plsa_2[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[0], topics_snet_all_plsa_2[2], depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[1], topics_snet_all_plsa_2[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], topics_snet_all_plsa_2[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[1], topics_snet_all_plsa_2[2], depth=30))
    # print('=========================')
    # print(dot_product(topics_snet_all_plsa[2], topics_snet_all_plsa_2[0], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], topics_snet_all_plsa_2[1], depth=30))
    # print(dot_product(topics_snet_all_plsa[2], topics_snet_all_plsa_2[2], depth=30))
    # print('=========================')
    # plsa self


def dot_product(list_1,list_2,depth=30):

    count = 0
    for i in list_1[0:depth]:
        if i in list_2[0:depth]:
            count = count + 1
    return count

def print_two_d(two_d):
    for i in two_d:
        print(i)




__end__ = '__end__'


if __name__ == '__main__':

    run_lda()

    pass