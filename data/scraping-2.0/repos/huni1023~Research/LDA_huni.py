# -*- coding:utf-8 -*-
'''
LDA
- Data visualization is hard to specify, using temporary jupyter notebook is better than static method
 
'''

import os, re, gensim
from datetime import date
import pandas as pd
import numpy as np

from sys import platform
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
tqdm.pandas()

from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

if 'ind' in platform: # for window
    plt.rc("font",family="Malgun Gothic")
    
elif 'darwin' in platform: # for mac
    plt.rc('font', family='AppleGothic') 
    font_path= '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    

class Build_LDA:
    def __init__(self, pd_DataFrame, col_name, tagger):
        self.pd_DataFrame = pd_DataFrame
        self.col_name = col_name
        self.tagger = tagger
    

    def txt_cleaner(self, input_text): # 준호님 코드 copy&paste
        output_text = input_text
#     output_text = output_text.replace("()","")
        output_text = re.sub('[0-9]+','NUM',output_text)  # 모든 숫자 NUM으로 대치(마스크)
        output_text = re.sub('[\U00010000-\U0010ffff]', '', output_text) # 이모지 제거
        return output_text


    def list_txt_cleaner(self, input_list_text):
        print(f'Total Document count {len(input_list_text)}')        
        output_ls = []
        for input in input_list_text:
            temp = self.txt_cleaner(input)
            output_ls.append(temp)
        return output_ls


    def data_viewer(self, idx):
        # upcoming
        return None


    def preprocessing_1(self, stop_words):
        temp = self.pd_DataFrame[self.col_name].tolist()
        output1 = self.list_txt_cleaner(temp)


        print('Tokenize using mecab, only nouns')
        def tokenize(input_txt, st):
            output = []
            for noun in self.tagger.nouns(input_txt):
                if noun not in st:
                    output.append(noun)
            return output
        
        word_data = [] # first output
        for txt in tqdm(output1):
            temp = tokenize(txt, st = stop_words)
            word_data.append(temp)
        return word_data


    def preprocessing_2(self, first_output, below_cnt, above_perc, sample = False):
        word_dict = corpora.Dictionary(first_output) # second output
        word_dict.filter_extremes(no_below=below_cnt, no_above= above_perc) # no_below option: count, no_above option: percentage
        if sample == True:
            print('\n>>>>> word dictionary generated\n')
            first10pairs = {k: word_dict.token2id[k] for k in list(word_dict.token2id)[:10]}
            last10pairs= {k: word_dict.token2id[k] for k in list(word_dict.token2id)[-10:]}

            print(f'Smallest ten key and value: {first10pairs}\n',
                    f'Bigges ten key and value: {last10pairs}')
        return word_dict


    def preprocessing_3(self, first_output, second_output, ):
        third_output = [second_output.doc2bow(doc) for doc in first_output]
        return third_output # corpus


    def model_search(self, word, dict, corpus, limit, start, step, save_folder, title):
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
        model_list = [] ; coherence_values = [] ; perplexity_values = []
        x = range(start, limit, step)
        for num_topics in tqdm(range(start, limit, step)):
            model = gensim.models.LdaMulticore(corpus=corpus, id2word= dict, 
                                                num_topics=num_topics, workers= 6)
            model_list.append(model)

    # computing coherence value
            coherence = CoherenceModel(model=model, 
                                texts=word, 
                                dictionary=dict, 
                                coherence='c_v')
            coherence_values.append(coherence.get_coherence())
            print(f'#{num_topics} cv: {round(coherence.get_coherence(),4)}') 

    # computing perplexity value
            # per_val=model.log_perplexity(corpus)
            # perplexity_values.append(per_val)

        print('\n>>>>>>Topic Training is done\n')
        topic_num = 0
        count = 0
        max_coherence = 0
        for m, cv in zip(x, coherence_values):
            coherence = cv
            if coherence >= max_coherence:
                max_coherence = coherence
                topic_num = m
                model_list_num = count   
            count = count+1

        plt.plot(x, coherence_values)
        plt.title('\n토픽개수당 coherence 점수\n')
        plt.xlabel('토픽 개수')
        plt.ylabel('Coherence 점수')
        # plt.legend()
        plt.show()

        optimal_model = model_list[model_list_num]
        
    # save model
        file_title = title + str(topic_num)
        today = date.today().isoformat()
        file_title = f'{str(today)[2:]}_' + file_title
        save_dir = os.path.join(os.getcwd(), save_folder, file_title)
        optimal_model.save(save_dir) # save 3 files
        return optimal_model



    def vis_docu(self, ldamodel, corpus, title_col):
        # upcoming
        return None



    def vis_topic(self, df_topic_sents_keywords, title_col):
        return None

