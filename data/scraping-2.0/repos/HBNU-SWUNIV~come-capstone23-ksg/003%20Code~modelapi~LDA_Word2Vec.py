import numpy as np
import json
from flask import Flask, request, jsonify


from tqdm import tqdm
import re
import pickle
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
import json

#from eunjeon import Mecab
from konlpy.tag import Mecab


from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric

import logging

import pickle
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors


mecab = Mecab('C:/mecab/share/mecab-ko-dic')

title = pd.read_json('C:/workspace/VueExpress/backend/data/board.json')
filename="C:/workspace/VueExpress/backend/data/board.json"

TOPICS_W_NUM = 10
num_topics = 20

def clean_text(text):
        text = text.replace(".", "").strip()
        text = text.replace("·", " ").strip()
        pattern = '[^ ㄱ-ㅣ가-힣|0-9]+'
        text = re.sub(pattern=pattern, repl='', string=text)
        return text

def get_nouns(tokenizer, sentence):
    tagged = tokenizer.pos(sentence)
    nouns = [s for s, t in tagged if t in ['NNG', 'NNP', 'VA', 'XR'] and len(s) >1]
    return nouns

def tokenize(df):
    tokenizer = Mecab(dicpath='C:/mecab/share/mecab-ko-dic')
    processed_data = []
    for sent in tqdm(df['contents']):
        sentence = clean_text(str(sent).replace("\n", "").strip())
        processed_data.append(get_nouns(tokenizer, sentence))
    return processed_data

def save_processed_data(processed_data):
    with open("tokenized_data_"+title, 'w', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in processed_data:
            writer.writerow(data)
            
def make_topictable_per_doc(model, corpus,df):
    topic_table = pd.DataFrame()
    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(model[corpus]):
        doc = topic_list[0] if model.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table._append(pd.Series([int(df['id'][i]), int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)

def print_topic_words(model) : 
    
    topic_list = []
    temp_topic_list = []
    for topic_id in range(num_topics): 
        topic_word_probs = model.show_topic(topic_id, TOPICS_W_NUM) #토픽아이디와 토픽별 단어개수
        temp_topic_list = []

        for topic_word, prob in topic_word_probs:
            temp_topic_list.append(topic_word)
        
        topic_list.append(temp_topic_list)
    
    return topic_list

def only_print_topice_word(model, model2, list) :
    for topic_id in range(num_topics): 
        topic_word_probs = model.show_topic(topic_id, TOPICS_W_NUM) #토픽아이디와 토픽별 단어개수
        try :
            print("Topic ID: {}".format(topic_id) + "      종합 결과 : " + str(model2.most_similar(positive= (model2.most_similar(positive=list[topic_id],topn=10) ), topn=3 )))
        except KeyError as ke:
            print("Topic ID: {}".format(topic_id))

        for topic_word, prob in topic_word_probs:
            print("\t{}\t{}".format(topic_word, prob))
        print("\n")

def topic_match_to_json(topictable, temp_total_word) :

    with open(filename, "r+" , encoding='utf-8') as file:
        data_writing = json.load(file)
        for i in range(len(data_writing)):
            
            try:
                temp_table = topictable[topictable['doc']==data_writing[i]['id']]
                temp_int = temp_table['highest_topic'].values.tolist()

                data_writing[i]['subject'] = temp_total_word[temp_int[0]]
            except:
                data_writing[i]['subject'] = 'out of vocabulary'

        
        file.seek(0)
        json.dump(data_writing, file, indent=4, ensure_ascii=False)


app = Flask(__name__)

@app.route('/topic')
def get():
    df = title
    df.columns=['id','writer','year','month','day','title','contents', 'image_tag', 'subject']
    df.dropna(how='any')
    processed_data= tokenize(df)
    
    processed_data = DataFrame(processed_data)
    processed_data[0] = processed_data[0].replace("", np.nan)
    processed_data = processed_data[processed_data[0].notnull()]
    processed_data = processed_data.values.tolist()
    processed_data2=[]
    
    for i in processed_data:
        i = list(filter(None, i))
        processed_data2.append(i)
        
    
    processed_data = processed_data2
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    dictionary = corpora.Dictionary(processed_data)
    dictionary.filter_extremes(no_below=4, no_above=0.25)
    corpus = [dictionary.doc2bow(text) for text in processed_data]

    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None

    temp = dictionary[0]
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    top_topics = model.top_topics(corpus) 
    
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics

    lda_visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False,mds='mmds')

    topictable = make_topictable_per_doc(model, corpus,df)
    topictable = topictable.reset_index()
    topictable.columns = ['id', 'doc', 'highest_topic', 'percent', 'percent_per']

    topictable[:50]


    result_topic_list = print_topic_words(model)

    loaded_model = KeyedVectors.load_word2vec_format('C:/workspace/VueExpress/modelapi/word2vec_model')

    total_topic_list = only_print_topice_word(model, loaded_model, result_topic_list)

    temp_total_word = []

    for i in range (len(result_topic_list)):
        try:
            total_word = loaded_model.most_similar(positive= (loaded_model.most_similar(positive=result_topic_list[i],negative=["광양시"],topn=10) ), topn=3 )
            print(total_word)
            temp_total_word.append(total_word[0][0] + ", " + total_word[1][0] + ", "  +total_word[2][0])
        except KeyError as ke:
            temp_total_word.append("out of vocabulary")

    topic_match_to_json(topictable, temp_total_word)
    print(total_topic_list)

    return None

if __name__ == '__main__':
     app.run(port=8097)