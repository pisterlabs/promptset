import json
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim import models
import numpy as np
import pandas as pd


def loaddata_csv(filepath):
    data = pd.read_csv(filepath, encoding='utf-8', header=None)
    data = data.drop(index=[0])
    data = data.drop(columns=[0])

    all = []
    for text in data[1]:
        all.append(cut_text(text))
    return all


def cut_text(text):
    # 文本切为list
    text = text.replace(",", "").replace("'", "").replace("[", '').replace("]", '')
    textlist = text.split(u" ")
    return textlist


def loaddata_json(file):
    # 读取数据
    json_file = open(file, encoding='utf-8')
    data = json.load(json_file)
    all = {}
    for tag in data.keys():
        all[tag] = []
        for text in data[tag]:
            all[tag].append(cut_text(text))
    return all


def ldamodel_evl(texts):
    # 模型建立和评估
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    u_mass = dict()
    c_v = dict()

    np.random.seed(1)
    for n_topics in range(3, 10):
        print("topics:", n_topics, ':')
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=n_topics)

        cm1 = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        temp_u_mass = cm1.get_coherence()
        u_mass[n_topics] = temp_u_mass
        print("u_mass:", temp_u_mass)

        cm2 = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        temp_c_v = cm2.get_coherence()
        c_v[n_topics] = temp_c_v
        print("c_v:", temp_c_v)

    return u_mass, c_v


if __name__ == '__main__':
    # 整体数据
    filepath_pos = 'pos_comment_0.960_2.csv'  # 去掉介词，人名，地点，时间等
    filepath_neg = 'neg_comment_0.960_2.csv'  # 去掉介词，人名，地点，时间等
    pos = loaddata_csv(filepath_pos)
    neg = loaddata_csv(filepath_neg)

    whole = {}
    whole['pos'] = {}
    whole['neg'] = {}

    print('whole:\npositive:')
    whole['pos']['u_mass'], whole['pos']['c_v'] = ldamodel_evl(pos)
    print('negative:')
    whole['neg']['u_mass'], whole['neg']['c_v'] = ldamodel_evl(neg)

    with open('whole.json', 'w', encoding='utf-8') as w:
        json_w = json.dumps(whole, ensure_ascii=False)
        w.write(json_w)
        w.close()
    '''
    #分标签数据
    tag_pos_file='tag_pos_comments.json'
    tag_neg_file='tag_neg_comments.json'
    tag_pos=loaddata_json(tag_pos_file)
    tag_neg=loaddata_json(tag_neg_file)

    all_u_mass=dict()
    all_c_v=dict()


    print("positive:")
    for tag in tag_pos.keys():
        print(tag,":")
        u_mass,c_v=ldamodel_evl(tag_pos[tag])
        all_u_mass[tag]={}
        all_c_v[tag]={}
        all_u_mass[tag]['pos']=u_mass
        all_c_v[tag]['pos']=c_v
        
    print("negative")
    for tag in tag_neg.keys():
        print(tag,":")
        u_mass,c_v=ldamodel_evl(tag_neg[tag])
        all_u_mass[tag]['neg']=u_mass
        all_c_v[tag]['neg']=c_v
        
    json_u_mass=json.dumps(all_u_mass,ensure_ascii=False)
    json_c_v=json.dumps(all_c_v,ensure_ascii=False)

    with open('u_mass.json','w',encoding='utf-8') as u:
        u.write(json_u_mass)
        u.close()
        
    with open('c_v.json','w',encoding='utf-8') as c:
        c.write(json_c_v)
        c.close()

  '''
