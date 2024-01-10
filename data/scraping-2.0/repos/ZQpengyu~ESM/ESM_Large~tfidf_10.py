import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import gensim
from gensim import corpora

import numpy as np
from string import punctuation
import warnings
import re
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

def get_lemma(word):
    #dogs->dog
    #aardwolves->aardwolf'
    #sichuan->sichuan
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


col_spliter = '\t'

swords = stopwords.words('english')

def word_tokenize(sent):
    """Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

s=0
data_set=[] #建立存储分词的列表
news_id = []
PATHs= ["../MIND-large/download/train/news.tsv", "../MIND-large/download/dev/news.tsv",
        "../MIND-large/download/test/news.tsv"]
for PATH in PATHs:
    with open(PATH, 'r', encoding='utf-8') as rd:
        ni = 0
        for line in rd:
            ni+=1

            # 新闻id, 类别， 子类别， title, 摘要， url, 以及两个用不到的

            N = []
            nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                col_spliter
            )
            news_id.append(nid)
            title = word_tokenize(title.lower())
            title = [get_lemma(x) for x in title]
            ab = word_tokenize(ab.lower())
            ab = [get_lemma(x) for x in ab]
            for i in title:
                if i not in punctuation and i not in swords:
                    N.append((i))
            for i in ab:
                if i not in punctuation and i not in swords:
                    N.append((i))
            data_set.append(N)
            # if ni == 2:
            #     break






dictionary = corpora.Dictionary(data_set)  # 构建 document-term matrix
new_dict = {v: k for k, v in dictionary.token2id.items()}



num_event=10
num_show_wordss = [4,6,7,8]
for num_show_words in num_show_wordss:
    print(num_show_words)



    corpus = [dictionary.doc2bow(text) for text in data_set] #(word_id, times)
    tfidf2 = TfidfModel(corpus)
    corpus_tfidf = tfidf2[corpus]
    #  output

    news_keywords = {}
    keywords = []
    print('\nTraining by gensim tfidf model......\n')
    for i, doc in zip(news_id, corpus_tfidf):

        sorted_words = sorted(doc, key=lambda x: x[1], reverse=True)  # type=list
        if i in news_keywords:
            continue
        else:
            news_keywords[i] = [new_dict[x[0]] for x in sorted_words[:num_show_words]]
            keywords.extend(news_keywords[i])

    count = 0
    for key, value in news_keywords.items():
        if len(value)<5:
            count += 1
    print(count)
    print(len(news_keywords))
    print(count/len(news_keywords))
    keywords = list(set(keywords))
    print(len(keywords))


    from transformers import BertTokenizer, BertModel
    import torch
    from sklearn.cluster import KMeans
    import numpy as np

    tokenizer = BertTokenizer.from_pretrained('./bert-model/')
    bert = BertModel.from_pretrained('./bert-model/')

    lists = keywords
    X = np.zeros(shape=(len(lists),768))

    for i in range(len(lists)):
        a = tokenizer.tokenize(lists[i])
        input = ['[CLS]']+a+['[SEP]']
        input_id = tokenizer.convert_tokens_to_ids(input)
        input_id=torch.tensor(input_id).long().unsqueeze(0)
        output, pooler= bert(input_ids=input_id)
        cls = output[0,0,:]
        X[i,:] = cls.detach().numpy()


    y_pred = KMeans(n_clusters=num_event, random_state=9).fit_predict(X)

    word2event = {}
    for i in range(len(lists)):
        word2event[lists[i]] = y_pred[i]


    # news_keywords

    news_event = {}
    for key, value in news_keywords.items():
        temp = [word2event[i] for i in value]
        temp_list = [0] * num_event
        event_list = [0] * num_event
        for i in temp:
            temp_list[i] += 1
        for i in range(len(temp_list)):
            event_list[i] = temp_list[i] / len(temp)
        news_event[key] = event_list

    import pickle
    with open(f'news2event{num_event}_{num_show_words}.pkl','wb') as fwb:
        pickle.dump(news_event, fwb)





#
# lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topic, passes=50, random_state=1)
# topic_list = lda.print_topics()
# print(topic_list)
# print('\n')
# topic_word_dict = {}
# for k in range(num_topic):
#     b = lda.print_topic(k)
#     b = b.split(' + ')
#     word_score_list = []
#     for i in b:
#         temp1 = i.split('*')
#         #print(temp1)
#         temp1[1] = eval(temp1[1])
#         word_score_list.append(temp1)
#     topic_word_dict[k] = word_score_list
#
# M = len(data_set)
#
# doc_topic_dict = {}  # key: 第i篇⽂章 value: 第i篇⽂章的主题分布
# doc_word_dict = {}  # key: 第i篇⽂章 value: 第i篇⽂章的主题所包含的词语
# new_dataset = []
# for i in range(M):
#     templist2 = []  # 临时变量，存储topici包含的词语
#     test_doc = data_set[i]  # 查看训练集中第i个样本
#     doc_bow = dictionary.doc2bow(test_doc)  # ⽂档转换成bow
#     # num_show_topic = 2  # 每个⽂档取其前2个主题
#     doc_topics = lda.get_document_topics(doc_bow, minimum_probability=1e-3)  # 某⽂档的主题分布
#     doc_topic_dict[i] = doc_topics
#     for topic in doc_topic_dict[i]:
#         temp_word = topic_word_dict[topic[0]]
#         for k in range(len(temp_word)):
#             temp_word[k][0] = float(temp_word[k][0]) * float(topic[1]) #0:p,1:w
#         templist2 += temp_word
#     c = {}
#     for word in templist2:
#         if word[0] not in c:
#             c[word[1]] = word[0]
#         else:
#             c[word[1]] = max(c[word[1]],word[0])
#     doc_word_dict[i] = c
#
#
# for i in range(M):
#     keyword = []
#     temp = []
#     for word in data_set[i]:
#         if word in doc_word_dict[i] and word not in temp:
#             keyword.append([doc_word_dict[i][word],word])
#             temp.append(word)
#
#     keyword.sort(key=lambda x: x[0], reverse=True)
#     keyword = [x[1] for x in keyword][:num_show_words]
#     if len(keyword)==0:
#         print(data_set[i])
#     new_dataset.append(keyword)
#
#
#     # templist2.sort(key=lambda x: x[0], reverse=True)
#     # templist2 = [x[1] for x in templist2][:num_show_words]
#     # doc_word_dict[i] = templist2
#     # new_dataset.append(templist2)
#
#
# total = len(new_dataset)
# two = len([x for x in new_dataset if len(x)>0])
# three = len([x for x in new_dataset if len(x)>7])
# print(two/total)
# print(three/total)
# print('finish')
