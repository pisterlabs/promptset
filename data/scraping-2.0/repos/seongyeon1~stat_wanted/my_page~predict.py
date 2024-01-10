from glob import glob
import networkx as nx
import operator
from openai import OpenAI

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer
from .utils import *
from my_skill.utils import *


import ast

def predict_clusters(new_data):
    # Load the models
    loaded_AutoModel = BertModel.from_pretrained('seongyeon1/wanted_bert')
    loaded_AutoTokenizer = BertTokenizer.from_pretrained('seongyeon1/wanted_bert')
    loaded_kmeans = torch.load('./model/kmeans_final')
    
    tokenized_new_data = [loaded_AutoTokenizer(sentence, return_tensors='pt', padding=True, truncation=True) for sentence in new_data]
    encoded_new_data = [loaded_AutoModel(**tokens).last_hidden_state.mean(dim=1).detach().numpy() for tokens in tokenized_new_data]

    # float32로 형변환
    encoded_new_data = [embedding.astype('float32') for embedding in encoded_new_data]
    loaded_kmeans.cluster_centers_ = loaded_kmeans.cluster_centers_.astype('float32')

    # 각 새로운 문장에 대해 클러스터를 예측
    new_data_clusters = loaded_kmeans.predict(np.vstack(encoded_new_data))

    return new_data_clusters


def get_cluster_job_ratios(new_data_clusters):
    df = pd.read_csv('./data/final_sentence_df.csv')
    cluster_job_ratios = {}

    new_data_clusters = list(set(new_data_clusters))
    for cluster in new_data_clusters:
        cluster = int(cluster)  # 클러스터 레이블을 int로 변환
        #직무별 개수가 다르므로, 이를 고려하여 cluster내 직무별 size 조절
        weighted_df = df[df['kmeans_label'] == cluster].groupby('직무').size() / df.groupby('직무').size()
        weighted_cluster_sum = weighted_df.sum()
        #조절한 뒤, cluster개 직무 비율 계산(합이 1이 되도록)
        job_ratios = weighted_df / weighted_cluster_sum
        cluster_job_ratios[cluster] = job_ratios.to_dict()

    # 전체 평균 직무 비율 계산
    overall_ratios = pd.concat([pd.Series(v) for v in cluster_job_ratios.values()], axis=1).mean(axis=1).to_dict()
    #상위 1, 2위 계산
    sorted_ratios = sorted(overall_ratios.items(), key=lambda x: x[1], reverse=True)
    
    selected_job = []
    #1위의 직무명
    selected_job.append(sorted_ratios[0][0])
    #2위의 직무명
    selected_job.append(sorted_ratios[1][0])
    print(selected_job)

    return {'cluster_data': cluster_job_ratios, 'overall_ratios': overall_ratios}, selected_job

def get_noun_score(df, df_sentence):
    #불용어 처리
    #불용어 사전 불러오기
    with open('./data/korean_stopwords_응통.txt','r',encoding='utf-8-sig') as f:
        stopwords_list=[]
        example =f.readlines()
        for line in example:
            stopwords_list.append(line.strip())

    #추가하고 싶은 불용어가 있다면 다음과 같이 넣어서 사용
    add = ['근무조건','고용형태','근무장소','급여조건','합류','서류','전형','인터뷰','합격','유의사항','공고','모집','채용','정규직','면접','전형','근무시','출근']

    #불용어 사전 정의
    stop = add+stopwords_list

    ##### wordextractor
    # 형태소에 해당하는 단어를 분리하는 학습 수행
    from soynlp.word import WordExtractor

    # 다른 파라미터는 그냥 두고, min_frequency만 1로 설정.
    word_extractor = WordExtractor(min_frequency=1)

    # 단어 토큰화를 위해 df['combined'] 학습
    word_extractor.train(df['sentence'].astype(str))

    # cohesion, branching entropy, accessor variety score 계산
    # cohension : 조건부확률을 통해 한글자씩 예측. cohension 값이 높은 위치가 하나의 단어를 이루고 있을 가능성이 큼.
    # branching entropy : 확률분포의 엔트로피를 통  해서 계산
    # accessor variety : 확률분포 없이, 다음 글자로 등장할 수 있는 경우의 수 계산
    word_score = word_extractor.extract()

    # soynlp의 토큰화 방식.

    # L-토큰화
    # 한국어의 경우 공백(띄어쓰기)으로 분리된 하나의 문자열은 ‘L 토큰 + R 토큰; 구조인 경우가 많음
    # 왼쪽에 오는 L 토큰은 체언(명사, 대명사)이나 동사, 형용사 등이고 오른쪽에 오는 R 토큰은 조사, 동사, 형용사 등이다.
    # 여러가지 길이의 L 토큰의 점수를 비교하여 가장 점수가 높 L단어를 찾는 방법
    from soynlp.tokenizer import LTokenizer

    # 최대점수토큰화
    # 띄어쓰기가 되어 있지 않는 긴 문자열에서 가능한 모든 종류의 부분문자열을 만들어서 가장 점수가 높은 것을 하나의 토큰으로
    # 우리는 띄어쓰기가 되어있는 텍스트니까 사용하지 않을 예정
    from soynlp.tokenizer import MaxScoreTokenizer

    # 규칙기반토큰화
    # 우리가 사용하기에 적절치않음.
    from soynlp.tokenizer import RegexTokenizer
    from soynlp.noun import LRNounExtractor

    # 명사 추출을 위해 df_sentence2['sentence'] 학습
    noun_extractor = LRNounExtractor()
    nouns = noun_extractor.train_extract(df['sentence'].astype(str))

    # 명사추출기의 score와 cohesion score를 함께 이용해서 토큰화 하기

    cohesion_score = {word:score.cohesion_forward for word, score in word_score.items()}
    noun_scores = {noun:score.score for noun, score in nouns.items()}

    combined_scores = {noun:score + cohesion_score.get(noun, 0)
        for noun, score in noun_scores.items()}

    combined_scores.update(
        {subword:cohesion for subword, cohesion in cohesion_score.items()
        if not (subword in combined_scores)}
    )
    LTokenizer = LTokenizer(scores=combined_scores)

    # 토큰화 한뒤, 토큰마다 명사 판별 진행하여 명사 추출하기. nouns 컬럼에 결과 저장.

    train_list=list(df_sentence.sentence)
    num_result = []
    for i in range(len(train_list)):
        num = []
        tok = LTokenizer.tokenize(str(train_list[i]))
        for j in tok:
            if noun_extractor.is_noun(j) and j not in stop:
                num.append(j)
        num_result.append(num)

    return num_result


def get_cluster_keywords(input_list, selected_job, new_data_clusters):

    #input_list 데이터프레임 변경
    # 분리했던 데이터 로드
    df1 = pd.read_csv('./data/final_sentence_df_1.csv')
    df2 = pd.read_csv('./data/final_sentence_df_2.csv')
    df3 = pd.read_csv('./data/final_sentence_df_3.csv')

    # 데이터 합치기
    df = pd.concat([df1, df2, df3])

    df_sentence = pd.DataFrame({"sentence":input_list})
    df['sentence'] = df['sentence'].str.lower()
    # df_sentence['cluster'] = new_data_clusters

    num_result = get_noun_score(df, df_sentence)
    df_sentence['nouns'] = num_result

    keyword = pd.read_csv("./data/keyword_df.csv", header=0)

    gpt_input = []
    for nouns, cluster in zip(df_sentence['nouns'], new_data_clusters):
        #군집의 키워드를 사용자 키워드와 비교하여 추출해옴
        keys_list = [str(item) for item in keyword.iloc[:, cluster].tolist() if item not in nouns]
        gpt_input.extend(keys_list)

    #최종 추출된 키워드를 사용자 키워드와 한번 더 비교함(다른 군집에서도 사용자 키워드가 중복된 경우가 있을 수 있음)
    user_keywords = list(set([word for sublist in df_sentence['nouns'].tolist() for word in sublist]))
    gpt_input_list = [item for item in gpt_input if item not in user_keywords and item != 'nan']
    print(gpt_input_list)
    return generate_text(gpt_input_list)

def my_gonggo(new_data_clusters):
    # 각 군집 count 초기화
    user_clusters_count = [0,0,0,0]

    # 각 군집의 등장 횟수 계산
    for cluster in new_data_clusters:
        cluster = int(cluster)
        user_clusters_count[cluster] += 1

    #등장 횟수를 비율로 변환
    total_count = np.sum(user_clusters_count)
    user_clusters_ratio = user_clusters_count / total_count

    print(user_clusters_count)
    print(user_clusters_ratio)

    gonggo = pd.read_csv('./recruit/data/Gonggo.csv')
    gonggo['label_ratios'] = gonggo['label_ratios'].apply(ast.literal_eval)
    
    from sklearn.metrics.pairwise import cosine_similarity

    # 사용자 문장과 가장 유사도가 높은 채용공고를 탐색
    # 코사인 유사도 계산
    #gonggo['cosine_similarity'] = gonggo['label_ratios'].apply(lambda x: cosine_similarity([user_clusters_ratio], [x])[0][0])
    gonggo['cosine_similarity'] = gonggo['label_ratios'].apply(lambda x: cosine_similarity([[0.416, 0.166, 0.083,0.333]], [x])[0][0])

    m = max(gonggo['cosine_similarity'])
    similar_index = [index for index, val in enumerate(gonggo['cosine_similarity']) if val == m]

    # 가장 유사한 벡터(들)의 인덱스
    return similar_index
