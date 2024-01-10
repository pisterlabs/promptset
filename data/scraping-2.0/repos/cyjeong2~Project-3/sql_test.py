import pymysql

from hanspell import spell_checker  #git clone으로 로컬설치
from googletrans import Translator
from konlpy.tag import Okt
# from PyKomoran import *
# import pyokt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import numpy as np
import itertools

import time
import random

import openai
import os



conn = pymysql.connect(host='localhost', user='admin', password='admin', db='project_third')

try:
    with conn.cursor() as curs:
        sql = "SELECT * FROM diary_memory ORDER BY created_at DESC"
        curs.execute(sql) # 실행할 쿼리분 넣기
        rs = curs.fetchall() #sql문 실행해서 데이터 가져오기
        # print(rs)  # 쿼리문 출력해보기
        main_cont = rs[0][1]
        emotion = rs[0][6]
        drawing = rs[0][5]
        print('='*10)
        print(main_cont)
        print('emotion : ', emotion) # emotion 출력
        print('drawing : ', drawing) # drawing 출력
        print('='*10)
        print()

        # for row in rs:
        #     print(row[0])
        #     # for data in row:
        #     #     print(data)
        #     #     print('-'*10)
        #     print()
finally:
    conn.close()

# 이미지 출력하기
start = time.time()

# 1) 맞춤법 검사
main_cont_300 = main_cont[:301]
spelled_cont = spell_checker.check(u'{}'.format(main_cont_300)).as_dict()['checked']
print(spelled_cont)
print('='*10)

# 2) 키워드 추출
okt = Okt()

tokenized_doc = okt.pos(spelled_cont)
# print(tokenized_doc)

# print('m')
# print([word for word in tokenized_doc])

# print([word[0] for word in tokenized_doc if word[1] == 'Noun'])
# print('m')
tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

# print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
# print('명사 추출 :',tokenized_nouns)


# 단어 합치기
n_gram_range = (2, 3)

count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
candidates = count.get_feature_names_out()

# print('trigram 개수 :',len(candidates))
# print('trigram 다섯개만 출력 :',candidates[:4])


# 유사 키워드 추출
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding = model.encode([main_cont])
candidate_embeddings = model.encode(candidates)

top_n = 4
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)


# 후보 간 유사성 최소화
def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings,
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


fin_keyword = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=top_n, nr_candidates=30)
end = (time.time() - start)
print(fin_keyword)
print('키워드 추출 : ', end)

#번역 및 입력
translator = Translator()
eng_keywords = ', '.join([translator.translate(word, dest='en').text for word in fin_keyword])
print(eng_keywords)
print('번역 : ', start-time.time())

search_keyword = eng_keywords + ', ' + emotion + ', ' + drawing


# 3) 이미지 전환 및 추출
# token key
openai.api_key = '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'

#함수
response = openai.Image.create(
    # 입력받은 키워드 입력
    prompt = search_keyword,
    
    # 출력할 그림 개수
    n = 4,

    # 출력할 그림 사이즈
    size = '256x256'
)

image_url_1 = response['data'][0]['url']
image_url_2 = response['data'][1]['url']
image_url_3 = response['data'][2]['url']
image_url_4 = response['data'][3]['url']

#접속 url (1시간 유지)
print(image_url)
print('이미지 : ', start-time.time())


# 해당 url 로 접속해서 이미지 저장해도 될 것 같습니다. 



