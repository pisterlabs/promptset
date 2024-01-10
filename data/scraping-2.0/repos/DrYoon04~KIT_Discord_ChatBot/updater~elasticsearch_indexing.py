import os
import elasticsearch
from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
import sys
import json
import datetime
from elasticsearch import helpers
from langchain.text_splitter import CharacterTextSplitter
from elasticsearch import Elasticsearch

#ea server connect

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
if es.ping():
    print('Yay Connect')
else:
    print('Awww it could not connect!')

# Elasticsearch 인덱스 생성
index_name = 'txt_data_index'
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

    import os
# 폴더에서 데이터 읽어오기
folder_path = 'module/data/notice_txt'
# 폴더 내의 모든 파일 목록 가져오기
file_list = os.listdir(folder_path)

# text_splitter 초기화
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# 각 텍스트 파일을 읽어와서 Elasticsearch에 색인화
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # text_splitter를 사용하여 텍스트 분리
    chunks = text_splitter.split_text(data)

    # 데이터를 Elasticsearch에 색인화
    for j, chunk in enumerate(chunks):
        document = {
            'text': chunk.strip(),  # 각 줄의 텍스트를 'text' 필드에 저장
            'file_name': file_name  # 파일 이름도 저장 (옵션)
        }
        es.index(index=index_name, id=i * len(file_list) + j + 1, body=document)
