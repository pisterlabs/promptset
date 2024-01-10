import os
import elasticsearch
from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
from elasticsearch import helpers

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


def elasticsearch_finder(search_query):
    # 특정 쿼리로 검색하기
    # 여기 입력
    search_body = {
        'query': {
            'match': {
                'text': search_query
            }
        },
        'size': 5 # 최대 5개의 문서만 반환(토큰 4000제한 때문에)
    }

    # 검색 실행
    results = es.search(index=index_name, body=search_body)

    # 결과 출력

    for hit in results['hits']['hits']:
        # print(hit['_source']['text'])

        top_hit = results['hits']['hits'][0]['_source']['text']


    import os
    import base64  
    from openai import OpenAI
    file_path = "gpt_key_base64.txt"
    with open(file_path, 'r') as file:
        # 파일 내용 읽기
        encoded_content = file.read()
        # base64 디코딩
        decoded_content = base64.b64decode(encoded_content)
        decoded_string = decoded_content.decode('utf-8')

    os.environ["OPENAI_API_KEY"] = decoded_string
    client = OpenAI(

        api_key=decoded_string,
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system","content": f"공지사항을 알려주는 쳇봇이다."
        "마크다운언어를 이용해서 깔끔하게 보여준다."},
        {"role": "user", "content": f"{top_hit}을보고 {search_query}에 대해 알려줘"},
    ],
    model="gpt-3.5-turbo-1106",
    temperature=0,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )

    # content 추출
    content = chat_completion.choices[0].message.content
    return content
