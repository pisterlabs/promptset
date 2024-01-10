import pandas
import re
import openai
from openai import OpenAIError
import json
import warnings
from transformers import GPT2Tokenizer
import os
import tiktoken

warnings.simplefilter(action='ignore', category=FutureWarning)

openai.api_key = my-chat-gpt-key
tokenizer = tiktoken.get_encoding("cl100k_base")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
MAX_TOKEN_LENGTH = 3800

fileName = "blog_서울 스타벅스 장애인 화장실_cut.csv"

csv = pandas.read_csv(fileName, encoding='utf-8')
output_file = 'blog_서울 스타벅스 장애인 화장실.json'


question = '''
당신의 임무는 아래 블로그 글을 분석하여 정해진 형식으로 출력하는 것입니다.  
입력된 블로그 글을 분석하여 반드시 아래의 정해진 형식으로 출력하세요.
분석 해야할 것은 다음과 같습니다:
1. 블로그 글이 스타벅스에 관한 내용인지 분석합니다.
2. 스타벅스와 관련된 내용이라면 스타벅스의 지점명을 알아냅니다,
3. 스타벅스 지점 내 장애인 화장실 유무를 확인합니다.

정해진 형식은 다음과 같습니다:
스타벅스에 관한 내용 : (y/n)
스타벅스의 지점명 : (지점명)
장애인 화장실 유무 : (y/n)

블로그 글은 다음과 같습니다 :

'''

for index, row in csv.iterrows():
    results = []
    article_num = int(row[0])
    text_data = question + str(row['article'])
    location_data = str(row['location'])
    tokens = tokenizer.encode(text_data)

    if len(tokens) > MAX_TOKEN_LENGTH:
        tokens = tokens[:MAX_TOKEN_LENGTH - 1]  # 마지막 토큰을 위한 자리를 남김
        text_data = tokenizer.decode(tokens)

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=100,
        messages=[{"role": "user", "content": text_data}]
    )

    chat_response = response.choices[0].message.content

    results.append({
        'index': article_num,
        'loc': location_data,
        'output': chat_response
    })

    print(chat_response, index)

    try:
        with open(output_file, 'r', encoding='utf-8-sig') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []

    data.append(results)

    with open(output_file, 'w', encoding='utf-8-sig') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent='\t')

    csv = csv.drop(0)
    csv.to_csv(fileName, index=False, encoding='utf-8-sig')
    csv = pandas.read_csv(fileName, encoding='utf-8')

if csv.empty:
    os.remove(fileName)
