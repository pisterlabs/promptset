# '머신러닝 딥러닝'으로 검색해서 크롤링한 결과 확인하기
# If false, just print Nan. --> 입력하지 말 것 if조건 무시하고 just print에 집중하는 거 같다.
# you must make the job name simply / simply말고 simple이라고 하면 simple을 직무명으로 출력한다. simply도 해당됨

import pandas as pd
import os
import time, random
from dotenv import load_dotenv
import openai
import re
import json

path = 'C:/Users/hoeeun/Desktop/Crawling_master/normalization'

# 크롤링 병합 결과 출력하기
new = pd.read_csv(path + '/data/0918/0918_master.csv')
# print(new.head())

total = len(new)

## chat-GPT활용
load_dotenv()
openai.api_key = os.getenv("openai.api_key")

##프롬프트 입력해서 GPT돌리기
def infer_ai(Text):
    true = ['computer vision', 'ml모델', '딥러닝 프레임워크', '딥러닝', '머신러닝/딥러닝', 'openai', 'object detection', '머신러닝/딥러닝 기반 알고리즘']
    false = ['설계', 'DSP/NPU/AP/VHDL/Verilog', '강사', 'sw개발', 'soc', '사업', '관리', '데이터분석', 'SQL', '통계분석', '전기/전자', '머신비전', '반도체']

    delimiter = "####"

    Prompt = f"""
    #Order
    You will be provided with the text.
    The text will be delimited with {delimiter} characters.
    Classify each text into a primary category and a secondary category.
    Provide your output in json format with the keys: primary and secondary.
    
    #Queries
    Primary categories: AI_True, AI_False
    AI_True categories: [{', '.join(true)}]
    AI_False categories: [{', '.join(false)}]

    Make sure to choose the category accurately.
    Simply show me the results, no further explanations or questions.
    """

    messages = [{'role': 'system', 'content': Prompt},
                {'role': 'user', 'content': f'{delimiter}{Text}.'}]

    print(f"messages here:{messages}")

    chat = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=messages,
        temperature=0
    )

    reply = chat.choices[0].message.content
    reply = json.loads(reply)
    print(f'ChatGPT: {reply}')

    try:
        reason = reply['secondary']
        ai = reply['primary']
    except Exception as e:
        reason = "Error"
        ai = "Error"

    print('------Here------' + '\n', f'{i}/{total}\n AI:{ai} Reason:{reason}')
    return (ai, reason)


# 직무 선택 반복문
for i in range(total):
    try:
        Text = new.loc[i, 'main'] + new.loc[i, 'require']
        Text = re.sub(r'\d+\.', "", Text)
        Text = re.sub(r"[^\w\s~]", "", Text).replace('ㆍ', '')
        Text = re.sub(r'\t', '', Text)

        ai, reason = infer_ai(Text)
        time.sleep(random.uniform(3, 5))

        if ai:
            new.loc[i, 'AI'] = ai
            new.loc[i, 'Reason'] = reason
        else:
            new.loc[i, 'AI'] = "Error"
            new.loc[i, 'Reason'] = "Error"
        print("=" * 100)
    except:
        pass

new.to_csv(path + "/data/0918/0918_master_v2.csv", index=False)
