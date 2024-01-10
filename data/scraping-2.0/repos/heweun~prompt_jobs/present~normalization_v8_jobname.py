# '머신러닝 딥러닝'으로 검색해서 크롤링한 결과 확인하기
# If false, just print Nan. --> 입력하지 말 것 if조건 무시하고 just print에 집중하는 거 같다.
# you must make the job name simply / simply말고 simple이라고 하면 simple을 직무명으로 출력한다. simply도 해당됨

import pandas as pd
import os
import time, random
from dotenv import load_dotenv
# import openai
import re
import json
from openai import OpenAI

path = 'C:/Users/hoeeun/Desktop/Crawling_master/normalization/data/'

# 크롤링 병합 결과 출력하기
new = pd.read_csv(path + 'for_check.csv')
print(new.head())

total = len(new)

## chat-GPT활용
load_dotenv()
client = OpenAI(
    api_key=os.getenv("openai.api_key")
)

##프롬프트 입력해서 GPT돌리기
def infer_ai(Text):
    true = ['computer vision', 'ml모델', '딥러닝 프레임워크', '딥러닝', '머신러닝/딥러닝', 'openai', 'object detection', '머신러닝/딥러닝 기반 알고리즘']
    false = ['설계', 'DSP/NPU/AP/VHDL/Verilog', '강사', 'sw개발', 'soc', '사업', '관리', '데이터분석', 'SQL', '통계분석', '전기/전자', '머신비전',
             '반도체']
    jobname = ['데이터 사이언티스트', '데이터 엔지니어', '머신러닝/딥러닝 리서처', '머신러닝/딥러닝 엔지니어', 'AI개발자', 'AI서비스 개발자', 'AI서비스 기획자', '프롬프트 엔지니어', 'AI아티스트']

    delimiter = "####"

    Prompt = f"""
    #Order1
    You are an AI job master. You will be provided with the text.
    The text will be delimited with {delimiter} characters.
    Classify each text into a primary category and a secondary category.
    
    #Queries1
    Primary categories: AI_True, AI_False
    AI_True categories: [{', '.join(true)}]
    AI_False categories: [{', '.join(false)}]
    
    #Order2
    If it's primary category is 'AI_True', make sure to choose the jobname category accurately. 
    Only in that Jobname categories. Nothing else.
    
    #Queries2
    Jobname categories: [{', '.join(jobname)}]

    If it's primary category is 'AI_False',
    then simply write jobname as nan. Nothing else.
    
    Provide your output in json format with the keys: primary, secondary, jobname.
    Make sure to choose the category accurately.
    """

    messages = [{'role': 'system', 'content': Prompt},
                {'role': 'user', 'content': f'{delimiter}{Text}.'}]

    print(f"messages here:{messages}")

    chat = client.chat.completions.create(
        model='gpt-4',
        messages=messages,
        temperature=0
    )

    reply = chat.choices[0].message.content
    reply = json.loads(reply)
    print(f'ChatGPT: {reply}')

    try:
        jobname = reply['jobname']
        ai = reply['primary']
        reason = reply['secondary']
    except Exception as e:
        jobname = "Error"
        ai = "Error"
        reason = "Error"

    print('------Here------' + '\n', f'{i}/{total}\n AI:{ai} reason:{reason} jobname:{jobname}')
    return (ai, reason, jobname)


# 직무 선택 반복문
for i in range(total):
    try:
        Text = new.loc[i, 'main'] + new.loc[i, 'require']
        Text = re.sub(r'\d+\.', "", Text)
        Text = re.sub(r"[^\w\s~]", "", Text).replace('ㆍ', '')
        Text = re.sub(r'\t', '', Text)

        ai, reason, jobname = infer_ai(Text)
        time.sleep(random.uniform(3, 5))

        if ai:
            new.loc[i, 'AI'] = ai
            new.loc[i, 'reason'] = reason
            new.loc[i, 'jobname'] = jobname
        else:
            new.loc[i, 'AI'] = "Error"
            new.loc[i, 'reason'] = "Error"
            new.loc[i, 'jobname'] = "Error"
        print("=" * 100)
    except:
        pass

new.to_csv(path + "for_check_v8.csv", index=False)
