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
# print(new.head())

total = len(new)

## chat-GPT활용
load_dotenv()
client = OpenAI(
    api_key=os.getenv("openai.api_key")
)


##프롬프트 입력해서 GPT돌리기
def infer_ai(Text):
    true = ['computer vision', 'ML model', 'deep learning framework', 'deep learning', 'machine learning', 'OpenAI','object detection', 'machine learning/deep learning-based algorithms']
    false = ['design', 'DSP/NPU/AP/VHDL/Verilog', 'instructor', 'software development', 'SoC','business', 'management', 'data visualization', 'SQL', 'statistical analysis', 'electrical/electronic','machine vision', 'semiconductor']
    jobname = ['Data scientist', 'Data enginner', 'Machinelearning/Deeplearning researcher', 'Machinelearning/Deeplearning enginner', 'AI developer', 'AI service developer', 'AI serive planner','Prompt enginner', 'AI artist']

    delimiter = "####"

    Prompt = f"""
    #Order
    You are an AI job master. You will be provided with the text.
    The text will be delimited with {delimiter} characters.
    Classify each text into a primary category, a secondary category and a tertiary category.

    #Queries
    Primary categories: AI_True, AI_False
    
    Secondary Categories: AI_True: [{', '.join(true)}], AI_False: [{', '.join(false)}]
    
    Tertiary Categories: AI_Ture: [{', '.join(jobname)}], AI_False: 'Nan'

    Provide your output in json format with the keys: primary, secondary, tertiary.
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
        jobname = reply['tertiary']
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

new.to_csv(path + "for_check_v9.csv", index=False)
