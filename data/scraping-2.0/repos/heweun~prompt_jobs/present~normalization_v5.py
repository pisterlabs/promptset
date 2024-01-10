#홀,짝으로 API분리해서 결과 출력하기

import pandas as pd
import os
from datetime import datetime
from collections import Counter
import time, random
from dotenv import load_dotenv
import openai
import re

path = 'C:/Users/hoeeun/Desktop/Crawling_master/normalization'

# 크롤링 병합 결과 출력하기
new = pd.read_csv(path + '/data/0731/0731master.csv')
new = new[new['crawlingSite'].isin(['점핏', '원티드'])]
new = new.reset_index(drop=True)
print(Counter(new['crawlingDate']))
print(Counter(new['crawlingSite']))

total = len(new)

## chat-GPT 활용
load_dotenv()
api_key = [os.getenv("openai.api_key"), os.getenv("openai.api_key2")]  # 두 개의 API 키를 리스트로 저장
print(api_key)

# 프롬프트 입력해서 GPT 돌리기
def generate_job_name(Text, api_key_index):
    Job = ['Data Scientist', 'Data Engineer', 'Machine Learning/Deep Learning Researcher',
           'Machine Learning/Deep Learning Engineer', 'AI Developer', 'AI Product Service Developer',
           'AI Service Planner', 'AI prompt Engineer', 'AI Artist']  # 리서처, 엔지니어 --> developer에 추가

    delimiter = "####"

    Prompt = f"""
    #Order
    You are a job name classifier, excelling at precise format adherence.
    Your duty is to follow the steps below to provide job title for AI job seekers.
    The text will be delimited with four hashtags, i.e. {delimiter}.
    
    Step 1: Extract relevant information from the text to give job info. Limit to 30 words.
    Step 2: Does the summary have a direct connection to the development, application, and management of machine learning and deep learning tech? (True or False)
    Step 3: If True, then you must choose one job title from the given list '{Job}'- nothing else. If no exact match, select the most similar, ensuring only one choice. If False, just print 'Nan'. 
    Avoid providing fabricated responses.

    Absolutely do not finish reasoning until you have completed all the steps.
    You must follow and complete this format:

    #Format
    {delimiter} <step 1 result>\n
    {delimiter} <step 2 result> as a boolean only\n
    {delimiter} <step 3 result> as a single word only\n
    
    Make sure to separate every step.
    And fill in the entire format.
    Simply show me the results only, no commands and explanations
    """

    Assistant = f"""I'll respond without reiterating the explanations or commands."""

    messages = [{'role': 'system', 'content': Prompt},
                {'role': 'user', 'content': f'{delimiter}{Text}.'},
                {'role': 'assistant', 'content': Assistant}]

    print(f"messages here:{messages}")

    chat = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=messages,
        temperature=0,
        api_key=api_key[api_key_index]  # 해당 인덱스의 API 키 사용
    )

    reply = chat.choices[0].message.content
    print(f'ChatGPT: {reply}', '\n')

    try:
        final_response = reply.split(delimiter)[-1].strip()
        ai_response = reply.split(delimiter)[-2].strip()
        summary = reply.split(delimiter)[-3].strip()
    except Exception as e:
        final_response = "Error"
        ai_response = "Error"
        summary = 'Error'

    print('------Here------' + '\n', f'{i}/{total}\n Summary:{summary}\n AI:{ai_response} Jobname:{final_response}')
    return (summary, ai_response, final_response)

# 직무 선택 반복문
for i in range(total):
    try:
        Text = new.loc[i, 'main'] + new.loc[i, 'require']
        if len(Text) >= 1000:
            Text = new.loc[i, 'require']
        else:
            Text = Text

        Text = re.sub(r'\d+\.', "", Text)
        Text = re.sub(r"[^\w\s~]", "", Text).replace('ㆍ', '')
        Text = re.sub(r'\t', '', Text)

        api_key_index = i % len(api_key)  # 짝수/홀수에 따라 API 키 인덱스 결정
        print(f'api_key:{api_key_index}')
        summary, ai, selected_title = generate_job_name(Text, api_key_index)
        time.sleep(random.uniform(3, 5))

        if selected_title:
            new.loc[i, 'summary'] = summary
            new.loc[i, 'AI'] = ai
            new.loc[i, 'category'] = selected_title
        else:
            new.loc[i, 'summary'] = "Error"
            new.loc[i, 'AI'] = "Error"
            new.loc[i, 'category'] = "Error"
        print("=" * 100)
    except:
        pass

new.to_csv(path + "/data/0731/0731master_v2.csv", index=False)
