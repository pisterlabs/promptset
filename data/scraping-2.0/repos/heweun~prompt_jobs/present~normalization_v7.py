# 홀,짝으로 API분리해서 결과 출력하기
# '머신러닝 딥러닝'으로 검색해서 크롤링한 결과 확인하기
# If false, just print Nan. --> 입력하지 말 것 if조건 무시하고 just print에 집중하는 거 같다.
# you must make the job name simply / simply말고 simple이라고 하면 simple을 직무명으로 출력한다. simply도 해당됨

import pandas as pd
import os
from collections import Counter
import time, random
from dotenv import load_dotenv
import openai
import re

path = 'C:/Users/hoeeun/Desktop/Crawling_master/normalization'

# 크롤링 병합 결과 출력하기
new = pd.read_csv(path + '/data/md_master.csv')
# print(Counter(new['crawlingDate']))
# print(Counter(new['crawlingSite']))
# print(new.head())

total = len(new)

## chat-GPT활용
load_dotenv()
openai.api_key = os.getenv("openai.api_key")

# assistant있는게 더 나은거 같다.
##프롬프트 입력해서 GPT돌리기
def generate_job_name(Text):
    true = ['computer vision','ml모델','딥러닝 프레임워크','딥러닝','머신러닝/딥러닝','openai','object detection','머신러닝/딥러닝 기반 알고리즘']
    false = ['설계','DSP','NPU','AP','VHDL','Verilog','강사','sw개발','soc','사업','관리','데이터분석','SQL','통계분석','전기/전자']

    delimiter = "####"

    Prompt = f"""
    #Order
    You are really clever at reading lists' contents and find those words from the text.
    The text will be delimited with four hashtags, i.e. {delimiter}.

    If the text contains words from this list True:{true}, print out 'True'.
    If there are any words in True list but the text contains words from this list False:{false}, print out 'False'. 
    You must choose only from the given lists - nothing else.
    
    You must complete this format but do not print out order:
    #Format
    {delimiter} <Answer (True or False)>
    {delimiter} <Tell me the reason why you choose that answer.>

    Make sure fill in the entire format.
    Simply show me the results, no further explanations or questions.
    """

    #Assistant = f"""I'll respond without reiterating the explanations or questions."""

    messages = [{'role': 'system', 'content': Prompt},
                {'role': 'user', 'content': f'{delimiter}{Text}.'}]
                #{'role': 'assistant', 'content': Assistant}]

    print(f"messages here:{messages}")

    chat = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=messages,
        temperature=0
    )

    reply = chat.choices[0].message.content
    print(f'ChatGPT: {reply}', '\n')

    try:
        reason = reply.split(delimiter)[-1].strip()
        ai = reply.split(delimiter)[-2].strip()
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

        ai, reason = generate_job_name(Text)
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

new.to_csv(path + "/data/md_master_v2.csv", index=False)
