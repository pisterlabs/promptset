import pandas as pd
import os
from datetime import datetime
from collections import Counter
import time, random
from dotenv import load_dotenv
import openai
import re

path = 'C:/Users/hoeeun/Desktop/Crawling_master/normalization'
## 크롤링 병합 결과 출력하기
new = pd.read_csv(path+'/data/0725/0725master.csv')
new = new[new['크롤링날짜'] == '2023-07-24']
new = new[new['크롤링사이트'].isin(['programmers', 'jumpfit', 'wanted'])]

# 인덱스를 없애고 새로운 데이터프레임을 생성
new = new.reset_index(drop=True)
print(Counter(new['크롤링날짜']))

total = len(new)

## chat-GPT활용
load_dotenv()
openai.api_key = os.getenv("openai.api_key")


#assistant있는게 더 나은거 같다.
##프롬프트 입력해서 GPT돌리기
def generate_job_name(Text):
    Job = ['Data Scientist', 'Data Engineer', 'Machine Learning/Deep Learning Researcher',
           'Machine Learning/Deep Learning Engineer', 'AI Developer', 'AI Product Service Developer',
           'AI Service Planner', 'Prompt Engineer', 'AI Artist'] #리서처, 엔지니어 -->developer에 추가

    delimiter = "####"

    Prompt = f"""
    #Order
    You are a job name classifier, and you excel at precisely adhering to the format.
    Your task is to perform the following actions for giving a job name to job hunters who are interested in AI job role.
    The text will be delimited with four hashtags, i.e. {delimiter}.
    
    Step 1: Summarize the following text to provide information about main duties and job qualifications in one sentence.
    Step 2: Dose the summarized sentence is related to an AI job role? (True or False)
    Step 3: If the answer is true, select only one job name from the following list '{Job}' that describes the job.
    You must exclusively select from the following list, absolutely nothing else.
    If there is no suitable job name, then simply write 'Nan'
    In case you are unsure of the answer, simply respond with 'Nan'. 
    Avoid providing fabricated answers.
    
    Absolutely do not finish reasoning until you have completed all the steps.
    If you can't reasoning, then simply write 'Nan' and proceed to the next.
    You must complete this format:

    #Format
    {delimiter} <step 1 reasoning>\n
    {delimiter} <step 2 reasoning> as a boolean only\n
    {delimiter} <step 3 reasoning> as a single word only\n

    Make sure to separate every step.
    And fill in the entire format.
    Simply show me the results, no further explanations or questions.
    """

    Assistant = f"""I must follow the format. 
    And I assure you that I will complete all the steps.
    I'll respond without reiterating the explanations or questions."""

    messages = [{'role':'system','content':Prompt},
                {'role': 'user', 'content': f'{delimiter}{Text}.'},
                {'role': 'assistant', 'content': Assistant}]

    print(f"messages here:{messages}")

    chat = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=messages,
        temperature=0
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

    print('------Here------' + '\n',f'{i}/{total}\n Summary:{summary}\n AI:{ai_response} Jobname:{final_response}')
    return (summary, ai_response, final_response)


# 직무 선택 반복문
for i in range(total):
    try:
        Text = new.loc[i, '주요업무'] + new.loc[i, '자격요건']
        if len(Text) >= 500:
            Text = new.loc[i, '자격요건']
        else:
            Text = Text

        Text = re.sub(r'\d+\.', "", Text)
        Text = re.sub(r"[^\w\s~]", "", Text).replace('ㆍ', '')
        Text = re.sub(r'\t', '', Text)

        summary, ai, selected_title = generate_job_name(Text)
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

new.to_csv(path+"/data/0725/0725master_v4.csv", index=False)