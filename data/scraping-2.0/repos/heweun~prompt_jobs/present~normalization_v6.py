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
print(Counter(new['crawlingDate']))
print(Counter(new['crawlingSite']))
print(new.head())

# total = len(new)
#
# ## chat-GPT 활용
# load_dotenv()
# api_key = [os.getenv("openai.api_key"), os.getenv("openai.api_key2")]  # 두 개의 API 키를 리스트로 저장
#
#
# # 프롬프트 입력해서 GPT 돌리기
# def generate_job_name(Text, api_key_index):
#
#     delimiter = "####"
#
#     Prompt = f"""
#     #Order
#     You are a job name maker, excelling at precise format adherence.
#     Your duty is to follow the steps below to provide job titles for categorization.
#     The text will be delimited with four hashtags, i.e. {delimiter}.
#
#     Step 1: Extract relevant information from the text to give job info. Limit to 30 words.
#     Step 2: Does the summary have a direct connection to the development, application, and management of machine learning and deep learning tech? (True or False)
#     Step 3: If True, make a proper job name in English. If False, do not make a job name.
#
#     Absolutely do not finish reasoning until you have completed all the steps.
#     You must follow and complete this format:
#
#     #Format
#     {delimiter} <step 1 result>\n
#     {delimiter} <step 2 result> as a boolean only\n
#     {delimiter} <step 3 result> as a single word only\n
#
#     Make sure to include {delimiter} to separate every step.
#     And fill in the entire format.
#     Simply show me the results only, no commands and explanations.
#     """
#
#     Assistant = f"""I'll respond without reiterating the explanations or commands."""
#
#     messages = [{'role': 'system', 'content': Prompt},
#                 {'role': 'user', 'content': f'{delimiter}{Text}.'},
#                 {'role': 'assistant', 'content': Assistant}]
#
#     print(f"messages here:{messages}")
#
#     chat = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo-0613',
#         messages=messages,
#         temperature=0,
#         api_key=api_key[api_key_index]  # 해당 인덱스의 API 키 사용
#     )
#
#     reply = chat.choices[0].message.content
#     print(f'ChatGPT: {reply}', '\n')
#
#     try:
#         final_response = reply.split(delimiter)[-1].strip()
#         ai_response = reply.split(delimiter)[-2].strip()
#         summary = reply.split(delimiter)[-3].strip()
#     except Exception as e:
#         final_response = "Error"
#         ai_response = "Error"
#         summary = 'Error'
#
#     print('------Here------' + '\n', f'{i}/{total}\n Summary:{summary}\n AI:{ai_response} Jobname:{final_response}')
#     return (summary, ai_response, final_response)
#
#
# # 직무 선택 반복문
# for i in range(total):
#     try:
#         Text = new.loc[i, 'main'] + new.loc[i, 'require']
#         if len(Text) >= 1000:
#             Text = new.loc[i, 'require']
#         else:
#             Text = Text
#
#         Text = re.sub(r'\d+\.', "", Text)
#         Text = re.sub(r"[^\w\s~]", "", Text).replace('ㆍ', '')
#         Text = re.sub(r'\t', '', Text)
#
#         api_key_index = i % len(api_key)  # 짝수/홀수에 따라 API 키 인덱스 결정
#         print(f'api_key:{api_key_index}')
#         summary, ai, selected_title = generate_job_name(Text, api_key_index)
#         time.sleep(random.uniform(3, 5))
#
#         if selected_title:
#             new.loc[i, 'summary'] = summary
#             new.loc[i, 'AI'] = ai
#             new.loc[i, 'category'] = selected_title
#         else:
#             new.loc[i, 'summary'] = "Error"
#             new.loc[i, 'AI'] = "Error"
#             new.loc[i, 'category'] = "Error"
#         print("=" * 100)
#     except:
#         pass
#
# new.to_csv(path + "/data/md_master_v2.csv", index=False)
