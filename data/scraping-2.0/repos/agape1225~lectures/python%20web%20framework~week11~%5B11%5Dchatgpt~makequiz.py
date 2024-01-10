import os
import openai
api_key = os.environ["api_key"]
openai.api_key = api_key 

filename = "대한민국상식"

prompt  = f"""
당신은 블러그의 운영자 입니다. {filename} 퀴즈 5개를 4개의 보기로 작성하려고 합니다. 
please generate  python code to write excel file using dataframe in table format with 7 columns.
please save the excel file as {filename}.xlsx.
python 코드만 작성하고 설명이나 주석없이 코드만 알려주세요

column 1 :  문제, column 2 : 보기1, column 3 :  문제2, column 4 : 보기3,   
column 5: 보기4, column 6 : 정답, column 7 : 정답설명

"""

print(prompt)
messages = []
messages.append({"role": "user", "content": prompt})
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
res = completion.choices[0].message['content']
print(res)
exec(res)