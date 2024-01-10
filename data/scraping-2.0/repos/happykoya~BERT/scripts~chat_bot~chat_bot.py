#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openai


openai.api_key = ""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Pythonプログラムだけを生成して"},
        {"role": "user", "content": "リストをバブルソートしてほしい"}
          #{"role": "assistant", "content": "Python"}
    ]   
)
#print(type(response))
#print(response)
program_list = []
result = response["choices"][0]["message"]["content"].encode("unicode-escape").decode("unicode-escape")
sentence = result.splitlines()
for i in range(len(sentence)):
    if sentence[i] == "```python":
        j = i
        while sentence[j] != "```":
            program_list.append(sentence[j+1])
            j+= 1

print(program_list)