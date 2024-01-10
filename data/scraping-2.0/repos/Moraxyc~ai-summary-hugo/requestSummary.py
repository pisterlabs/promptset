'''
Author: Moraxyc me@morax.icu
Date: 2023-08-16 02:32:59
LastEditors: Moraxyc me@morax.icu
LastEditTime: 2023-08-16 11:50:06
FilePath: /ai-summary-hugo/requestSummary.py
Description: 通过API向chatgpt请求文章总结

Copyright (c) 2023 by Moraxyc, All Rights Reserved. 
'''
import openai
import os

def generate_summary(prompt):
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "请在100字内用中文总结以下文章的核心内容: "},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except openai.error.OpenAIError as e:
        print("OpenAI Error:", e)
        return None