# 방침에서 대제목을 반환하는 모듈
import openai
import ast

# Module
from .Match_Title_Prompt import match_create_prompt

import sys
import os

# import config
api_key = os.getenv("OPENAI_API_KEY")

# LLM 기반으로 추출한 대제목과 룰셋의 대제목 매칭해서 딕셔너리 형태로 출력
def Match_Title(title_list, rule):
    gpt_prompt = match_create_prompt(title_list, rule)
    message = [{"role": "user", "content": gpt_prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.1,
        max_tokens=4096,
        frequency_penalty=0.0
    )
    print((response['choices'][0]['message']['content']),"은 Match에서의 LLM의 응답입니다.")
    title_dict = ast.literal_eval((response['choices'][0]['message']['content']))
    return title_dict