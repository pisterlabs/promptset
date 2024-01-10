# -*- coding:utf-8 -*-
import openai
import sys
import json
import time


def get_key():
    return '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'


def prompt_pretreatment(query: str, candidates: list):

    similarity_prompt = "我将向你提供一个查询语句和一个候选语句列表，请你从语义相似和内容相似两个方面综合考虑选出候选语句列表中与查询语句匹配程度最高的候选项。\n" \
                        "请将返回结果格式调整为json格式。\n" \
                        "例如：匹配成功，返回结果{\"code\": \"succ\",\"sentence\": \"匹配程度最高的候选语句\"}。匹配失败，返回结果{\"code\": \"fail\",\"sentence\": \"\"}。\n" \
                        "注意: 请不要输出除我要求格式以外的任何其它内容。请你输出候选语句中成功匹配的候选项时，不要对候选项本身的内容做任何改动。\n" \
                        "查询语句: {q}。\n" \
                        "候选语句列表: {c}。"

    similarity_prompt = similarity_prompt.replace("{q}", f"\"{query}\"")
    # candidates_list = str(candidates).replace('[', '')
    # candidates_list = candidates_list.replace(']', '')
    similarity_prompt = similarity_prompt.replace("{c}", str(candidates))
    
    if len(similarity_prompt) > 3200:
        similarity_prompt = similarity_prompt[:3200]
    
    return similarity_prompt


def openai_query(content, apikey):
    # 防止返回速率超过openai限制
    time.sleep(2)
    openai.api_key = apikey
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # gpt-3.5-turbo-0301
        messages=[
            {"role": "user", "content": content}
        ],
        temperature=0.1,  # 控制生成的随机性 0-1越高随机性越强
        max_tokens=128,  # 生成内容的最大token限制
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(response)
    return response.choices[0].message.content


if __name__ == '__main__':

    # temp_list = ['中国科学技术大学教授','江西赣锋锂业股份有限公司董事长，民建江西省委副主委']
    # #
    # prompt = prompt_pretreatment("江西赣锋锂业股份有限公司", temp_list)
    # print(prompt)
    # ans = openai_query(prompt, get_key())
    # print(eval(ans))
    pass
