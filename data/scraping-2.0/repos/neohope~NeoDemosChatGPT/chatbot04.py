#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import tiktoken
import yaml

'''
通过ChatCompletion接口，实现聊天机器人功能
试用tiktoken进行token长度计算
'''

encoding = tiktoken.get_encoding("cl100k_base")

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


class Conversation2:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append( {"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]
        num_of_tokens = response['usage']['total_tokens']
        self.messages.append({"role": "assistant", "content": message})

        prompt_count = len(encoding.encode(self.prompt))
        question_count = len(encoding.encode(question1))
        answer_count = len(encoding.encode(answer))
        total_count = prompt_count + question_count + answer_count
        print("Prompt消耗 %d Token, 问题消耗 %d Token，回答消耗 %d Token，总共消耗 %d Token" % (prompt_count, question_count, answer_count, total_count))
                
        if len(self.messages) > self.num_of_round*2 + 1:
            del self.messages[1:3]
        return message, num_of_tokens


if __name__ == '__main__':
    get_api_key()

    prompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内"""

    question1 = "你是谁？"
    question2 = "请问鱼香肉丝怎么做？"
    question3 = "那蚝油牛肉呢？"
    question4 = "我问你的第一个问题是什么？"
    question5 = "你计算的时候上下文会追溯多久？"
    
    conv2 = Conversation2(prompt, 3)
    questions = [question1, question2, question3, question4, question5]
    for question in questions:
        answer, num_of_tokens = conv2.ask(question)
        print("询问 {%s} 消耗的token数量是 : %d" % (question, num_of_tokens))
