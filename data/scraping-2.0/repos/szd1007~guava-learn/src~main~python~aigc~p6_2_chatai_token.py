import openai
import os
import tiktoken


openai.api_key = os.environ.get("OPENAI_API_KEY")

class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role":"system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                #model="gpt-4", 9月出账单就可以用最新模型了，搭个网站自己用 https://platform.openai.com/docs/models/gpt-4
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e
        message = response["choices"][0]["message"]["content"]
        num_of_tokens = response["usage"]["total_tokens"]
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            # remove the first round conversation
            del self.messages[1:3]
        # return message, num_of_tokens
        return message



# prompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答要满足以下要求
# 1. 你的回答必须是中文
# 2. 回答限制在100个字以内"""
#
# conv1 = Conversation(prompt, 3)
# question1 = "你是谁？"
# question2 = "请问鱼香肉丝怎么做？"
# question3 = "那蚝油牛肉呢？"
# question4 = "我问你的第一个问题是什么？"
# question5 = "我问你的第一个问题是什么？"
#
# questions = [question1, question2, question3, question4, question5]
#
# encoding = tiktoken.get_encoding("cl100k_base")
# for question in questions:
#     answer, num_of_tokens = conv1.ask(question)
#     print("询问{%s}消耗token的数量是 : %d" % (question, num_of_tokens))
#     prompt_count = len(encoding.encode(prompt))
#     question_count = len(encoding.encode(question))
#     answer_count = len(encoding.encode(answer))
#     total_count = prompt_count + question_count + answer_count
#     print("Prompt消耗 %d Token, 问题消耗 %d Token, 回答消耗 %d Token, 总共消耗 %d Token" %
#           (prompt_count, question_count, answer_count, total_count))



