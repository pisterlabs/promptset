import openai
import os
from config import API_KEY

openai.api_key = API_KEY


class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt  # 聊天开始时的系统提示
        self.num_of_round = num_of_round  # 聊天回合数
        self.messages = []  # 保存聊天历史
        self.messages.append({"role": "system", "content": self.prompt})

    def chat_generator(self):
        while True:
            user_input = input("User: ")
            self.messages.append({"role": "user", "content": user_input})
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.messages,
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=1,
                )
            except Exception as e:
                print(e)
                yield str(e)

            message = response.choices[0].message['content']
            self.messages.append({"role": "assistant", "content": message})
            # 如果聊天历史的消息数超过了指定的回合数，删除聊天历史的第一轮对话，以节省模型的 token 使用
            if len(self.messages) > self.num_of_round * 2 + 1:
                del self.messages[1:3]  # 为了保持系统提示，保留了前 1 条消息

            yield message


class Conversation2:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
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
        num_of_tokens = response['usage']['total_tokens']  # 使用的总 token 数
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round * 2 + 1:
            del self.messages[1:3]
        return message, num_of_tokens


if __name__ == '__main__':
    prompt1 = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内"""
    conversation = Conversation(prompt1, 5)
    chat_gen = conversation.chat_generator()

    for answer in chat_gen:
        print("Chatbot:", answer)

    # conv2 = Conversation2(prompt, 3)
    # questions = [question1, question2, question3]
    # for question in questions:
    #     answer, num_of_tokens = conv2.ask(question)
    #     print("询问 {%s} 消耗的token数量是 : %d" % (question, num_of_tokens))
