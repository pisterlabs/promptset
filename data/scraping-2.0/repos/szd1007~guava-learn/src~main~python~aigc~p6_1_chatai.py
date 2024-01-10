import openai
import os


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
                #model="gpt-3.5-turbo",
                model="gpt-4-1106-preview",  # 9月出账单就可以用最新模型了，搭个网站自己用 https://platform.openai.com/docs/models/gpt-4
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e
        message = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            # remove the first round conversation
            del self.messages[1:3]
        return message


prompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答要满足以下要求
1. 你的回答必须是中文
2. 回答限制在100个字以内"""

conv1 = Conversation(prompt, 2)
question1 = "你是谁？"
print("User :%s" % question1)
print("Assistant : %s\n" % conv1.ask(question1))

question2 = "请问鱼香肉丝怎么做？"
print("User :%s" % question2)
print("Assistant : %s\n" % conv1.ask(question2))

question3 = "那蚝油牛肉呢？"
print("User :%s" % question3)
print("Assistant : %s\n" % conv1.ask(question3))
