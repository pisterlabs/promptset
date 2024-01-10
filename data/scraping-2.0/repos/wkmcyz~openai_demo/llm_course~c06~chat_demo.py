import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")


class Conversation:
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
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round * 2 + 1:
            del self.messages[1:3]  # Remove the first round conversation left.
        return message


if __name__ == '__main__':
    prompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求。
    1. 你的回答必须是中文
    2. 回答需要限制在100个字之内
    """
    conv1 = Conversation(prompt, 2)
    q1 = "你是谁？"
    print(f"Q: {q1}")
    print(f"A: {conv1.ask(q1)}")

    q2 = "你会做什么菜？"
    print(f"Q: {q2}")
    print(f"A: {conv1.ask(q2)}")

    q3 = "小炒牛肉怎么做？"
    print(f"Q: {q3}")
    print(f"A: {conv1.ask(q3)}")

    q4 = "我问你的第一个问题是什么？"
    print(f"Q: {q4}")
    print(f"A: {conv1.ask(q4)}")