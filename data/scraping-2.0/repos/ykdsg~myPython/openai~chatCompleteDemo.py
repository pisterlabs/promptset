import os
import openai
from dotenv import load_dotenv

# openai chatComplete例子

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def demo():
    # system：指示 API 如何行为。基本上它是 OpenAI 的主要提示。
    # user：你想问的问题。它是单个或多个对话中的用户输入。它可以是多行文本。
    # assistant：当你编写一段对话时，你需要使用这个角色来附加响应。以便 API 记住讨论的内容。
    # openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Who won the world series in 2020?"},
    #         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    #         {"role": "user", "content": "Where was it played?"}
    #     ])
    # 在一条消息中，可以发送多个角色。上面代码片段中显示的行为、您的问题和历史记录。

    # 定义一个数组来保存 OpenAI 的整个消息。然后向用户显示提示并接受system指令。
    messages = []
    system_message = input("What type of chatbot you want me to be?")
    messages.append({"role": "system", "content": system_message})

    print("Alright! I am ready to be your friendly chatbot" + "\n" + "You can now type your messages.")
    message = input("")
    messages.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    reply = response["choices"][0]["jmessage"]["content"]
    print(reply)


if __name__ == '__main__':
    demo()
