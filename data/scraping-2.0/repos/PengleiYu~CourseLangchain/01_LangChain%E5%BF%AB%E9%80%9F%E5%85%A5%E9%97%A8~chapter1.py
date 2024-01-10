# https://time.geekbang.org/column/article/699400

import openai
from langchain.llms import OpenAI

# text模型单轮，输出多个结果
# response = openai.Completion.create(
#     model="text-davinci-003",
#     temperature=0.5,
#     max_tokens=100,
#     prompt="请给我的花店起个名")
# print(response.choices[0].text.strip())
# print(response)

# chat模型，接近人类回答
# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": "You are a creative AI."},
#         {"role": "user", "content": "请给我的花店起个名"},
#     ],
#     temperature=0.8,
#     max_tokens=60
# )
# print(response['choices'][0]['message']['content'])
# print(response)


# langchain封装的text模型
# llm = OpenAI(
#     model="text-davinci-003",
#     temperature=0.8,
#     max_tokens=100, )
# response = llm.predict("请给我的花店起个中文名,结果请配上英文翻译")
# print(response)


# langchain封装的chat模型
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

chat = ChatOpenAI(model="gpt-4",
                  temperature=0.8,
                  max_tokens=60)

messages = [
    SystemMessage(content="你是一个很棒的智能助手，并且回答很简洁"),
    HumanMessage(content="请给我的花店起个名，并翻译为英文")
]
response = chat(messages)
print(response)
