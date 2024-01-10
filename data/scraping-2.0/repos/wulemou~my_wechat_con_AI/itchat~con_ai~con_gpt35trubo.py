# create by h3c s31362


import os
from dotenv import load_dotenv
from openai import OpenAI


# 读取本项目内的 .env 文件变量
env_path = os.path.join(os.path.dirname(__file__), "..", '.env')
load_dotenv(env_path)


my_35_key = os.getenv('GPT35APIKEY')


client = OpenAI(api_key=my_35_key)


def gpt35_send_rcv(message: str) -> str:
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are chatgpt, a personal assistant, to help your questioner, to think in order "
                                    "to accomplish its goal, to think carefully about whether his question is so simple,"
                                    " to give a human answer.."},
      {"role": "user", "content": message}
    ]
  )

  print(completion.choices[0].message)

  return str(completion.choices[0].message)

if __name__ == '__main__':
    gpt35_send_rcv("你是谁?")


