import json
import openai


def ask_gpt(msg):
    openai.api_key = '自己的key'
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": msg}
        ]
    )
    result = json.loads(str(completion.choices[0].message))
    return result['content']


if __name__ == '__main__':
    answer = ask_gpt("你好，你是谁啊")
    print(answer)
