import os
from dotenv import load_dotenv
import openai

# .envファイルのパスを取得
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

openai.api_key = os.getenv("CHATGPT_API_KEY")
print(openai.api_key)

def Answer_ChatGPT(question):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": "次の文章を画像生成用プロンプト(英語)に変換してください：" + question,
        }]
    )

    response = completion.choices[0].message['content']
    return response

if __name__ == '__main__':
    q = input("質問内容:")
    print(Answer_ChatGPT(q))
