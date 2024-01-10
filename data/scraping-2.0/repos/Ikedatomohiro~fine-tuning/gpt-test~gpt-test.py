from openai import OpenAI
import os
from dotenv import load_dotenv

def main():
  load_dotenv()

  client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY'),
  )

  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたは、Go言語に詳しいエンジニアです。Goに関する質問に回答してください。"},
        {"role": "user", "content": "Go言語の特徴はなんですか？"}
    ]
  )

  print(response)

if __name__ == '__main__':
  main()
