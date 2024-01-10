
from openai import OpenAI
import os
from dotenv import load_dotenv

def main():
  load_dotenv()

  client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
  )

  response = client.chat.completions.create(
      model=os.getenv('FT_MODEL'),
      messages=[
          {"role": "system", "content": "Marvは事実に基づいたチャットボットであり、皮肉も言います。"},
          {"role": "user", "content": "温室効果ガス排出量を削減する上で最大の課題は何ですか？"}
      ]
  )

  print(response)

if __name__ == '__main__':
  main()
