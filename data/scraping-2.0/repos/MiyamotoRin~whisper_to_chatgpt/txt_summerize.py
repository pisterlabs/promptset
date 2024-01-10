import os
import sys
import openai
from dotenv import load_dotenv

load_dotenv(".env")
openai.api_key = os.environ.get("OPENAI_API_KEY")

def summerize(input_file):
  # UTF8エンコードのテキストファイルの読み込み
  with open(input_file, encoding='utf-8') as f:
      text = f.read()
      
  # テキストを区切ってリスト化
  overlap = 100
  length = 1000
  text_list = [text[i:i+length] for i in range(0, len(text)-length+1, length-overlap)]

  # 区切った文字列リストをmessagesに格納
  send_messages = [{"role": "user", "content": "以下の文章を1000字以内で要約してください。"}]
  for text in text_list:
      send_messages.append({"role": "assistant", "content": text})

  completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=send_messages,
    temperature=0.7,
  )

  with open("./sum.txt", mode='w', encoding='utf-8') as f:
      f.write(completion.choices[0].message.content)
      
  return completion.choices[0].message.content

if __name__ == "__main__":
  if len(sys.argv) > 1:
      input_file = sys.argv[1]
      print(summerize(input_file))
  else:
      print("Usage: python txt_summerize.py <path_to_text_file>")