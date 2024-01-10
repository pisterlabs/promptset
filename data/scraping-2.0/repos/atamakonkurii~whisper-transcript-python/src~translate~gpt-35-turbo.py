import os
import requests
import json
import re
from dotenv import load_dotenv
import openai

# transcriptJapanese.srtファイルをdeepLで翻訳
def translate_text(text, api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages = [
        {"role": "user", "content": "Translate the following Japanese text to Chinese(traditional): " + text},
      ],
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

def translate_srt_file(input_file, output_file, api_key):
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if re.match("^\d+$", line.strip()) or re.match("^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$", line.strip()):
                fout.write(line)
            elif line.strip():
                translated_line = translate_text(line.strip(), api_key)
                fout.write(translated_line + "\n")
            else:
                fout.write("\n")

# 環境変数を読み込む
load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]  # ここにDeepL APIキーを入力してください
input_file = "./output/srt/transcriptJapanese.srt"  # 変換するSRTファイルの名前
output_file = "./output/srt/transcriptTaiwanese.srt"  # 変換後のSRTファイルの名前

translate_srt_file(input_file, output_file, api_key)

