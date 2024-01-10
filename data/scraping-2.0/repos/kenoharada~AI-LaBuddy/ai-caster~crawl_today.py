import json
import requests
from bs4 import BeautifulSoup

url = "https://kids.yahoo.co.jp/today/"

# URLにアクセスしてHTMLを取得
response = requests.get(url)
html = response.text

soup = BeautifulSoup(html, 'html.parser')
today = soup.find('h2', id='date').find('span').get_text(strip=True)
date_element = soup.find('dl', id='dateDtl')

if date_element is not None:
    date = date_element.find('span').get_text(strip=True)
    content = date_element.find('dd').get_text(strip=True)

    data = {
        'today': today,
        'date': date,
        'text': content
    }
    print(data)

    json_data = json.dumps(data, ensure_ascii=False)

    # ファイルへの保存例
    with open('today_info.json', 'w', encoding='utf-8') as file:
        file.write(json_data)
else:
    print("要素が見つかりませんでした。")
import re

def parse_voice(text):
    # パターンを定義します：名前と発言
    pattern = re.compile(r'(\w+): (.+?)(?=\w+:|$)', re.DOTALL)

    # 文字列から一致する部分をすべて見つける
    matches = pattern.findall(text)

    # マッチした結果を指定の形式に変換
    dialogue_list = [{'person': m[0], 'text': m[1].strip()} for m in matches]
    return dialogue_list

import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# MODEL_NAME = 'gpt-3.5-turbo-16k'
MODEL_NAME = 'gpt-4'
contents = f"{data['today']} {data['date']} {data['text']}"
system_message = f"""
テレビ番組で、アナウンサーの森さんと国民的人気アイドルの愛ちゃん2人が「{contents}」について話し合っています。
アナウンサーは「{contents}」について簡単な話題提供を行いつつ、話を振ります。
アイドルは、元気はつらつに愛嬌よく面白おかしいエピソードを話します。
話題について一通り盛り上がったらアナウンサーが「次の話題に移ります」と言って会話が終了します。
一連の会話の流れを文章に起こしてください。
フォーマットは
アナウンサー: 発言
アイドル: 発言
のようにお願いします。
"""

response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": system_message}
                ],
                temperature=0.7,
            )['choices'][0]['message']['content']
print(response)
print(parse_voice(response))
dialogue_list = parse_voice(response)
with open('today_dialogue.json', 'w', encoding='utf-8') as file:
    json.dump(dialogue_list, file, ensure_ascii=False)


