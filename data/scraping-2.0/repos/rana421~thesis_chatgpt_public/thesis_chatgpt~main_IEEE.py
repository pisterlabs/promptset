import openai
import datetime
import json
import os
import requests
from skpy import Skype
from operator import itemgetter
import setting

config_path = os.path.join(os.path.dirname(__file__), "config.json")
# OpenAIのapiキー
openai.api_key = setting.OPENAI_API_KEY

# Skypeのログイン情報
USER         = setting.SKYPE_USERNAME
PWD          = setting.SKYPE_PASSWORD
GROUP_ID     = setting.SKYPE_GROUP_ID
IEEE_API_KEY = setting.IEEE_API_KEY

with open(config_path) as f:
    config = json.load(f)
    key_word = config["key_word"]

IEEE_base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"


def get_summary(result):
    system = """与えられた論文の要点を3点のみでまとめ、以下のフォーマットで日本語で出力してください。```
    タイトルの日本語訳
    ・要点1
    ・要点2
    ・要点3
    ```"""

    text = f"title: {result['title']}\n body: {result['abstract']}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text}
        ],
        temperature=0.25,
    )
    summary = response['choices'][0]['message']['content']

    title_en = result['title']
    title, *body = summary.split('\n')
    body = '\n'.join(body)
    date_str = result['insert_date']
    message = f"発行日: {date_str}\n[{title_en}]({result['pdf_url']})\n{title}\n{body}\n"
    return message


def main(key_word):
    sk = Skype(USER, PWD)
    ch = sk.chats.chat(GROUP_ID)
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y%m%d")

    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    # クエリパラメータを指定してAPIを呼び出す
    params = {
        "apikey": IEEE_API_KEY,
        "format": "json",
        "abstract": key_word,
        "start_year": current_year,
        "end_year": current_year,
        "start_month": current_month,
        "end_month": current_month,
        "max_records": 200
    }

    response = requests.get(IEEE_base_url, params=params)
    if response.status_code == 200:
        print("API call successful")
    else:
        print("API call unsuccessful with status code:", response.status_code)

    # APIからのレスポンスをJSON形式で取得
    result = json.loads(response.text)

    articles_sorted_by_date = sorted(
        result['articles'],
        key=itemgetter('insert_date'),
        reverse=True
    )

    result_list = []
    for article in articles_sorted_by_date:
        if yesterday == article['insert_date']:
            result_list.append(article)

    if len(result_list) == 0:
        message = "昨日の論文はありませんでした。"
        print(message)
        return

    # 論文情報をskypeに投稿する
    for i, result in enumerate(result_list):
        try:
            # skypeに投稿するメッセージを組み立てる
            message = f"IEEEの論文です！ " + str(i+1) + "本目  " + get_summary(result)
            # skypeにメッセージを投稿する
            ch.sendMsg(message, rich=True)
            print(f"Message posted: {message}")
        except Exception as e:
            print(f"Error posting message: {e}")


if __name__ == '__main__':
    main(key_word)
