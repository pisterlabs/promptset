import arxiv
import openai
import datetime
import json
import os
from skpy import Skype
import setting

config_path = os.path.join(os.path.dirname(__file__), "config.json")
# OpenAIのapiキー
openai.api_key = setting.OPENAI_API_KEY

# Skypeのログイン情報
USER     = setting.SKYPE_USERNAME
PWD      = setting.SKYPE_PASSWORD
GROUP_ID = setting.SKYPE_GROUP_ID

with open(config_path) as f:
    config = json.load(f)
    key_word = config["key_word"]

def get_summary(result):
    system = """与えられた論文の要点を3点のみでまとめ、以下のフォーマットで日本語で出力してください。```
    タイトルの日本語訳
    ・要点1
    ・要点2
    ・要点3
    ```"""

    text = f"title: {result.title}\nbody: {result.summary}"
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': text}
                ],
                temperature=0.25,
            )
    summary = response['choices'][0]['message']['content']
    title_en = result.title
    title, *body = summary.split('\n')
    body = '\n'.join(body)
    date_str = result.published.strftime("%Y-%m-%d %H:%M:%S")
    message = f"発行日: {date_str}  {result.entry_id}\n{title_en}\n{title}\n{body}\n"

    return message


def main(key_word):
    sk = Skype(USER, PWD)
    ch = sk.chats.chat(GROUP_ID)
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # queryを用意、今回は、三種類のqueryを用意
    query = f'abs:%22 {key_word} %22'

    # arxiv APIで最新の論文情報を取得する
    search = arxiv.Search(
        query=query,                                # 検索クエリ
        max_results=100,                            # 取得する論文数
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 論文を投稿された日付でソートする
        sort_order=arxiv.SortOrder.Descending,      # 新しい論文から順に取得する
    )
    # searchの結果をリストに格納
    result_list = []
    for result in search.results():
        if yesterday == result.published.strftime("%Y-%m-%d"):
            result_list.append(result)

    if len(result_list) == 0:
        message = "昨日の論文はありませんでした。"
        print(message)
        return

    # 論文情報をskypeに投稿する
    for i, result in enumerate(result_list):
        try:
            # skypeに投稿するメッセージを組み立てる
            message = "ArxivEの論文です！ " + str(i+1) + "本目  " + get_summary(result)
            # skypeにメッセージを投稿する
            ch.sendMsg(message)
            print(f"Message posted: {message}")
        except Exception as e:
            print(f"Error posting message: {e}")


if __name__ == '__main__':
    main(key_word)
