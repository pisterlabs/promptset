# -*- coding: utf-8 -*-

"""リクエストサンプル

OpenAI APIを使用して、単純なリクエストを送るサンプル

"""

import openai
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
import sys
import logging
import json


def get_completion(prompt: str, engine: str) -> str:
    """
    プロンプトを受け取り、OpenAIにリクエストを送信して、結果を返す

    Parameters
    ----------
    prompt : str
        プロンプト
    model : str, optional
        エンジンの指定, by default "gpt-3.5-turbo"
    Returns
    -------
    str
        OpenAIからのレスポンス
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        # エンジンの指定
        engine=engine,
        deployment_id=engine,
        # ユーザーの発言として、日本語を入力
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0]["message"]["content"]


# .envファイルから環境変数を読み込み
load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")
# APIキーを環境変数から取得
openai.api_key: str = token.token
# openai.api_key: str = os.getenv("OPENAI_API_KEY")
openai.api_base: str = os.getenv("OPENAI_API_BASE")
openai.api_version: str = os.getenv("OPENAI_API_VERSION")
openai.api_type: str = os.getenv("OPENAI_API_TYPE")
ai_model: str = os.getenv("AZURE_MODEL")
dir_path = os.path.dirname(os.path.realpath(__file__))
csv_file_name: str = os.path.join(dir_path, "rules.csv")
# CSVの読み込み。一行目をヘッダーとして、カンマ区切りで読み込む
import pandas as pd

df = pd.read_csv(csv_file_name, header=0, sep=",")

result_json_list = []
# 一行ずつ処理
for index, row in df.iterrows():
    # CSVデータの解釈してその意味を返却させるプロンプトを定義、行を文字列として渡す
    prompt: str = f"""Could you put the first two together and interpret the meaning and return its meanful summary and url in JSON (sumaary,url) in Japanese? you don't need to return originals : ```{row["想定問い合わせ例文"],row["回答"],row["URL"]}```"""
    # 結果のJSONをリストに格納する
    completion_result_org = get_completion(prompt, ai_model)
    # JSONは文字列なので、辞書型に変換する
    try:
        completion_result = json.loads(completion_result_org)
        result_json_list.append(completion_result)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"JSON string: {completion_result_org}")

# 結果をJSONファイル（UTF-8）に出力
with open(os.path.join(dir_path, "result.json"), "w", encoding="utf-8") as f:
    json.dump(result_json_list, f, ensure_ascii=False, indent=4)
