import asyncio
import json
import logging
import os
from datetime import datetime
import traceback
import aiohttp
import base64
from backoff import expo, on_exception
from google.cloud import pubsub_v1
from gspread import service_account_from_dict
from html2text import html2text
from openai import OpenAI, AsyncOpenAI
from urllib.parse import urlparse
import gspread
import openai
import time

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 環境変数からスプレッドシートIDとクレデンシャルを取得
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID') 
GOOGLE_CREDENTIALS_BASE64 = os.getenv('CREDENTIALS_BASE64')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#　スクレイピングできなさそうなところ、できないものを追加しておく。
EXCLUDED_DOMAINS = ['github.com', 'youtube.com', 'wikipedia.org']

def init_openai():
  return OpenAI(api_key=OPENAI_API_KEY)

# Base64エンコードされたGoogleクレデンシャルをデコードし、gspreadクライアントを認証
try:
    creds_json = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
    creds = json.loads(creds_json)
    gc = gspread.service_account_from_dict(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).get_worksheet(1)  # sheet2に対応するインデックスを指定
except Exception as e:
    logging.error(f"gspreadクライアントの認証中にエラーが発生しました: {e}")
    raise

# OpenAIのクライアントを初期化
client = OpenAI(api_key=OPENAI_API_KEY)


# 非同期用のOpenAIクライアント
async_client = AsyncOpenAI()

async def fetch_content_from_url(url):
    try:
        logging.INFO(f"URLからコンテンツの取得を開始: {url}")

        # ユーザーエージェントを設定
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=100) as response:
                content = await response.text()

            logging.INFO(f"URLからコンテンツの取得が成功: {url}")
            return content

    except Exception as e:
        logging.error(f"URLからのコンテンツ取得中にエラーが発生しました: {e}")
        raise

async def openai_api_call(model, temperature, messages, max_tokens, response_format):
    try:
        # OpenAI API呼び出しを行う非同期関数
        response = await async_client.chat.completions.create(model=model, temperature=temperature, messages=messages, max_tokens=max_tokens, response_format=response_format)
        return response.choices[0].message.content  # 辞書型アクセスから属性アクセスへ変更
    except Exception as e:
        logging.error(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
        raise

async def summarize_content(content):
    try:
        summary = await openai_api_call(
        "gpt-3.5-turbo-1106",
        0,
        [
            {"role": "system", "content": f'あなたは優秀な要約アシスタントです。"""{content}"""の内容をできる限り多くの情報を残しながら日本語で要約して出力してください。'},
            {"role": "user", "content": content}
        ],
        2800,
        # タイプ指定をサボらない
        { "type": "text" }
        )
        return summary
    except Exception as e:
        print(f"要約時にエラーが発生しました。: {e}")
        traceback.print_exc()
        return ""

paramater = '''
{
    "properties": {
        "importance": {
            "type": "integer",
            "description": "How impactful the topic of the article is. Scale: 0-10."
        },
        "timeliness": {
            "type": "integer",
            "description": "How relevant the information is to current events or trends. Scale: 0-10."
        },
        "objectivity": {
            "type": "integer",
            "description": "Whether the information is presented without bias or subjective opinion. Scale: 0-10."
        },
        "originality": {
            "type": "integer",
            "description": "The novelty or uniqueness of the content. Scale: 0-10."
        },
        "target_audience": {
            "type": "integer",
            "description": "How well the content is adjusted for a specific audience. Scale: 0-10."
        },
        "diversity": {
            "type": "integer",
            "description": "Reflection of different perspectives or cultures. Scale: 0-10."
        },
        "relation_to_advertising": {
            "type": "integer",
            "description": "If the content is biased due to advertising. Scale: 0-10."
        },
        "security_issues": {
            "type": "integer",
            "description": "Potential for raising security concerns. Scale: 0-10."
        },
        "social_responsibility": {
            "type": "integer",
            "description": "How socially responsible the content presentation is. Scale: 0-10."
        },
        "social_significance": {
            "type": "integer",
            "description": "The social impact of the content. Scale: 0-10."
        }
        "reason": {
        "type": "string",
        "description": "the basis for each numerical score. Output in 1-sentence Japanese with respect to all parameters"
        }
    },
    "required": ["importance", "timeliness", "objectivity", "originality", "target_audience", "diversity", "relation_to_advertising", "security_issues", "social_responsibility", "social_significance", "reason"]
}
'''
    

async def generate_score(summary):
    try:
        score = await openai_api_call(
            "gpt-3.5-turbo-1106",
            0,
            [
                {"role": "system", "content": f'あなたは優秀な先進技術メディアのキュレーターです。信頼性,最新性,重要性,革新性,影響力,関連性,包括性,教育的価値,時事性,倫理性をもとに、"""{summary}"""を10点満点でスコアリングして、JSON形式で返します。平均点は5点でスコアを付けるようにしてください。"""{paramater}"""のJSON形式で返してください。'},
                {"role": "user", "content": summary}
            ],
            4000,
            { "type":"json_object" }
            )
        return score
    except Exception as e:
        print(f"スコア測定時にエラーが発生しました。: {e}")
        traceback.print_exc()
        return ""

# カテゴリー、要約、リード文を生成する非同期関数
async def generate_textual_content(content):
    # 先に要約を行う
    summary = await summarize_content(content)
    
    # 要約に基づきリード文を生成
    score = await generate_score(summary)
    score_json = json.loads(score)
    #それぞれの内容を取得
    keys = ["importance", "timeliness", "objectivity", "originality", "target_audience", "diversity",  "relation_to_advertising", "security_issues", "social_responsibility", "social_significance"]
    #scoresはスコアの中身全部のこと
    scores = [str(score_json[key]) for key in keys]
    reason = score_json["reason"]
    #それぞれの内容を返す。
    return summary, *scores, reason
# Function to write to the Google Sheet with exponential backoff
@on_exception(expo, gspread.exceptions.APIError, max_tries=3)
@on_exception(expo, gspread.exceptions.GSpreadException, max_tries=3)
def write_to_sheet_with_retry(row):
    time.sleep(1)  # 1秒スリープを追加
    try:
        logging.INFO("Googleスプレッドシートへの書き込みを開始")
        sheet.insert_row(row, index=2)
        logging.INFO("Googleスプレッドシートへの書き込みが成功")
    except Exception as e:
        logging.error(f"Googleスプレッドシートへの書き込み中にエラーが発生しました: {e}")
        raise

# Function to process content and write it to the sheet
async def process_and_write_content(title, url):
    # URLからドメインを解析
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # 特定のドメインをチェックしてスキップ
    if any(excluded_domain in domain for excluded_domain in ['github.com', 'youtube.com', 'wikipedia.org']):
        logging.INFO(f"処理をスキップ: {title} ({url}) は除外されたドメインに属しています。")
        return

    logging.INFO(f"コンテンツ処理が開始されました: タイトル={title}, URL={url}")
    html_content = await fetch_content_from_url(url)
    text_content = html2text(html_content)
    summary, scores, reason = await generate_textual_content(text_content)
    # 時刻
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [title, url, now, scores, reason, summary]
    write_to_sheet_with_retry(row)

# Main function to be called with the news data
def main(event, context):
    try:
        news_data = json.loads(base64.b64decode(event['data']).decode('utf-8'))
        title = news_data.get('title')
        url = news_data.get('url')
        if title and url:
            asyncio.run(process_and_write_content(title, url))
    except Exception as e:
        logging.error(f"メイン処理中にエラーが発生しました: {e}")
