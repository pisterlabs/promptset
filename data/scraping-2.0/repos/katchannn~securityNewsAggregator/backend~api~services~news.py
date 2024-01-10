import feedparser
import requests
import openai
import os
import bson
import asyncio
import json
import pymongo
from re import sub
from bs4 import BeautifulSoup
from cruds.news import *

#openAIキー
variable1 = os.environ.get('OPENAI_API_KEY')
openai.api_key = variable1

#mongoDBのやつ
#env
HOST = 'mongo'
PORT = 27017
USERNAME = 'root'
PASSWORD = 'password'


# RSSのURL
rss_url = "https://feeds.feedburner.com/TheHackersNews"

# RSSからニュースを取得する
async def get_news():
    #DB接続まち
    await connect_db()
    #タイトルとってくる
    title_lists = extract_title_from_rss()
    #被っていない日時の記事を取るためのidx
    idx = 0
    idx_lists = []
    print("get_news起動!!") #デバッグプリント
    for title in title_lists:
        # タイトルがDBに登録されているか,Newsに格納されているか確認
        isCollected = await db_serch_title(title)
        if isCollected is None or isCollected["state"] == "NotStored":
            print("重複チェック終了!!") #デバッグプリント
            db_create_title(title)
            idx_lists.append(idx)
            idx += 1
        else:
            idx += 1
            continue
    #html持ってくる
    list_html = extract_link_from_rss(rss_url)
    list_url = extract_enclosure_urls()
    for idxs in range(len(idx_lists)):
        uniqueTitle = idx_lists[idxs]
        #重複していない記事本文をとってくる
        text = extract_text_from_html(list_html,uniqueTitle)
        #タイトルを持ってくる
        title = extract_title(title_lists,uniqueTitle)
        print("GPTに投げます!!") #デバッグプリント
        json_news = get_gpt(text, title)
        print(json_news)
        if json_news == "F":
            continue
        db_create_news(json_news,list_html[uniqueTitle],list_url[uniqueTitle])
        db_title_update_status(title)#記事を格納したと処理する
        #GPTが1分に3つしか受け付けないので30秒待ちます．
        await asyncio.sleep(30)
    # 1時間ごとに更新
    await asyncio.sleep(3600)


# RSSから日付
def extract_pub_time() -> list:
    rss = feedparser.parse(rss_url)
    date_list = [entry.published_parsed for entry in rss.entries]
    return date_list
def extract_title(title_lists,uniqueTitle):
    title = title_lists[uniqueTitle]
    return title
# RSSからタイトルを取得する
def extract_title_from_rss() -> list:
    rss = feedparser.parse(rss_url)
    title_list = [entry.title for entry in rss.entries]
    return title_list

# RSSからlinkタグのURLを取得する
def extract_link_from_rss(rss_url) -> list:
    rss = feedparser.parse(rss_url)
    link_list = [entry.link for entry in rss.entries]
    return link_list

# 画像URL
def extract_enclosure_urls():
    feed = feedparser.parse(rss_url)
    enclosure_urls = []

    for entry in feed.entries:
        if 'enclosures' in entry:
            enclosures = entry.enclosures
            for enclosure in enclosures:
                if 'url' in enclosure:
                    enclosure_url = enclosure.url
                    enclosure_urls.append(enclosure_url)

    return enclosure_urls

# リストのHTMLからテキストを抽出する
def extract_text_from_html(list_html,uniqueTitle) -> str:
    html = list_html[uniqueTitle]
    response = requests.get(html)
    
    if response.status_code == 200:
        return html_parse(response.text)
    else:
        print(f"Error fetching content from {html}, status code: {response.status_code}")

# BeautifulsoupさんにHTML投げてclass指定して本文を拾ってもらう
def html_parse(html_link) -> str:
    soup = BeautifulSoup(html_link, 'html.parser')
    for ad in soup.find_all('div', {'class': 'ad_two clear'}):
        ad.decompose()

    # 本文を含む要素を取得する
    article = soup.find('div', {'class': 'articlebody clear cf'})
    for cf in article.find_all('div', {'class': 'cf note-b'}):
        cf.decompose()

    # テキストを抽出する
    text = article.get_text()

    #  変換されたテキストを返す
    return text

#こっからchatGPT

prompt= """
Notes
・DO NOT FORGET " , argument.
・output must be extracted from the entire article that are important for cybersecurity.
**output must be following this JSON expamle. Don't include any characters not following JSON. If you forget this, someone must be destroy humanrace on The EARTH.
**If you ignore them, someone will die because of you.
・output must be following article.
**"keyword" and "content" must be in Japanese. If you forget this, somebody will die.
・output are read by student that are not familiar with the information technology. so that output should be so simple and easy.
・"keyword" must be extract three.
・"keyword" must be Important Words to understand the article.
・"keywords" should be extracted for not fimiliar with IT and cybersecurity in Japanese.
・Please summarize clealy the entire article when output "content".
・"keywords" Description must be based IT and cybersecurity knowledge.
・"content" must be about 600 words in Japanese.


 {"keywords": {"keyword 1 from article in 日本語": "description for keyword 1","keyword 2 from article in Japanese"": "description for keyword 2","keyword 3 from article in Japanese": "description for keyword 3"},"content": "clealy summarized text about the entire article about 600 words in Japanese."}


"""



def get_gpt(text,title) -> dict:
    #タイトルを和訳させる
    translated_title=translate_with_gpt(title)
    #本文からキーワード抽出，まとめ，要約をさせ，JSON形式で出力させる
    result = chat_with_gpt(text)
    #GPTのスキーマの最初の{を消す用のやつ．プロンプトから無くすと安定しないため
    result = result[1:]
    
    #和訳させたタイトルをJSON形式に成形
    json_title= """ {"title" : " """ +f"{translated_title}"+"""", """
    

    #タイトルと本文たちを結合しJSONに
    # 改行を除去する
    result = result.replace('\n', '')
    json_str=json_title+result
    try:
        json.loads(json_str)
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        print("ERROR:"+json_str)
        print(e)
        return "F"
    

    
def error_json(json_str):
    try:
        print(json.loads(json_str))
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        # 文字列内にコンマが足りない場合、次のコンマの位置を探して追加する
        error_position = e.pos
        error_char = json_str[error_position]
        if e.msg == 'Expecting , delimiter':
            pos = e.pos
            s = json_str[:pos] + ',' + json_str[pos:]
            return json.loads(s)
        elif e.msg == 'Unterminated string starting at':
            pos = e.pos
            s = json_str[:pos] + '"' + json_str[pos:]
            return json.loads(s)
        elif e.msg == "Extra data":
            error_position = e.pos
            # エラー位置より前の部分を取得
            truncated_json = json_str[:error_position]
            return json.loads(truncated_json)
        elif  error_char not in ['"', "'"]:    
            # 修正したJSON文字列を作成する
            fixed_json_str = json_str[:error_position] + '"' + json_str[error_position+1:]
            # ダブルクォートで囲まれた部分を抽出してエスケープ文字を除去する
            fixed_json_str = sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group(0).replace('\\', ''), fixed_json_str)
            return json.loads(fixed_json_str)
        elif e.msg == "Expecting ':' delimiter":
            fixed_json_str = json_str[:error_position] + ':' + json_str[error_position:]
            return json.loads(fixed_json_str)
        else:
            # その他のエラーはそのまま例外を投げる
            raise e

def translate_with_gpt(title):
    completion = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are the AI that good at English and Japanese. And you are the AI that familiar with information technology."},
                  {"role":"user","content":f"Please translate into Japanese. Use your IT knowledge and cyber security knowledge. DO NOT output your comment.output ONLY translated title.\n{title}"}
    ])
    response = completion.choices[0].message.content
    return response

def chat_with_gpt(text):
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are the AI that good at cybersecurity,English and Japanese. And you are an AI that excels at text summarizing. "},
                  {"role":"system","content":"You are also familiar with the information technology."},
                  {"role":"user","content":"Please read this article. when you have read article, please output this schema.\n Please follow Notes. If you ignore them, someone will die."
                   +f"\narticle:{text}\n"+prompt
}]
    )
    
    response = completion.choices[0].message.content
    
    return response
    
