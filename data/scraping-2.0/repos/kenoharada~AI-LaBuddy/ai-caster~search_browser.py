import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import requests
import openai
import os
import glob
import json
import requests
from typing import List
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

MODEL_NAME = 'gpt-3.5-turbo-16k'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'


# Googleで検索を行う
def search_google(query):
    # ブラウザのドライバを指定
    driver = webdriver.Chrome()
    # Googleを開く
    driver.get("http://www.google.com")
    time.sleep(1)
    
    # 検索窓に検索語を入力し、検索を実行
    # 検索窓がロードされるまで待つ
    search_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "q")))

    # 検索窓に検索語を入力し、検索を実行
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    
    time.sleep(2)  # ページがロードされるまで待つ

    # 検索結果を取得
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    search_results = soup.find_all('div', attrs={'class': 'yuRUbf'})
    print(search_results)
    
    results = []
    for result in search_results[:3]:
        title = result.find('h3').text
        url = result.find('a').get('href')
        results.append({'Title': title, 'URL': url})
        # URL先のページのbodyを取得
        driver.get(url)
        time.sleep(2)
        page_soup = BeautifulSoup(driver.page_source, 'html.parser')
        page_body = page_soup.body.text if page_soup.body else ''
        if page_body != '':
            results.append({'Title': title, 'URL': url, 'Page': page_body})
    driver.quit()
    
    return results

# HTMLを作成し、ブラウザで開く
def create_html(results, filename):
    # PandasのDataFrameに変換
    df = pd.DataFrame(results)

    # DataFrameをHTMLに変換
    html = df.to_html(index=False)

    # HTMLファイルを書き出し
    with open(filename, 'w') as f:
        f.write(html)

    # ブラウザでHTMLを開く
    os.system(f'open {filename}')


def create_keyword(user_query='OpenAIについて教えて'):
    response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": 'ユーザーの疑問に答えるためにGoogle検索を行います、Googleの検索窓に入力するキーワードや検索オプションを出力してください。出力はそのまま検索窓に入力します。ユーザーの疑問に答えるために必要な観点を考え、それを調べるためにgoogleの検索窓に入力するキーワードをスペース区切りで考えてください。出力は以下のフォーマットで3つ出力してください。観点: {} keywords: {}'},
                    {"role": "user", "content": user_query},
                ],
                temperature=0
            )['choices'][0]['message']['content']
    
    return response


def create_summary(user_query, results):
    context = ''
    for result in results:
        if 'Page' in result:
            context += f"title: {result['Title']} url: {result['URL']} content: {result['Page']}\n"
    response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f'以下の文章をまとめて、{user_query}に答えるようなレポートをHTMLとして作成してください。参考にした部分は[1]のような引用表記とレポートの最後に出典を明示して下さい。説明がよく伝わるように要点の概要図も図表と文字を組み合わせてページの上部に配置してください。出力はHTMLでお願いします。'},
                    {"role": "user", "content": context},
                ],
                temperature=0
            )['choices'][0]['message']['content']
    
    return response


import re

def extract_keywords(text):
    # 正規表現パターンでキーワードを抽出する
    pattern = r'keywords:\s*(.+?)(?=\n\n|\n|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    keywords_list = [match.strip().split() for match in matches]
    return keywords_list

if __name__ == "__main__":
    # 検索語を指定
    user_query = "松尾研について教えて"
    result = create_keyword(user_query=user_query)
    print(result)
    keywords_list = extract_keywords(result)
    for keywords in keywords_list:
        query = ' '.join(keywords)
        print(query)
        break

    # Googleで検索を行う
    results = search_google(query)
    # from openai.embeddings_utils import get_embedding
    # for result in results:
    #     try: 
    #         result['embedding'] = get_embedding(
    #         result['Page'],
    #         engine="text-embedding-ada-002"
    #         )
    #     except Exception as e:
    #         print(e)
    #         continue

    

    # HTMLを作成し、ブラウザで開く
    create_html(results, "results.html")

    html = create_summary(user_query, results)

    with open('report.html', 'w') as f:
        f.write(html)
    os.system('open report.html')