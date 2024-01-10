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
from datetime import datetime, timedelta

# 現在の日付を取得
today = datetime.now().date()

# 1日前の日付を計算
one_day_ago = today - timedelta(days=1)
# 日付を指定された形式で表示
date = one_day_ago.strftime("%Y-%m-%d")

# 検索クエリ
# https://kinsta.com/jp/blog/google-search-operators/
query = f'"generative AI" OR "GPT" OR "LLM" site:https://techcrunch.com/ after:{date}'
# 営業 GPT site:https://twitter.com/*/status after:2023-05-30
# 人工知能 site:ac.jp

openai.api_key = os.getenv('OPENAI_API_KEY')

MODEL_NAME = 'gpt-3.5-turbo-16k'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'


# Googleで検索を行う
def search_google(query):
    # ブラウザのドライバを指定
    driver = webdriver.Chrome()
    # Googleを開く
    driver.get("http://www.google.com")
    
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
    
    results = []
    for result in search_results:
        title = result.find('h3').text
        url = result.find('a').get('href')
        # URL先のページのbodyを取得
        driver.get(url)
        time.sleep(2)
        page_soup = BeautifulSoup(driver.page_source, 'html.parser')
        try:
            article_title = page_soup.find('h1', attrs={'class': 'article__title'}).text
            article = page_soup.find('div', attrs={'class': 'article-content'}).text
        except:
            article_title = ''
            article = ''
        if article_title != '' and '/sponsor/' not in url:
            results.append({'title': article_title, 'url': url, 'article': article})
    driver.quit()
    
    return results


if __name__ == '__main__':
    results = search_google(query)
    # jsonファイルに保存
    with open('techcrunch.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)