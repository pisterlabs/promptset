import openai, json, os
import urllib.request
import json
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from urllib import request
import pymsteams
import random, time


openai.api_key = os.environ["OPENAI_API_KEY"]
teamsapi = os.environ["TEAMS_WEB_HOOK_URL"]


def ask(question):
    prompt = f"### 指示\n論文の内容を要約した上で，重要なポイントを箇条書きで3点書いてください。また、URL(httpから始まる文字列)が入力されていた場合はそのURLを一番最後に出力する事。\n### 箇条書きの制約###\n- 最大3個\n- 日本語\n- 箇条書き1個を50文字以内\n###対象とする論文の内容###\n{question}\n###以下のように出力してください###\n- 箇条書き1\n- 箇条書き2\n- 箇条書き3\n-入力されたURL(URLが入力されていなければ無視して良い)"
    print(prompt)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    answer = response.choices[0].text.strip()
    print(answer)
    time.sleep(10)
    myTeamsMessage = pymsteams.connectorcard(teamsapi)
    myTeamsMessage.title(tagurl)
    myTeamsMessage.text(answer)
    myTeamsMessage.send()
    return answer



kensakulist = ['kubernetes', 'コンテナ', 'Docker','CNCF', 'kubernetes', 'k8s', 'マイクロサービス','クラウド']

for kensaku in kensakulist:
    try:
        url = "https://cir.nii.ac.jp/articles?q=" + str(kensaku) + "&count=200&sortorder=0"
        print(url)




        response = request.urlopen(url)
        soup = BeautifulSoup(response)
        response.close()



        tag_list = soup.select('a[href].availableLink')
        for tag in tag_list:
            tagurl = tag.get('href')
            print(tagurl)


        tag_list = soup.select('a[href].availableLink')
        for tag in tag_list:
            try:
                tagurl = tag.get('href')
                response = request.urlopen(tagurl)
                soup = BeautifulSoup(response)
                response.close()

                tables = soup.findAll('table')[0]
                rows = tables.findAll('td', class_="td_detail_line_repos w80")


                for row in rows:
                    text1 = row.get_text()
                    text1 = re.sub('\s+', ' ', text1)
                    print(len(text1))
                    if 500 < len(text1):
                        teur = str(text1) + str(tagurl)
                        ask(teur)
            except:
                pass

    except:
        pass
