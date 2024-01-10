import os
import openai
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time


import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("./treehacks2021-64fdf-firebase-adminsdk-vp0jf-b909431bb5.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://treehacks2021-64fdf.firebaseio.com'})

db = firestore.client()
doc_ref = db.collection(u'stocks')


df = pd.read_csv('secwiki_tickers.csv')
dp = pd.read_csv('stocks.csv',names=['pTicker'])

pTickers = dp.pTicker.values  # converts into a list
stocks = {}
stock_details = {
    # details: {
    #     name: "",
    #     articles: [],
    #     environmental_score: 0,
    #     environmental_articles_total: "",
    #     environmental_summary: "",
    # }
}

for i in range(len(pTickers)):
    test = df[df.Ticker==pTickers[i]]
    if not (test.empty):
        stocks[pTickers[i]] = list(test.Name.values)[0]
        stock_details[pTickers[i]] = {}
        stock_details[pTickers[i]]["details"] = {}
        stock_details[pTickers[i]]["details"]["name"] = stocks[pTickers[i]]

openai.api_key = os.environ["OPEN_AI_KEY"]
browser = webdriver.Chrome('./chromedriver')

print(stocks)

for ticker, name in stocks.items():
    # search duckduckgo for results
    results_url = f'https://duckduckgo.com/html?q={name.replace(" ", "+")}+Environmental+Impact'
    browser.get(results_url)
    # get resulting urls
    results = browser.find_elements_by_class_name('result__url')
    stock_details[ticker]["details"]["articles"] = {}
    stock_details[ticker]["details"]["environmental_articles_total"] = ""
    for i in range(min(4, len(results))):
        print(results[i].text)
        result_url = results[i].text
        stock_details[ticker]["details"]["articles"]["https://"+result_url] = {}

    for url in stock_details[ticker]["details"]["articles"]:
        print("getting url: ", url)
        # get resultant article
        try:
            browser.get(url)
            article_results = browser.find_elements_by_tag_name("p")
            res = ""
            for body in article_results:
                res += body.text
            print("getting summary")
            response = openai.Completion.create(
              engine="davinci",
              prompt=f'{body.text}\n\ntl;dr:',
              temperature=0.3,
              max_tokens=60,
              top_p=1.0,
              frequency_penalty=0.0,
              presence_penalty=0.0
            )
            print("got summary")
            stock_details[ticker]["details"]["environmental_articles_total"] += response["choices"][0]["text"]
            stock_details[ticker]["details"]["articles"]["https://"+result_url]["title"] = browser.title
        except:
            print('oof')
    response = openai.Completion.create(
      engine="davinci",
      prompt=f'{stock_details[ticker]["details"]["environmental_articles_total"]}\n\ntl;dr:',
      temperature=0.3,
      max_tokens=60,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    stock_details[ticker]["details"]["environmental_summary"] = response["choices"][0]["text"]

    response = openai.Completion.create(
        engine="davinci",
        prompt=f'Description: "{stock_details[ticker]["details"]["environmental_summary"]}"\nSentiment (positive, neutral, negative):',
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response)
    stock_details[ticker]["details"]["environmental_score"] = response["choices"][0]["text"]
    print(f'score for {ticker} is {stock_details[ticker]["details"]["environmental_score"]}')
    print(stock_details)
    temp_ref = doc_ref.document(u''+ticker)
    temp_ref.update(stock_details[ticker])

