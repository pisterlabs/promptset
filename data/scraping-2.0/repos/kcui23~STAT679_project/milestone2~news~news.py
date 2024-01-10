import os
import time
import random
import openai
import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime


def crawler(n_pages=1, ticker="nvda"):
    time_stamp = int(time.time())
    url = f'''
    https://seekingalpha.com/api/v3/symbols/{ticker}/news?filter[since]=0&filter[until]={time_stamp}&id=nvda&include=author%2CprimaryTickers%2CsecondaryTickers%2Csentiments&isMounting=false&page[size]=20&page[number]=
    '''
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Cookie": "cookiesu=591694835194054; device_id=02e9e5105707187692a3ebf043d62941; remember=1; xq_is_login=1; u=8176314854; s=ab12mnrdfx; bid=f24325f9c5bb92500d7f9d541ef6ef8f_lmra6p3v; __utmz=1.1695188801.2.2.utmcsr=github.com|utmccn=(referral)|utmcmd=referral|utmcct=/SJTUMisaka/xueqiu; __utma=1.486665064.1695186225.1695587344.1695604057.4; xq_a_token=76b609375630ee3af674d6ff1312edcc54cda518; xqat=76b609375630ee3af674d6ff1312edcc54cda518; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjgxNzYzMTQ4NTQsImlzcyI6InVjIiwiZXhwIjoxNjk5MTk3MDMxLCJjdG0iOjE2OTY2MDUwMzExMzcsImNpZCI6ImQ5ZDBuNEFadXAifQ.f7xKDW5MpDFMH2Opwn90zwIVVTsSZDcM8BT12a_ID-SfjvDJabSF-i7iejn5UH2TGmfdHT3uJjG8tEwphtUZGhqT4wB1cQI6jOtAToMRnTPjEIlM4_FYrFCL9KyxltsL2HE75AzoZiNYrx9L4JYWaTHwVb8EyOlxZJCb7azWIajJvEgPbKOJODA25J9iu5qmankMpG0RcGHeVvajJbZyt-yU1rTJI8LEeo_RsxgBIxJg9K5HiiMkWs3VNkyXhqqZ5mHxRMaT7Fl5XAT1kRorW799DJBpwFZhY0fNNtNB7B0D0EUL5fBENGzKVGrUGu9QTGkVLZNGpFvIB4ACnXJ8Gg; xq_r_token=034012b5249fa1ae316050a7251e6d9a403ea76b; Hm_lvt_1db88642e346389874251b5a1eded6e3=1695402118,1695435276,1695587332,1696605033; snbim_minify=true; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1696631105; acw_tc=2760779616966318847376306e9556dc4f4ed8169fb3c338239efbc1e25e52"
    }
    news = pd.DataFrame(columns=['title', 'time', 'author_id', 'sentiment'])
    max_retries = 3
    for i in range(1, n_pages):
        tmp_url = url + str(i)
        for i in range(max_retries):
            try:
                response = requests.get(tmp_url, headers=headers)
                response_json = response.json()
                news = update_news(response_json, news)
                break
            except requests.RequestException as e:
                time.sleep(3)
    return news


def update_news(response_json, news):
    for item in response_json['data']:
        title = item['attributes']['title']
        time = item['attributes']['publishOn']
        time = datetime.fromisoformat(time).strftime("%Y-%m-%d")
        author_id = item['relationships']['author']['data']['id']
        sentiment = ''
        news = news.append({'title': title, 'time': time, 'author_id': author_id,
                           'sentiment': sentiment}, ignore_index=True)
    return news


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def _single_get_sentiment(user_prompt):
    system_prompt = """ 
New Instructions: Assume the role of a financial analyst with expertise in evaluating the impact of news on stock prices. Your task is to analyze news headlines concerning specific companies and determine their potential effect on the companies' stock prices.

Procedure:

Initial Response: Provide a one-word sentiment analysis of the news impact on the stock price of the mentioned company.
Bullish: if the news is likely to positively influence the stock price.
Bearish: if the news is likely to negatively impact the stock price.
Neutral: if the news seems to have no clear impact or if the outcome is uncertain.
Question Format: "Is this headline Bullish, Bearish, or Neutral for the stock price of [Company Name]? Headline: [headline]"
    """
    times = 0
    while True:
        times += 1
        response = chat_completions_with_backoff(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        sentiment = response['choices'][0]['message']['content']
        if sentiment in ['Bullish', 'Bearish', 'Neutral']:
            break
        else:
            if times > 2:
                sentiment = 'Neutral'
                break
            continue
    return sentiment


def get_sentiment(news, ticker):
    for i in tqdm(range(len(news))):
        title = news.loc[i, 'title']
        user_prompt = f"Is this headline Bullish, Bearish, or Neutral for the stock price of {ticker}? Headline: {title}"
        sentiment = _single_get_sentiment(user_prompt)
        news.loc[i, 'sentiment'] = sentiment
    return news


def data_aggregation(news, ticker):
    price = yf.download(ticker.upper(), start=news['time'].min(),
                        end=news['time'].max(), interval="1d")
    price.to_csv(f'news_{ticker}_price.csv')
    price = pd.read_csv(f'news_{ticker.lower()}_price.csv')
    price = price[['Date', 'Close']]
    price['Date'] = pd.to_datetime(price['Date'])
    price.columns = ['time', 'price']

    news['time'] = pd.to_datetime(news['time'])
    news = news.set_index('time')
    news['count'] = 1
    news = news.groupby([pd.Grouper(freq='D'), 'sentiment']).sum()
    news = news.reset_index()
    news = news.pivot(index='time', columns='sentiment', values='count')
    news = news.fillna(0)

    data = pd.merge(price, news, on='time', how='left')
    data.fillna(0, inplace=True)
    data.to_csv(f'news_{ticker}_time_close_sent.csv', index=False)


if __name__ == '__main__':
    ticker = "nvda"
    n_pages = 20
    file_name = f'news_{ticker}.csv'
    news = crawler(n_pages=n_pages, ticker=ticker)
    results = get_sentiment(news, ticker)
    results.to_csv(file_name, index=False)
    data_aggregation(results, ticker)
