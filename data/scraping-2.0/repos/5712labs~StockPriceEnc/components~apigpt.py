##########################################################################
### 공통함수 ###############################################################
##########################################################################
# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import json
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import feedparser
from newsapi import NewsApiClient


def get_vector_chroma(prompt):
    os.environ['OPENAI_API_KEY'] = st.secrets["api_dw"]
    openai.api_key = os.getenv('OPENAI_API_KEY')
    persist_directory="db"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding, 
        persist_directory=persist_directory)  
    vectordb.as_retriever(search_kwargs={"k": 2})
    docs = vectordb.similarity_search_with_relevance_scores(prompt)
    query_df = pd.DataFrame()
    augmented_query = '' # 벡터DB 유사도
    for doc in docs:
        if doc[1] < 0.75:
            continue
        augmented_query += doc[0].page_content + '\n'
        re_df = pd.DataFrame([[doc[1], doc[0].page_content, doc[0].metadata['source'], 'IDIDID']])
        query_df = pd.concat([query_df, re_df])
    if len(query_df) > 0 :
        query_df.reset_index(drop=True, inplace=True)
        query_df.columns = ['score', 'text', 'source', 'id']
    return augmented_query, query_df

def get_functions_list(): 
    functions_list = [
        {
            "name": "get_current_weather", # 펑션 이름
            # 펑션 기능에 대한 설명
            "description": "Get the current weather in a given location",
            "parameters": { # 함수가 허용하는 매개 변수
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_economic_indicators", # 펑션 이름
            "description": "경제지표 브리핑",
            "parameters": { # 함수가 허용하는 매개 변수
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The stock, e.g. Apple, APPL",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days",
                    }
                },
            },
        },
        {
            "name": "get_news_newsapi", # 펑션 이름
            # 펑션 기능에 대한 설명
            "description": "Get the current news in a given search queries",
            # "description": "최신 뉴스",
            "parameters": { # 함수가 허용하는 매개 변수
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "search queries, e.g. 건설, 대우건설, 경제",
                    },
                    # "rows": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "numOfRows": {
                        "type": "integer",
                        "description": "The number of rows",
                    }
                }
                # "required": ["country"],
            },
        },

        {
            "name": "get_company_info", # 펑션 이름
            # 펑션 기능에 대한 설명
            "description": "특정 업체의 관련 정보 가져오기",
            "parameters": { # 함수가 허용하는 매개 변수
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "The company, e.g. 대우건설",
                    }
                    # "rows": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    # "numOfRows": {
                    #     "type": "integer",
                    #     "description": "The number of rows",
                    # }
                },
                "required": ["company"],
            },
        }



    ]
    return functions_list

def get_economic_indicators(num_days):

    products = [
        {'name': '달러인덱스', 'symbol': 'DX-Y.NYB'},
        {'name': '크루드오일', 'symbol': 'CL=F'},
        {'name': '금', 'symbol': 'GC=F'},
        {'name': 'S&P500', 'symbol': '^GSPC'},
        {'name': '천연가스', 'symbol': 'LNG'},
        {'name': '10년물', 'symbol': '^TNX'},
        {'name': '원자재', 'symbol': 'DBC'}
        ]
    
    change_eco_df = pd.DataFrame() # 변동률
    last_df = pd.DataFrame() # 변동률
    for idx, product in enumerate(products):

        get_product_data = yf.Ticker(product['symbol'])
        start_date = datetime.today() - relativedelta(days=num_days)
        product_df = get_product_data.history(period='1d', start=start_date, end=datetime.today())
            # 일간변동률, 누적합계
        product_df['dpc'] = (product_df.Close/product_df.Close.shift(1)-1)*100
        product_df['cs'] = round(product_df.dpc.cumsum(), 2)

        change2_df = pd.DataFrame(
            {
                'Date2': product_df.index,
                'symbol': product['name'],
                'Close': round(product_df.Close, 2),
                'rate': product_df.cs,
                }
        )
        # change2_df.reset_index(drop=False, inplace=True)
        change2_df.reset_index(drop=True, inplace=True)
        change2_df.columns = ['Date', 'symbol', 'Close', 'rate']
        change_eco_df = pd.concat([change_eco_df, change2_df])

        last2_df = pd.DataFrame(product_df.iloc[len(product_df.index)-1]).T
        last3_df = pd.DataFrame(
            {
                'symbol': product['name'],
                'Date': last2_df.index,
                'Close': last2_df.Close, 
                'rate': last2_df.cs,
                }
        )
        last_df = pd.concat([last_df, last3_df])

    return change_eco_df, last_df

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

def get_company_info(company):
    """Get the information in a given company"""
    company_info = {
        "name": company,
        "summary": "옥천 푸르지오 분양예정 매출액 10000조",
        "grade": "A",
    }
    # return json.dumps(company_info, ensure_ascii=False)
    return json.dumps(company_info, ensure_ascii=False)

def get_news_newsapi(search, numOfRows):
    if search == None:
        newsapi = NewsApiClient(api_key=st.secrets["api_news"])
        # /v2/top-headlines
        # top_headlines = newsapi.get_top_headlines(q='bitcoin',
        #                                         # sources='bbc-news,the-verge',
        #                                         category='business',
        #                                         language='en',
        #                                         country='us')
        # # /v2/everything
        # all_articles = newsapi.get_everything(q='bitcoin',
        #                                     sources='bbc-news,the-verge',
        #                                     domains='bbc.co.uk,techcrunch.com',
        #                                     from_param='2017-12-01',
        #                                     to='2017-12-12',
        #                                     language='en',
        #                                     sort_by='relevancy',
        #                                     page=2)
        # /v2/top-headlines/sources
    # https://newsapi.org/v2/everything?q=apple&from=2023-10-13&to=2023-10-13&sortBy=popularity&apiKey=6dc465ca5ca84475bf9caa3767080730
    # https://newsapi.org/v2/everything?q=Apple&from=2023-10-13&sortBy=popularity&apiKey=6dc465ca5ca84475bf9caa3767080730
    # https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey=6dc465ca5ca84475bf9caa3767080730
    # https://newsapi.org/v2/top-headlines?sources=techcrunch&apiKey=6dc465ca5ca84475bf9caa3767080730
    # https://newsapi.org/v2/top-headlines?sources=business&apiKey=6dc465ca5ca84475bf9caa3767080730
    # https://newsapi.org/v2/top-headlines?country=kr&apiKey=6dc465ca5ca84475bf9caa3767080730
    # https://newsapi.org/v2/top-headlines?country=kr&category=business&apiKey=6dc465ca5ca84475bf9caa3767080730
    # https://newsapi.org/v2/top-headlines/sources?apiKey=6dc465ca5ca84475bf9caa3767080730
        # top_headlines = newsapi.get_top_headlines(country='kr')
        # sources = newsapi.get_sources()
        # print(sources) 
    # 'ae', ar at au be bg br ca ch cn co cu cz de eg fr gb gr hk hu id ie il in it jp kr lt 
    # 'lv', ma mx my ng nl no nz ph pl pt ro rs ru sa se sg si sk th tr tw ua us ve za

    # business entertainment general health science sports technology

        # url = ('https://newsapi.org/v2/top-headlines?'
        #     'country=ng&'
        #     f'apiKey={st.secrets["api_news"]}')
        # response = requests.get(url)
        # return response.json()
        countries = ['kr', 'us', 'jp', 'cn']
        responses = []
        for country in countries:
            url = (f'https://newsapi.org/v2/top-headlines?country={country}&apiKey={st.secrets["api_news"]}')
            response = requests.get(url)
            for resp in response.json()['articles'][:numOfRows]:
                responses.append(resp)            
        # responses += response.json()['articles']
        # print(responses)
        return responses #<class 'list'>

    else :     
        # https://news.google.com/rss/search?q=대우건설&hl=ko&gl=KR&ceid=KR%3Ako
        # https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko
        # rss_url = 'https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko'
        rss_url = f'https://news.google.com/rss/search?q={search}&hl=ko&gl=KR&ceid=KR%3Ako'
        # rss_url = 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR%3Ako'
        feed = feedparser.parse(rss_url)
        # print(feed)
        # 처음 10개 뉴스 항목 출력
        responses = []
        for entry in feed.entries[:numOfRows]:
            responses.append(entry)
        # return feed #<class 'feedparser.util.FeedParserDict'>
        return responses #<class 'list'>

def get_news_google(country, numOfRows):

# news.google.com/rss/search?q=대우건설&hl=ko&gl=KR&ceid=KR%3Ako
# https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko

    # rss_url = 'https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko'
    # rss_url = 'https://news.google.com/rss/search?q=대우건설&hl=ko&gl=KR&ceid=KR%3Ako'
    rss_url = 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR%3Ako'
    feed = feedparser.parse(rss_url)
    # print(feed)
    # 처음 10개 뉴스 항목 출력
    # for entry in feed.entries[:10]:
    #     # print(entry)
    #     print(f"제목: {entry.title}")
    #     print(f"일자: {entry.published}")
    #     print(f"링크: {entry.link}")
    #     print("\n")
    return feed
    # return json.dumps(entry)