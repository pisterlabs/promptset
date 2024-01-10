#!/usr/bin/env python
# coding: utf-8

# ### chatgpt python 연동
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import os
from time import sleep
import openai
from datetime import datetime
import os
import time


file_dir = '/home/shjj08choi/finance_mlops/data_crawler'

now = datetime.now()
today_date1 = now.strftime('%Y%m%d')
today_date2 = now.strftime('%Y-%m-%d')
today_date_time_csv = now.strftime("%Y%m%d_%H%M")
ticker_nm = '005930'


# 한국 ticker 리스트 

kor_ticker_list = pd.read_csv(f'{file_dir}/kor_ticker_list.csv')
kor_corp_name_list = kor_ticker_list['corp_name']

# 코스피 200 리스트
index_code_df = pd.read_csv(f'{file_dir}/kor_kospi_index_list_df.csv', 
                            dtype = {'index_code':str,
                                    'ticker':str})
kospi_200_index = index_code_df[index_code_df['index_code'] == '1028'] # 코스피 200 인덱스 코드 1028
kospi_200_index = kospi_200_index.reset_index(drop = True)
kospi_200_corp_name_list = kospi_200_index['corp_name']

# 
chatgpt_apikey_df = pd.read_csv(f'{file_dir}/chatgpt_apikey.csv')
chatgpt_apikey = chatgpt_apikey_df['api_key'][0]


# In[70]:


openai.api_key = chatgpt_apikey
model_list = openai.Model.list()



# #### naver crawling
def get_news_summary(key_word):
    
    # 1. 검색 url을 통해 뉴스의 리스트를 읽어 오는 코드
    url = f'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={key_word}'

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"}
    res = requests.get(url, headers= headers)
    soup = bs(res.text, 'html.parser')
    news_list = soup.find_all('div', attrs={'class':'info_group'})
    
    # 각각 뉴스의 리스트에서 링크를 타고 들어간 후 각각의 뉴스 내용을 합치는 코드
    news_url_list = [] # 네이버 뉴스 url 리스트
    new_article_list= [] # 네이버 뉴스 기사 통합 리스트
    for i in range(3): # 최대 5개 정도만 추출
        news_article = news_list[i].find_all("a", attrs={"class":"info"})
        if len(news_article) > 1:
            news_nm_url = news_article[1].get('href')
            news_url_list.append(news_nm_url)
            res = requests.get(news_nm_url, headers=headers)
            soup = bs(res.text, 'lxml')
            news_article_nm = soup.find_all('div', attrs = {'class': 'newsct_article _article_body'})
            news_article_text = news_article_nm[0].get_text()
            new_article_list.append(news_article_text)
            
    # CHAT GPT에 들어갈 최종 뉴스기사
    total_article = '_'.join(new_article_list)
    total_article = total_article.replace("\n","")
    
    
    # 뉴스기사와 함께 요약문 
    input_text_2 = ' \n 전망이 어떨지 요약 좀 해줘'
    
    question = f'{total_article} {input_text_2}'
    
    # Chat gpt
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k", 
      messages=[{"role": "user", 
                 "content": question}]
)
    return completion.choices[0]['message']['content']


now = datetime.now()
today_date_time_csv = now.strftime("%Y%m%d_%H%M")
print(f'뉴스 크롤링 시작 {today_date_time_csv}')  

file_name = 'news_summary'
news_summary_df = pd.DataFrame()

for i in range(len(kospi_200_index)):
    time.sleep(1)
    kospi_200_corp_name = kospi_200_index['corp_name'][i]
    kospi_200_ticker = kospi_200_index['ticker'][i]
    now = datetime.now()
    today_date_time_csv = now.strftime("%Y%m%d-%H:%M:%S")
    print(f'{kospi_200_corp_name} 시작 {today_date_time_csv}')
    try:
        result_val = get_news_summary(kospi_200_corp_name)
        
        df_raw = pd.DataFrame({
          'corp_name':kospi_200_corp_name,
          'ticker':kospi_200_ticker,
          'summary':result_val
        }, index = [0])
        
        news_summary_df = pd.concat([news_summary_df, df_raw])
        # if not os.path.exists(f'{file_dir}/{file_name}_{today_date1}.csv'):
        #     df_raw.to_csv(f'{file_dir}/{file_name}_{today_date1}.csv', index=False, mode='w')
        # else:
        #     df_raw.to_csv(f'{file_dir}/{file_name}_{today_date1}.csv', index=False, mode='a', header=False)
        
        print(f'{kospi_200_corp_name} 완료 {today_date_time_csv}')
    except:
        df_raw = pd.DataFrame({
          'corp_name':kospi_200_corp_name,
          'summary':"None"
        }, index = [0])
        
        news_summary_df = pd.concat([news_summary_df, df_raw])
        
        # if not os.path.exists(f'{file_dir}/{file_name}_{today_date1}.csv'):
        #     df_raw.to_csv(f'{file_dir}/{file_name}_{today_date1}.csv', index=False, mode='w')
        # else:
        #     df_raw.to_csv(f'{file_dir}/{file_name}_{today_date1}.csv', index=False, mode='a', header=False)
            
        print(f'{kospi_200_corp_name} 에러 {today_date_time_csv}')

    
now = datetime.now()
today_date_time_csv = now.strftime("%Y%m%d_%H%M")
print(f'뉴스 크롤링 완료 {today_date_time_csv}')  
news_summary_df = news_summary_df.reset_index(drop = True)

news_summary_df.to_csv(f'{file_dir}/news_summary_{today_date_time_csv}.csv')




