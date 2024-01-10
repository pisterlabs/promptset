from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from dateutil import parser
import re
from newspaper import Article
import requests
import feedparser
import spacy
import openai

# All articles load
articles = pd.read_csv('US_All Articles' + datetime.now().strftime("_%y%m%d") + '.csv')
########################################### <Select 1 : using string> ##############################################
Articles_Select1 = {'Company':[], 'Title':[],'Link':[],'Contents':[]}
for index, row in articles.iterrows():
    for company in Top50_Name_list:
        if (company in row['Title']) or (company in row['Content(RAW)']) :
            Articles_Select1['Company'].append(company)
            Articles_Select1['Title'].append(row['Title'])
            Articles_Select1['Link'].append(row['Link'])
            Articles_Select1['Contents'].append(row['Content(RAW)'])
            break
Articles_Select1 = (pd.DataFrame(Articles_Select1))

########################################### <Select 2 : NER - SpaCy> ##############################################
# spaCy의 NER 모델 로드
nlp = spacy.load("en_core_web_sm")
Articles_Select2 = {'Company':[], 'Title':[],'Link':[],'Contents':[]}
for i in range(len(Articles_Select1)):
    # 분석할 텍스트
    title = Articles_Select1['Title'][i]
    link = Articles_Select1['Link'][i]
    content = Articles_Select1['Contents'][i]
    # NER 처리
    doc1 = nlp(title)
    doc2 = nlp(content)
    # 추출된 고유명사 저장 (ORG 및 PRODUCT)
    entities1 = [(ent.text, ent.label_) for ent in doc1.ents if ent.label_ in ["ORG", "PRODUCT"]]
    entities2 = [(ent.text, ent.label_) for ent in doc2.ents if ent.label_ in ["ORG", "PRODUCT"]]
    # 리스트에 있는 기업명 필터링
    filtered_entities1 = [entity for entity, label in entities1 if (label in ["ORG", "PRODUCT"]) and (entity in Top50_Name_list)]
    filtered_entities2 = [entity for entity, label in entities2 if (label in ["ORG", "PRODUCT"]) and (entity in Top50_Name_list)]
    if filtered_entities1 != []:
        Articles_Select2['Company'].append(', '.join(set(filtered_entities1)))
        Articles_Select2['Title'].append(title)
        Articles_Select2['Link'].append(link)
        Articles_Select2['Contents'].append(content)
    elif filtered_entities2 != []:
        Articles_Select2['Company'].append(', '.join(set(filtered_entities2)))
        Articles_Select2['Title'].append(title)
        Articles_Select2['Link'].append(link)
        Articles_Select2['Contents'].append(content)

Articles_Select2 = pd.DataFrame(Articles_Select2)

########################################### <Select 3 : OpenAI> ##############################################
# 발급받은 API 키 설정(SNU 빅데이터핀테크 API)
OPENAI_API_KEY = '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'
# openai API 키 인증
openai.api_key = OPENAI_API_KEY
# ChatGPT - 3.5 turbo updated
def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

Articles_Final = {'Company':[], 'Title':[],'Link':[],'Content(RAW)':[]}
for i in range(len(Articles_Select2)):
    title = Articles_Select2['Title'][i]
    content = Articles_Select2['Contents'][i]
    company = Articles_Select2['Company'][i]
    link = Articles_Select2['Link'][i]
    prompt = f"""Article Title = {title} //
            Article Contents = {content} //
            Company Name to Check = {company} //

            Based on the article titled '{title}' and its content,
            please analyze whether the term '{company}' refers to an actual company.
            If '[{company}' is related to a real company, output 'O'.
            If it is not related to a real company, output 'X'.
            """
    response = get_completion(prompt)
    print(response)
    if response == 'O':
        Articles_Final['Company'].append(company)
        Articles_Final['Title'].append(title)
        Articles_Final['Link'].append(link)
        Articles_Final['Content(RAW)'].append(content)

########################################### <US Website News Final DataFrame> ##############################################
Articles_Final = pd.DataFrame(Articles_Final)
Articles_Final.to_csv('US_Final Selected Articles' + datetime.now().strftime("_%y%m%d") + '.csv')
