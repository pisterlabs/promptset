#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from newspaper import Article
from bs4 import BeautifulSoup
import requests
from dateutil import parser
import time
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from newspaper import Article

def initialize_chrome_driver():
  # Chrome 옵션 설정 : USER_AGENT는 알아서 수정
  USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
  chrome_options = Options()
  chrome_options.page_load_strategy = 'normal'  # 'none', 'eager', 'normal'
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')
  chrome_options.add_argument('--disable-gpu')
  chrome_options.add_argument(f'user-agent={USER_AGENT}')
  # Chrome 드라이버 설정
  service = Service()
  wd = webdriver.Chrome(service=service, options=chrome_options)
  return wd

def vdate_util(article_date):
  try:
    article_date = parser.parse(article_date, dayfirst=True).date()
  except ValueError:
    # If parsing fails, handle the relative dates
    article_date = article_date.lower()
    time_keywords = ["h", "hrs", "hr", "m", "hours", "hour", "minutes", "minute", "mins", "min", "seconds", "second", "secs", "sec", "giờ"]
    if any(keyword in article_date for keyword in time_keywords):
      article_date = vtoday
    elif "d" in article_date or "days" in article_date or "day" in article_date or "ngày" in article_date:
      # Find the number of days and subtract from today
      number_of_days = int(''.join(filter(str.isdigit, article_date)))
      article_date = vtoday - timedelta(days=number_of_days)
    else:
      return None
  return article_date

# 에러 메시지 작성 함수
def Error_Message(message, add_error):
    if message is not str() : message += '/'
    message += add_error
    return message

vtoday = datetime.now().date()
vyesterday = (datetime.now() - timedelta(days=1)).date()
vtoday_list = [vtoday, vyesterday]

today = datetime.now().strftime('%d/%m/%Y')
yesterday = (datetime.now() - timedelta(days=1)).strftime('%d/%m/%Y')
today_list = [today, yesterday]

articles = []
error_list = []

vurl_1 = 'http://quochoi.vn/'
vurl_2 = 'http://www.sbv.gov.vn/'
vurl_3 = 'http://moj.gov.vn/'
vurl_4 = 'http://mic.gov.vn/'
vurl_5 = 'http://www.most.gov.vn/'
vurl_6 = 'https://xaydung.gov.vn/vn/chuyen-muc/1259/quy-hoach-kien-truc.aspx'
vurl_7 = "https://moet.gov.vn/tintuc/Pages/tin-tong-hop.aspx"
vurl_8 = 'http://www.monre.gov.vn/'
vurl_8_1 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=%C4%90%E1%BA%A5t%20%C4%91ai"
vurl_8_2 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=T%C3%A0i%20nguy%C3%AAn%20n%C6%B0%E1%BB%9Bc"
vurl_8_3 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=%C4%90%E1%BB%8Ba%20ch%E1%BA%A5t%20v%C3%A0%20kho%C3%A1ng%20s%E1%BA%A3n"
vurl_8_4 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=M%C3%B4i%20tr%C6%B0%E1%BB%9Dng"
vurl_8_5 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Kh%C3%AD%20t%C6%B0%E1%BB%A3ng%20th%E1%BB%A7y%20v%C4%83n"
vurl_8_6 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=%C4%90o%20%C4%91%E1%BA%A1c%20v%C3%A0%20b%E1%BA%A3n%20%C4%91%E1%BB%93"
vurl_8_7 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Bi%E1%BB%83n%20v%C3%A0%20h%E1%BA%A3i%20%C4%91%E1%BA%A3o"
vurl_8_8 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Bi%E1%BA%BFn%20%C4%91%E1%BB%95i%20kh%C3%AD%20h%E1%BA%ADu"
vurl_8_9 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Vi%E1%BB%85n%20th%C3%A1m"
vurl_9 = 'http://www.molisa.gov.vn/'
vurl_10 = 'http://www.mard.gov.vn/'
vurl_11 = 'http://www.chinhphu.vn/'
vurl_11_1 = "https://vpcp.chinhphu.vn/tin-noi-bat.htm"
vurl_11_2 = "https://baochinhphu.vn/chi-dao-quyet-dinh-cua-chinh-phu-thu-tuong-chinh-phu.htm"
vurl_12 = 'https://moh.gov.vn/web/guest/tin-noi-bat'
vurl_13 = 'https://www.mpi.gov.vn/portal/pages/Tin-kinh-te--xa-hoi.aspx'
vurl_14 = 'http://fia.mpi.gov.vn/'
vurl_15 = 'https://cic.org.vn/'
vurl_16 = 'https://vietnamtourism.gov.vn/cat/55'
vurl_17 = 'http://www.mod.gov.vn/'
vurl_18 = 'http://www.mofa.gov.vn/'
vurl_19 = 'https://moha.gov.vn/'
vurl_20 = 'https://www.mof.gov.vn/webcenter/portal/tttc/pages_r/o/ttsk'
vurl_21 = 'https://moit.gov.vn/tin-tuc'
vurl_22 = 'http://congbao.chinhphu.vn/'
vurl_23 = 'http://m.mattran.org.vn'
vurl_24 = 'https://vcci.com.vn/tin-vcci'
vurl_25 = 'https://vinasme.vn/'
vurl_26 = 'http://vr.com.vn/'
vurl_27 = 'https://hanoi.gov.vn/'
vurl_28 = 'https://hochiminhcity.gov.vn/'
vurl_29 = 'https://www.danang.gov.vn/'
vurl_30 = 'http://cadn.com.vn/'
vurl_31 = 'https://www.binhduong.gov.vn/'
vurl_32 = 'https://www.quangngai.gov.vn/web/portal-qni/tin-noi-bat'
vurl_33 = 'http://bacninh.gov.vn/'
vurl_34 = 'https://www.dongnai.gov.vn/Pages/tintucsukien.aspx?gId=2'
vurl_35 = 'https://www.customs.gov.vn/'
vurl_36 = 'https://vccinews.vn/'
vurl_37 = 'https://evn.com.vn/'
vurl_38 = 'https://vietteltelecom.vn/'
vurl_39 = 'https://www.pvn.vn/sites/en/Pages/default.aspx'
vurl_40 = 'https://www.eemc.com.vn/'
vurl_41 = 'http://www.gso.gov.vn'
vurl_42 = 'http://www.khucongnghiep.com.vn'
vurl_43 = 'http://www.thudo.gov.vn'
vurl_44 = 'http://www.hapi.gov.vn'
vurl_45 = 'http://www.hochiminhcity.gov.vn'
vurl_46 = 'http://www.dpi.hochiminhcity.gov.vn'
vurl_47 = 'http://www.itpc.gov.vn'
vurl_48 = 'http://www.ipc.danang.gov.vn'
vurl_49 = 'http://haiphong.gov.vn'
vurl_50 = 'http://cantho.gov.vn'

vurl_51 = 'https://thesaigontimes.vn/'
vurl_52 = 'https://hanoimoi.vn/'
vurl_53 = 'https://baodanang.vn/'
vurl_54 = 'https://vietnamnet.vn/'
vurl_55 = 'https://vneconomy.vn/'
vurl_56 = 'https://www.vietnamplus.vn/'
vurl_57 = 'https://nhandan.vn/'

bing_search_urls1 = [vurl_1, vurl_2, vurl_3, vurl_4, vurl_5, vurl_9, vurl_10, vurl_14, vurl_17, vurl_19, vurl_22, vurl_25, vurl_26, vurl_27, vurl_28, vurl_29, vurl_30, vurl_36, vurl_37, vurl_38, vurl_39, vurl_40, vurl_41, vurl_42, vurl_43, vurl_44, vurl_45, vurl_46, vurl_47, vurl_48, vurl_49, vurl_50]

bing_search_urls2 = [vurl_51, vurl_52, vurl_53, vurl_54, vurl_55, vurl_56, vurl_57]

websites = bing_search_urls1
bing_urls = []
for website in websites:
    search_query = f'site:{website}'
    bing_url = f'https://www.bing.com/search?q={search_query}&setmkt=VN-VN&setlang=vi&setloc=VN'
    bing_urls.append(bing_url)
    
# 오늘자 URL
new_bing_urls = []
for bing_url in bing_urls:
    # ? 다음의 파라미터 부분 추출
    params_start = bing_url.find('?') + 1
    params = bing_url[params_start:]

    # 각 파라미터 추출
    param_list = params.split('&')

    # filters 파라미터 추가
    param_list.append('filters=ex1%3a%22ez1%22')

    # FORM 파라미터 추가
    param_list.append('FORM=000017')

    # 변경된 파라미터를 &로 연결하여 새로운 bing_url 생성
    new_bing_url = "https://www.bing.com/search?" + '&'.join(param_list)
    new_bing_urls.append(new_bing_url)
    
# Initialize an empty list to store dictionaries
links_all = []  # 리스트 추가

for index, new_bing_url in enumerate(new_bing_urls):
    all_links = []
    # 페이지를 넘어가며 결과의 모든 링크 수집
    for page in range(50):  # 최대 50 페이지까지만 수집 (원하는 페이지 수로 수정 가능)
        wd = initialize_chrome_driver()
        wd.get(new_bing_url + f'&first={page * 10}')  # 페이지 번호에 따라 파라미터 추가
        html = wd.page_source
        soup = BeautifulSoup(html, 'html.parser')
        if page == 0:
            if soup.find('span', class_='sb_count'):
                span_text = soup.find('span', class_='sb_count').text
                # 숫자에서 쉼표 제거
                span_text_without_comma = re.sub(r',', '', span_text)
                # 정규 표현식을 사용하여 숫자 추출
                match = re.search(r'Về (\d+) kết quả', span_text_without_comma)
                result = int(match.group(1)) if match else 0

            else:
                result = 0

        news_items = soup.find_all('li', class_='b_algo')

        for news_item in news_items:
            title = news_item.select_one('h2 a').text
            link = news_item.select_one('h2 a')['href']
            # 기사 링크를 links 리스트에 추가
            links_all.append(link)
            all_links.append(link)

            news_dict = {
                'Title': title,
                'Link': link,
                'Content(RAW)': None  # 나중에 채워질 내용
            }

            articles.append(news_dict)

        # "다음 페이지" 버튼이 없으면 종료
        if not soup.find('a', class_='sb_pagN'):
            break

for i, link in enumerate(links_all):
    try:
        # PDF 파일인 경우 예외 처리
        if link.endswith('.pdf'):
            raise ValueError("PDF file is not supported.")

        article = Article(link, language='vi')

        # 기사 다운로드 및 파싱
        article.download()
        article.parse()

        # 기사의 제목, 날짜 및 본문 추출
        text = article.text

        if 'vietnamnet' in link:
            # HTML에서 텍스트 추출
            soup = BeautifulSoup(text, 'html.parser')
            paragraphs = soup.find_all('p')

            # 각 <p> 태그의 내용을 리스트에 추가
            text = '\n'.join(p.get_text() for p in paragraphs)

        # 해당 기사의 'Content' 열에 추출한 내용 추가
        articles[i]['Content(RAW)'] = text

    except ValueError as ve:
        # 해당 기사의 'Content' 열에 None 추가
        articles[i]['Content(RAW)'] = None

    except Exception as e:
        # 해당 기사의 'Content' 열에 None 추가
        articles[i]['Content(RAW)'] = None
        
#vurl_6 = 'https://xaydung.gov.vn/vn/chuyen-muc/1259/quy-hoach-kien-truc.aspx'

def extract_date(date_str):
    try:
        # 괄호 안의 날짜 추출
        date_part = date_str.split("(")[1].split(")")[0]
        parsed_date = parser.parse(date_part, dayfirst=True).date()
        formatted_date = parsed_date.strftime('%d/%m/%Y')  # '%d/%m/%Y' 형식으로 날짜 포맷 변경
        return formatted_date
    except Exception:
        return None

wd = initialize_chrome_driver()
wd.get(vurl_6)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')

try:
    # 첫 번째 형식의 목록 페이지 처리
    news_items = soup.find_all('h2', class_='H2_Favorite2')
    # 두 번째 형식의 목록 페이지 처리
    news_items.extend(soup.find_all('div', class_='list_tin_tieude'))

    if not news_items:
        error_message = "No articles found"

    for item in news_items:
        a_tag = item.find('a')
        title = a_tag.get_text(strip=True)
        link = a_tag['href']
        date_str = item.find('span', class_='news_datetime').get_text(strip=True)
        article_date = extract_date(date_str)

        if article_date in today_list:
            wd.get(link)
            time.sleep(3)
            detailed_html = wd.page_source
            detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
            content_div = detailed_soup.find('div', class_='Around_News_Content')
            article_text = ' '.join(p.get_text(strip=True) for p in content_div.find_all('p'))

            articles.append({
                'Title': title,
                'Link': link,
                'Content(RAW)': article_text
            })

except Exception as e:
  error_list.append({
      'Error Link': vurl_6,
      'Error': str(e)
      })

# vurl_7 = "https://moet.gov.vn/tintuc/Pages/tin-tong-hop.aspx"
wd = initialize_chrome_driver()
wd.get(vurl_7)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select("[class=f1]")
  if not news_items: error_list.append({'Error Link': vurl_7, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.ques-infor').text.strip()
    article_date = vdate_util(date_str)
    if not article_date: error_list.append({'Error Link': vurl_7, 'Error': "None Date"})
    if article_date in vtoday_list:
      title = item.select_one('a').text.strip()
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://moet.gov.vn/tintuc/Pages/tin-tong-hop.aspx{item.a['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      else:
        try:
          article = Article(link, language='vi')
          article.download()
          article.parse()
          content = article.text
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
        except Exception as e:
            error_list.append({
             'Error Link': link,
             'Error': str(e)
             })
except Exception as e:
    error_list.append({
     'Error Link': vurl_7,
     'Error': str(e)
     })
    
# vurl_8_1 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=%C4%90%E1%BA%A5t%20%C4%91ai"
wd = initialize_chrome_driver()
wd.get(vurl_8_1)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_1, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_1,
     'Error': str(e)
     })
    
# 수자원
# vurl_8_2 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=T%C3%A0i%20nguy%C3%AAn%20n%C6%B0%E1%BB%9Bc"
wd.get(vurl_8_2)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_2, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_2,
     'Error': str(e)
     })
    
# 광물
# vurl_8_3 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=%C4%90%E1%BB%8Ba%20ch%E1%BA%A5t%20v%C3%A0%20kho%C3%A1ng%20s%E1%BA%A3n"
wd.get(vurl_8_3)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_3, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_3,
     'Error': str(e)
     })
    
# 환경
# vurl_8_4 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=M%C3%B4i%20tr%C6%B0%E1%BB%9Dng"
wd.get(vurl_8_4)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_4, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_4,
     'Error': str(e)
     })
    
# 기상
# vurl_8_5 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Kh%C3%AD%20t%C6%B0%E1%BB%A3ng%20th%E1%BB%A7y%20v%C4%83n"
wd.get(vurl_8_5)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()

try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_5, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_5,
     'Error': str(e)
     })
    
# 측량 지도
# vurl_8_6 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=%C4%90o%20%C4%91%E1%BA%A1c%20v%C3%A0%20b%E1%BA%A3n%20%C4%91%E1%BB%93"
wd.get(vurl_8_6)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()

try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_6, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_6,
     'Error': str(e)
     })
    
# 해양
# vurl_8_7 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Bi%E1%BB%83n%20v%C3%A0%20h%E1%BA%A3i%20%C4%91%E1%BA%A3o"
wd.get(vurl_8_7)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_7, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_7,
     'Error': str(e)
     })
    
# 기후변화
# vurl_8_8 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Bi%E1%BA%BFn%20%C4%91%E1%BB%95i%20kh%C3%AD%20h%E1%BA%ADu"
wd.get(vurl_8_8)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_8, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
        articles.append({
                        'Title': title,
                        'Link': link,
                        'Content(RAW)': content
                        })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_8,
     'Error': str(e)
     })
    
# 정찰? 탐구?
# vurl_8_9 = "https://monre.gov.vn/Pages/ChuyenMuc.aspx?cm=Vi%E1%BB%85n%20th%C3%A1m"
wd.get(vurl_8_9)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.row.item')
  if not news_items: error_list.append({'Error Link': vurl_8_9, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.item-infor span:nth-of-type(2)').text.strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      a_tag = item.select_one('.item-title a')
      title = a_tag.text
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://monre.gov.vn{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      wd.get(link)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      content_div = soup.select_one('.news-content > .ms-rtestate-field')
      content = []
      if content_div:
          for p_tag in content_div.find_all('p'):
              if p_tag.text:
                  content.append(p_tag.text)
      if content == None: error_message = Error_Message(error_message, "None Content")
      else:
          articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
except Exception as e:
    error_list.append({
     'Error Link': vurl_8_9,
     'Error': str(e)
     })
    
#vurl_11 = "http://www.chinhphu.vn"

try:
    wd = initialize_chrome_driver()
    wd.get(vurl_11)
    time.sleep(3)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    error_message = str()

    news_items = soup.find_all('div', class_='main-article-item')
    for news_item in news_items:
        a_element = news_item.find('a')
        title = a_element.text.strip()
        link = a_element['href']

        # 해당 link로 들어가서 날짜를 가져오기
        wd.get(link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        div_element = article_soup.find('div', {'data-role': 'publishdate'})
        date_text = div_element.contents[0].strip()
        if date_text in today_list:
            paragraphs = article_soup.find_all('p')
            content_list = [p.text for p in paragraphs if p]
            content = '\n'.join(content_list)
            # 결과를 articles 리스트에 추가
            articles.append({
                'Title': title,
                'Link': link,
                'Content(RAW)': content
            })

except Exception as e:
    error_list.append({
     'Error Link': vurl_11,
     'Error': str(e)
     })
    
# 특종?
# vurl_11_1 = "https://vpcp.chinhphu.vn/tin-noi-bat.htm"
wd.get(vurl_11_1)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  hot_item = soup.select_one('.box-news-hot-content')
  if not hot_item: error_list.append({'Error Link': vurl_11_1, 'Error': "None MainNews"})
  hot_date_text = hot_item.select_one('.box-time').get_text()
  hot_date_str = hot_date_text.split('|')[1].strip()
  hot_article_date = vdate_util(hot_date_str)
  if hot_article_date in vtoday_list:
    hot_title = hot_item.select_one('.box-focus-link-title').text.strip()
    if not hot_title: error_message = Error_Message(error_message, "None Hot Title")
    hot_link = f"https://vpcp.chinhphu.vn{hot_item.select_one('.box-focus-link-title')['href']}"
    if not hot_link: error_message = Error_Message(error_message, "None Hot Link")
    else:
        try:
            article = Article(hot_link, language='vi')
            article.download()
            article.parse()
            hot_content = article.text
            articles.append({'Title': hot_title, 'Link': hot_link, 'Content(RAW)': hot_content})
        except Exception as e:
            error_list.append({
                'Error Link': hot_link,
                'Error': str(e)
            })
  news_items = soup.select('.box-stream-item')
  if not news_items: error_list.append({'Error Link': vurl_11_1, 'Error': "None News"})
  for item in news_items:
    date_text = item.select_one('.box-time').get_text()
    date_str = date_text.split('|')[1].strip()
    article_date = vdate_util(date_str)
    if article_date in vtoday_list:
      title = item.select_one('.box-stream-link-title').text.strip()
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://vpcp.chinhphu.vn{item.select_one('.box-stream-link-title')['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      else:
        try:
          article = Article(link, language = 'vi')
          article.download()
          article.parse()
          content = article.text
          articles.append({'Title': title, 'Link': link, 'Content(RAW)': content})
        except Exception as e:
          error_list.append({
           'Error Link': link,
           'Error': str(e)
           })
except Exception as e:
    error_list.append({
     'Error Link': vurl_11_1,
     'Error': str(e)
     })
    
# 국무총리 뉴스
# vurl_11_2 = "https://baochinhphu.vn/chi-dao-quyet-dinh-cua-chinh-phu-thu-tuong-chinh-phu.htm"
wd = initialize_chrome_driver()
wd.get(vurl_11_2)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()

try:
  box_items = soup.select('.box-category-item-first')
  if not box_items: error_list.append({'Error Link': vurl_11_2, 'Error': "None MainNews"})
  for box_item in box_items:
    box_date_str = box_item.select_one('.box-category-time').text.strip()
    box_article_date = vdate_util(box_date_str)
    if not box_article_date: error_message = Error_Message(error_message, "None Box Date")
    if box_article_date in vtoday_list:
      a_tag = box_item.select_one('.box-category-link-title')
      href = a_tag['href']
      if href.startswith('https://'):
        continue
      box_link = f"https://baochinhphu.vn{href}"
      if not box_link: error_message = Error_Message(error_message, "None Box Link")
      box_title = a_tag.text.strip()
      if not box_title: error_message = Error_Message(error_message, 'None Box Title')
      else:
        try:
            article = Article(box_link, language = 'vi')
            article.download()
            article.parse()
            box_content = article.text
            articles.append({'Title': box_title, 'Link': box_link, 'Content(RAW)': box_content})
        except Exception as e:
            error_list.append({
                'Error Link': box_link,
                'Error': str(e)
            })
  news_items = soup.select('.box-stream-item')
  if not news_items: error_list.append({'Error Link': vurl_11_2, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.box-stream-time').text.strip()
    article_date = vdate_util(date_str)
    if not article_date: error_message = Error_Message(error_message, "None Date")
    if article_date in vtoday_list:
      a_tag = item.select_one('.box-stream-link-title')
      href = a_tag['href']
      if href.startswith('https://'):
        continue
      link = f"https://baochinhphu.vn{href}"
      if not link: error_message = Error_Message(error_message, "None Link")
      title = a_tag.text.strip()
      if not title: error_message = Error_Message(error_message, "None Title")
      else:
        try:
          article = Article(link, language = 'vi')
          article.download()
          article.parse()
          content = article.text
          articles.append({'Title': title, 'Link': link, 'Content(RAW)': content})
        except Exception as e:
            error_list.append({
                 'Error Link': link,
                 'Error': str(e)
         })
except Exception as e:
    error_list.append({
     'Error Link': vurl_11_2,
     'Error': str(e)
     })
    
#vurl_12 = 'https://moh.gov.vn/web/guest/tin-noi-bat'
def extract_date(date_str):
    try:
        # "ngày" 다음 부분을 추출하여 날짜로 변환
        date_part = date_str.split("ngày")[1].strip()
        parsed_date = parser.parse(date_part, dayfirst=True).date()
        formatted_date = parsed_date.strftime('%d/%m/%Y')
        return formatted_date
    except Exception:
        return None

try:
    wd = initialize_chrome_driver()
    wd.get(vurl_12)  # 실제 URL로 변경
    time.sleep(5)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    news_items = soup.find_all('div', class_='col-md-8 col-xs-7 content')
    if not news_items:
        error_message = "No articles found"

    for article_div in news_items:
        # 제목과 링크 추출
        title_link_element = article_div.find('h3', class_='asset-title').find('a')
        if title_link_element:
            title = title_link_element.get_text(strip=True)
            link = title_link_element['href']
        else:
            title = "No title found"
            link = "No link found"

        # 날짜 추출
        date_element = article_div.find('span', class_='time')
        date_str = date_element.get_text(strip=True) if date_element else "No date found"
        article_date = extract_date(date_str)

        if article_date in today_list:
            # 셀레니움을 사용하여 상세 페이지의 내용을 가져옴
            wd.get(link)
            time.sleep(3)  # 상세 페이지 로딩 대기
            detailed_html = wd.page_source
            detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
            article_content = detailed_soup.find('div', class_='journal-content-article')
            article_text = article_content.get_text().strip() if article_content else "No text found"

            articles.append({
                'Title': title,
                'Link': link,
                'Content(RAW)': article_text
            })

except Exception as e:
  error_list.append({
      'Error Link': vurl_12,
      'Error': str(e)
      })

#vurl_13 = 'https://www.mpi.gov.vn/portal/pages/Tin-kinh-te--xa-hoi.aspx'
wd = initialize_chrome_driver()
wd.get(vurl_13)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
base_url = "https://www.mpi.gov.vn"

try:
    news_items = soup.find_all('li', class_='TD-channel-main-td channel-news-folder')
    if not news_items:
        error_message = "No articles found"

    for article_li in news_items:
        # 제목과 링크 추출
        title_link_element = article_li.find('a')
        if title_link_element:
            title = title_link_element.get_text(strip=True)
            link = title_link_element['href']
            if not link.startswith('http://') and not link.startswith('https://'):
                link = base_url.rstrip('/') + '/' + link.lstrip('/')
        else:
            title = "No title found"
            link = "No link found"

        # 날짜 추출
        date_element = article_li.find('label', class_='pub-time')
        date = date_element.get_text(strip=True).split('|')[0].strip() if date_element else "No date found"

        # newspaper 라이브러리를 사용하여 기사 내용 가져오기
        if date in today_list:
            try:
                news_article = Article(link, language='vi')
                news_article.download()
                news_article.parse()
                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': news_article.text
                })
            except Exception as e:
                error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })
except Exception as e:
  error_list.append({
      'Error Link': vurl_13,
      'Error': str(e)
      })

#vurl_16 = 'https://vietnamtourism.gov.vn/cat/55'

try:
  wd = initialize_chrome_driver()
  wd.get(vurl_16)
  time.sleep(5)
  error_message = str()

  def custom_date_parse(date_string):
      # Remove parentheses and trim
      cleaned_date = re.sub(r'[()]', '', date_string).strip()
      try:
          parsed_date = parser.parse(cleaned_date, dayfirst=True).date()
          formatted_date = parsed_date.strftime('%d/%m/%Y')  # '%d/%m/%Y' 형식으로 날짜 포맷 변경
          return formatted_date
      except ValueError:
          return None

  base_url = "https://vietnamtourism.gov.vn/cat/55?page="
  num_pages = 2
  wd = initialize_chrome_driver()

  for page_num in range(1, num_pages + 1):
      url = f"{base_url}{page_num}"
      wd.get(url)
      time.sleep(2)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      news_items = soup.find_all('div', class_='section-default-list-news-item')

      for article_div in news_items:
          # 제목 및 링크 추출
          title_link_element = article_div.find('a', class_='section-default-list-news-post-info-title-link')
          title = title_link_element.get_text(strip=True) if title_link_element else "No title found"
          link = title_link_element['href'] if title_link_element else "No link found"

          # 날짜 추출
          date_element = article_div.find('span', class_='post-publish-date')
          date = date_element.get_text(strip=True) if date_element else "No date found"
          article_date = custom_date_parse(date)

          # 본문 추출
          if article_date in today_list:
              wd.get(link)
              time.sleep(5)
              article_html = wd.page_source
              article_soup = BeautifulSoup(article_html, 'html.parser')
              paragraphs = article_soup.find_all('p', style='text-align:justify')
              article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

              articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': article_text
              })

except Exception as e:
  error_list.append({
      'Error Link': vurl_16,
      'Error': str(e)
      })

#vurl_18 = 'https://www.mofa.gov.vn'
wd = initialize_chrome_driver()
wd.get(vurl_18)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')

try:
    news_items = soup.find_all('td', class_='titlenews')

    for item in news_items:
        a_tag = item.find('a')
        if a_tag:
            title = a_tag.get_text(strip=True)
            link = a_tag['href']
            date_str = item.find('span', class_='discreet').get_text(strip=True)
            date_parts = date_str.split('-')
            article_date = '/'.join(date_parts)

            if article_date in today_list:
                wd.get(link)
                time.sleep(3)
                detailed_html = wd.page_source
                detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
                content_div = detailed_soup.find('div', class_='plain')
                article_text = ' '.join(p.get_text(strip=True) for p in content_div.find_all('p', align='justify'))

                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': article_text
                })

except Exception as e:
  error_list.append({
      'Error Link': vurl_18,
      'Error': str(e)
      })

#vurl_20 = 'https://www.mof.gov.vn/webcenter/portal/tttc/pages_r/o/ttsk'
wd = initialize_chrome_driver()
wd.get(vurl_20)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
base_url = "https://www.mof.gov.vn"  # 실제 웹사이트의 기본 URL로 변경

try:
    news_items = soup.find_all('div', class_='bnc-content')
    if not news_items:
        error_message = "No articles found"

    for item in news_items:
        # 제목과 링크 추출
        title_link_element = item.find('a')
        if title_link_element:
            title = title_link_element.get_text(strip=True)
            link = title_link_element['href']
            if not link.startswith('http://') and not link.startswith('https://'):
                link = base_url.rstrip('/') + '/' + link.lstrip('/')
        else:
            title = "No title found"
            link = "No link found"

        # 날짜 추출
        date_element = item.find('span', class_='time')
        date = date_element.get_text(strip=True) if date_element else "No date found"
        article_date = date.split(' ')[0]

        # newspaper 라이브러리를 사용하여 기사 내용 가져오기
        if article_date in today_list:
            try:
                news_article = Article(link, language='vi')
                news_article.download()
                news_article.parse()
                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': news_article.text
                })
            except Exception as e:
                error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })
except Exception as e:
  error_list.append({
      'Error Link': vurl_20,
      'Error': str(e)
      })

#vurl_21 = 'https://moit.gov.vn/tin-tuc'
wd = initialize_chrome_driver()
wd.get(vurl_21)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
base_url = "https://moit.gov.vn"
try:
    news_items = soup.find_all('div', class_='article-right show-left')
    if not news_items:
        error_message = "No articles found"

    for article_div in news_items:
        # 제목과 링크 추출
        title_link_element = article_div.find('a', class_='article-title common-title')
        if title_link_element:
            title = title_link_element.get_text(strip=True)
            link = title_link_element['href']
            if not link.startswith('http://') and not link.startswith('https://'):
                link = base_url.rstrip('/') + '/' + link.lstrip('/')
        else:
            title = "No title found"
            link = "No link found"

        # 날짜 추출
        date_element = article_div.find('span', class_='article-date')
        date = date_element.get_text(strip=True) if date_element else "No date found"

        # newspaper 라이브러리를 사용하여 기사 내용 가져오기
        if date in today_list:
            try:
                news_article = Article(link, language='vi')
                news_article.download()
                news_article.parse()
                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': news_article.text
                })
            except Exception as e:
                error_list.append({
                    'Error Link': vurl_21,
                    'Error': str(e)
            })
except Exception as e:
  error_list.append({
      'Error Link': vurl_21,
      'Error': str(e)
      })

#vurl_23 = 'http://m.mattran.org.vn'
urls = [vurl_23 + '/tin-tuc/', vurl_23 + '/doi-ngoai-kieu-bao/']
wd = initialize_chrome_driver()

for url in urls:
    wd.get(url)
    time.sleep(5)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')

    try:
        story_items = soup.find_all('div', class_='story')
        for item in story_items:
            link_tag = item.find('a', class_='story__link')
            if link_tag:
                link = vurl_23 + link_tag['href']
                title = link_tag.find('span', class_='text').get_text(strip=True)

                date_str = item.find('time').get_text(strip=True) if item.find('time') else None

                if date_str in today_list:
                    wd.get(link)
                    time.sleep(3)
                    detailed_html = wd.page_source
                    detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
                    content_div = detailed_soup.find('div', class_='article__body', id='abody')
                    article_text = ' '.join(p.get_text(strip=True) for p in content_div.find_all('p', style=True))

                    articles.append({
                        'Title': title,
                        'Link': link,
                        'Content(RAW)': article_text
                    })

    except Exception as e:
       error_list.append({
          'Error Link': vurl_23,
          'Error': str(e)
          })
    
#vurl_24 = 'https://vcci.com.vn/tin-vcci'
wd = initialize_chrome_driver()
wd.get(vurl_24)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()

try:
    news_items = soup.find_all('div', class_='item')
    if not news_items:
        error_list.append({
            'Error Link': vurl_24,
            'Error': "None News"
        })
    else:
        for item in news_items:
          title = item.find('h2').text.strip()
          link = item.find('a')['href'].strip()
          raw_date = item.find('span', class_='time').text.strip()
          # 날짜 형식이 'dd-mm-yyyy'인 경우에 맞게 파싱
          article_date = raw_date.split(' ')[-1]
          if article_date in today_list:
              # 기사 링크로 드라이버 이동
              wd.get(link)
              time.sleep(5)
              article_html = wd.page_source
              news_soup = BeautifulSoup(article_html, 'html.parser')
              # Find the 'div' with class 'textcontent' and extract all 'p' tags
              content_div = news_soup.find('div', class_='textcontent')
              if content_div:  # Check if the 'div' was found
                  paragraphs = content_div.find_all('p')
                  content = ' '.join(paragraph.text for paragraph in paragraphs)
                  if error_message is not str():
                    error_list.append({
                        'Error Link': vurl_24,
                        'Error': error_message
                        })
                  else:
                    articles.append({
                        'Title': title,
                        'Link': link,
                        'Content(RAW)': content
                        })
except Exception as e:
  error_list.append({
      'Error Link': vurl_24,
      'Error': str(e)
      })

#vurl_31 = 'https://www.binhduong.gov.vn/'

try:
    wd = initialize_chrome_driver()
    wd.get(vurl_31)
    time.sleep(3)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    error_message = str()
    # 이미 출력한 title을 저장하는 집합(set)
    printed_titles = set()

    # <div class="div-Tonghop-Title">를 포함하는 모든 상위 <a> 태그 찾기
    target_elements = soup.find_all('div', class_="div-Tonghop-Title")
    # 결과 출력
    for div_element in target_elements:
        # 위로 올라가며 상위 <a> 태그 찾기
        a_element = div_element.find_previous('a')

        # a_element가 존재하면서 phanloai가 있는 경우에만 출력
        if a_element and 'phanloai' in a_element.attrs:
            title = div_element.text

            # 이미 출력한 title인지 확인하고 중복되지 않은 경우에만 출력
            if title not in printed_titles:
                link = a_element['href']
                link = 'https://www.binhduong.gov.vn/' + link

                # 출력한 title을 저장
                printed_titles.add(title)

                # 해당 link로 들어가서 날짜를 가져오기
                wd.get(link)
                time.sleep(3)
                article_html = wd.page_source
                article_soup = BeautifulSoup(article_html, 'html.parser')

                # 날짜를 가져오기
                date_element = article_soup.find('span', id="ctl00_ctl33_g_23ce25f7_72cf_4c4a_b013_922bcdc41ed2_ctl00_Label_DateTime_Now")

                # 날짜가 존재하고 오늘 날짜와 같은 경우에만 기사 본문을 가져와서 articles에 추가
                if date_element and date_element.text in today_list:
                    paragraphs = article_soup.find_all('p')
                    content_list = [p.text for p in paragraphs if p]
                    content = '\n'.join(content_list)
                    # 결과를 articles 리스트에 추가
                    articles.append({
                        'Title': title,
                        'Link': link,
                        'Content(RAW)': content
                    })

except Exception as e:
    error_list.append({
     'Error Link': vurl_31,
     'Error': str(e)
     })
    
#vurl_32 = 'https://www.quangngai.gov.vn/web/portal-qni/tin-noi-bat'
wd = initialize_chrome_driver()
wd.get(vurl_32)
time.sleep(5)

html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')

try:
    news_items = soup.find_all('div', class_='box-news-xx')

    for item in news_items:
        a_tag = item.find('a', class_='image')
        h2_tag = item.find('h2', class_='h2-title-xx')
        date_tag = item.find('p', class_='date_post')

        if a_tag and h2_tag and date_tag:
            title = h2_tag.get_text(strip=True)
            link = a_tag['href']
            date_str = date_tag.get_text(strip=True)
            article_date = date_str.split(' ')[1]

            if article_date in today_list:
                wd.get(link)
                time.sleep(3)
                detailed_html = wd.page_source
                detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
                content_div = detailed_soup.find('div', class_='asset-full-content')
                article_text = ' '.join(p.get_text(strip=True) for p in content_div.find_all('p'))

                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': article_text
                })

except Exception as e:
  error_list.append({
      'Error Link': vurl_32,
      'Error': str(e)
      })

#vurl_34 = 'https://www.dongnai.gov.vn/Pages/tintucsukien.aspx?gId=2'

# WebDriver 초기화 및 페이지 로드
wd = initialize_chrome_driver()
wd.get(vurl_34)
time.sleep(5)

html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
base_url = 'https://www.dongnai.gov.vn/Pages/'

try:
    # 두 가지 유형의 기사 검색
    news_items_with_date = soup.find_all('div', class_='lnew')
    news_items_without_date = soup.find_all('div', class_='news1')
    news_items = news_items_with_date + news_items_without_date

    for article_div in news_items:
        title_link_element = article_div.find('a')
        title = title_link_element.get_text(strip=True) if title_link_element else "No title found"
        relative_link = title_link_element['href'] if title_link_element else "No link found"
        link = base_url + relative_link  # 기본 URL과 상대 경로를 결합

        if article_div in news_items_with_date:
            # 날짜가 포함된 기사 형식에서 날짜 추출
            date_element = article_div.find('span')
            date_str = date_element.get_text(strip=True) if date_element else None
            if date_str:
                article_date = date_str.replace('(', '').replace(')', '')
            else:
                article_date = None

            if article_date in today_list:
                wd.get(link)
                time.sleep(3)
                detailed_html = wd.page_source
                detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
                target_div = detailed_soup.find('div', class_='divNoiDungTinBai') or detailed_soup.find('div', class_='ExternalClass7FA29CA78DC54769852B69DAFFB2A158')
                if target_div:
                    article_text = target_div.get_text(separator=' ', strip=True)
                else:
                    article_text = "No text found"

                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': article_text
                })

        elif article_div in news_items_without_date:
            # 상세 페이지에서 날짜를 추출
            wd.get(link)
            time.sleep(3)
            detailed_html = wd.page_source
            detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
            date_li = detailed_soup.find_all('li')
            for a in date_li:
                if a.find('i', class_='fa fa-calendar'):
                    # 'fa fa-calendar' 아이콘을 포함하는 'li' 태그에서 날짜 및 시간 추출
                    date_str = a.get_text(strip=True)
                    break
            article_date = date_str.split(' ')[2]

            if article_date in today_list:
                target_div = detailed_soup.find('div', class_='divNoiDungTinBai') or detailed_soup.find('div', class_='ExternalClass7FA29CA78DC54769852B69DAFFB2A158')
                if target_div:
                    article_text = target_div.get_text(separator=' ', strip=True)
                else:
                    article_text = "No text found"

                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': article_text
                })

except Exception as e:
    error_list.append({
        'Error Link': vurl_34,
        'Error': str(e)
    })
    
#vurl_35 = 'https://www.customs.gov.vn'
base_url = 'https://www.customs.gov.vn'
urls = [base_url + '/index.jsp?pageId=4&cid=32', base_url + '/index.jsp?pageId=4&cid=25']

for url in urls:
    try:
        wd = initialize_chrome_driver()
        wd.get(url)
        time.sleep(5)
        html = wd.page_source
        soup = BeautifulSoup(html, 'html.parser')
        news_items = soup.find_all('h3', class_='content_title')
        if not news_items:
            print("No articles found in", url)

        for article_div in news_items:
            # 제목과 링크 추출
            title_link_element = article_div.find('a')
            title = title_link_element.get_text(strip=True) if title_link_element else "No title found"
            link = 'https://www.customs.gov.vn' + title_link_element['href'] if title_link_element else "No link found"

            # 날짜 추출
            date_element = article_div.find_next_sibling('p', class_='note-layout')
            date_str = date_element.get_text(strip=True) if date_element else "No date found"
            article_date = date_str.split(' ')[0]

            if article_date in today_list:
                # 셀레니움을 사용하여 상세 페이지의 내용을 가져옴
                wd.get(link)
                time.sleep(3)  # 상세 페이지 로딩 대기
                detailed_html = wd.page_source
                detailed_soup = BeautifulSoup(detailed_html, 'html.parser')
                article_content = detailed_soup.find('content', class_='component-content')
                article_text = article_content.get_text().strip() if article_content else "No text found"

                articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': article_text
                })

    except Exception as e:
      error_list.append({
          'Error Link': vurl_35,
          'Error': str(e)
          })
    
    
# Create a DataFrame from the list of dictionaries
news_df_gov = pd.DataFrame(articles)
# Remove rows where 'Content(Raw)' is None or an empty string
news_df_gov = news_df_gov[news_df_gov['Content(RAW)'].notna() & (news_df_gov['Content(RAW)'] != '')]
# Reset the index
news_df_gov.reset_index(drop=True, inplace=True)
news_df_gov = news_df_gov.drop_duplicates(subset=['Title'])

news_df_gov.to_csv('V_Articles_GOV' + datetime.now().strftime("_%y%m%d") + '.csv')

articles_local = []

websites = bing_search_urls2
bing_urls = []
for website in websites:
    search_query = f'site:{website}'
    bing_url = f'https://www.bing.com/search?q={search_query}&setmkt=VN-VN&setlang=vi&setloc=VN'
    bing_urls.append(bing_url)

# 오늘자 URL
new_bing_urls = []
for bing_url in bing_urls:
    # ? 다음의 파라미터 부분 추출
    params_start = bing_url.find('?') + 1
    params = bing_url[params_start:]

    # 각 파라미터 추출
    param_list = params.split('&')

    # filters 파라미터 추가
    param_list.append('filters=ex1%3a%22ez1%22')

    # FORM 파라미터 추가
    param_list.append('FORM=000017')

    # 변경된 파라미터를 &로 연결하여 새로운 bing_url 생성
    new_bing_url = "https://www.bing.com/search?" + '&'.join(param_list)
    new_bing_urls.append(new_bing_url)

# Initialize an empty list to store dictionaries
links_all = []  # 리스트 추가

for index, new_bing_url in enumerate(new_bing_urls):
    all_links = []
    # 페이지를 넘어가며 결과의 모든 링크 수집
    for page in range(50):  # 최대 50 페이지까지만 수집 (원하는 페이지 수로 수정 가능)
        wd = initialize_chrome_driver()
        wd.get(new_bing_url + f'&first={page * 10}')  # 페이지 번호에 따라 파라미터 추가
        html = wd.page_source
        soup = BeautifulSoup(html, 'html.parser')
        if page == 0:
            if soup.find('span', class_='sb_count'):
                span_text = soup.find('span', class_='sb_count').text
                # 숫자에서 쉼표 제거
                span_text_without_comma = re.sub(r',', '', span_text)
                # 정규 표현식을 사용하여 숫자 추출
                match = re.search(r'Về (\d+) kết quả', span_text_without_comma)
                result = int(match.group(1)) if match else 0

            else:
                result = 0

        news_items = soup.find_all('li', class_='b_algo')

        for news_item in news_items:
            title = news_item.select_one('h2 a').text
            link = news_item.select_one('h2 a')['href']
            # 기사 링크를 links 리스트에 추가
            links_all.append(link)
            all_links.append(link)

            news_dict = {
                'Title': title,
                'Link': link,
                'Content(RAW)': None  # 나중에 채워질 내용
            }

            articles_local.append(news_dict)

        # "다음 페이지" 버튼이 없으면 종료
        if not soup.find('a', class_='sb_pagN'):
            break

for i, link in enumerate(links_all):
    try:
        # PDF 파일인 경우 예외 처리
        if link.endswith('.pdf'):
            raise ValueError("PDF file is not supported.")

        article = Article(link, language='vi')

        # 기사 다운로드 및 파싱
        article.download()
        article.parse()

        # 기사의 제목, 날짜 및 본문 추출
        text = article.text

        if 'vietnamnet' in link:
            # HTML에서 텍스트 추출
            soup = BeautifulSoup(text, 'html.parser')
            paragraphs = soup.find_all('p')

            # 각 <p> 태그의 내용을 리스트에 추가
            text = '\n'.join(p.get_text() for p in paragraphs)

        # 해당 기사의 'Content' 열에 추출한 내용 추가
        articles_local[i]['Content(RAW)'] = text

    except ValueError as ve:
        # 해당 기사의 'Content' 열에 None 추가
        articles_local[i]['Content(RAW)'] = None

    except Exception as e:
        # 해당 기사의 'Content' 열에 None 추가
        articles_local[i]['Content(RAW)'] = None
        
        
# Create a DataFrame from the list of dictionaries
news_df_local = pd.DataFrame(articles_local)
# Remove rows where 'Content(Raw)' is None or an empty string
news_df_local = news_df_local[news_df_local['Content(RAW)'].notna() & (news_df_local['Content(RAW)'] != '')]
# Reset the index
news_df_local.reset_index(drop=True, inplace=True)
news_df_local = news_df_local.drop_duplicates(subset=['Title'])

news_df_local.to_csv('V_Articles_LOCAL' + datetime.now().strftime("_%y%m%d") + '.csv')

error_list_df = pd.DataFrame(error_list)
error_list_df.to_csv('V_Error List' + datetime.now().strftime("_%y%m%d") + '.csv')

V_Top30_Name_list = ['LSCV', 'SK Vietnam Group', 'SK INNOVATION HCMC OFFICE', 'SAMSUNG SDS VN CO.', 'NGÂN HÀNG DẦU HYUNDAI', 'Doosan', 'POSCO VIETNAM', 'Năng lượng Doosan', 'LTD. HCMC BRANCH', 'TỰ ĐỘNG HÓA HANWHA TECHWIN', 'LTD. (SECL)', 'Lotte', 'DOOSAN ENERBILITY VIETNAM', 'HYUNDAI CONSTRUCTION', 'DOOSAN HEAVY INDUSTRIES & CONSTRUCTION CO.', 'POSCO', 'OCI Việt Nam', 'HYUNDAI THANH CONG MANUFACTURING VIETNAM', 'Asiana Airlines Hanoi Branch', 'Hyundai Vietnam', 'Samsung', 'Hanwha Corporation Ho Chi Minh', 'Korean Air Ho Chi Minh', 'LS Cable & System Vietnam', 'KOLON GLOBAL CORP HANOI OFC', 'LS CABLE & SYSTEM VIETNAM', 'NĂNG LƯỢNG HANWHA', 'S.G', 'HYUNDAI', 'điện tử LS', 'Samsung', 'SAMSUNG SDS VN CO.', 'Vietnam GS Enterprise One-Member LCC', 'HANWHA TECHWIN AUTOMATION VIETNAM CO.', 'HYUNDAI ENGINEERING CO.', 'SK Innovation HCMC Office', 'HỆ THỐNG ĐIỆN & NĂNG LƯỢNG HYUNDAI', 'HYUNDAI OILBANK CO.', 'HÓA CHẤT LOTTE', 'KOLON', 'LS Vina', 'POSCO-VST', 'HYUNDAI OILBANK VIETNAM REPRESENTATIVE OFFICE', 'SK ĐỔI MỚI', 'CÔNG TY CỔ PHẦN HANWHA', 'LOTTE CHEMICAL CORPORATION (REP. OFFICE)', 'HYUNDAI ELECTRIC & ENERGY SYSTEMS', 'SAMSUNG SDS', 'LG INTERNATIONAL CORP. (HCM OFFICE)', 'LTD. - HCMC BRANCH', 'LX Pantos', 'LTD.', 'POSCO VIETNAM', 'SK Earthon', 'SK Vietnam Group', 'POSCO KỸ THUẬT & XÂY DỰNG', 'Hyundai Oilbank Representative Office in Hanoi', 'ASIANA AIRLINES', 'POSCO Vietnam', 'SAMSUNG ELECTRONICS HCMC CE COMPLEX CO.', 'LS Electric', 'HYUNDAI MOBIS', 'SK Innovation', 'KOLON TOÀN CẦU', 'OCI Vietnam Co. Ltd.', 'Asiana Airlines Hochiminh', 'S.G CHEMICAL', 'LSIS CO.', 'Korea Express', 'LG CHEM CO.', 'HTMV', 'LS CÁP & HỆ THỐNG', 'GS KỸ THUẬT & XÂY DỰNG', 'HÓA CHẤT SG', 'LTD. REPRESENTATIVE OFFICE', 'HYUNDAI ENGINEERING & CONSTRUCTION CO.', 'LSVINA', 'HYUNDAI ENGINEERING & CONSTRUCTION', 'LoTTE Chemical', 'Hóa chất LG', 'LG Chemical', 'HANWHA TECHWIN AUTOMATION\nVIETNAM COMPANY LIMITED', 'LS Cable & System', 'JSC', 'HIVN', 'Korean Air', 'HÀN QUỐC EXPRESS-PACKSIMEX', 'KOLON GLOBAL CORP HOCHIMIN OFC', 'Asiana Airlines Hà Nội', 'HYUNDAI MOBIS', 'GS', 'Samsung Engineering', 'GS Construction', 'KOREAN AIR', 'Hanwha Corporation Manila', "Hyundai','HD Hyundai Electric", 'Vietnam GS Industry One-Member LCC', 'PANTOS LOGISTICS VIỆT NAM', 'HYUNDAI ENGINEERING', 'HYUNDAI MOBIS VIETNAM', 'POSCO-Vietnam', 'SK SEWING MACHINE', '"LG Electronics', 'HANWHA ENERGY CORP', 'LG Chem', 'ĐIỆN TỬ SAMSUNG VINA', 'LX International', 'LTD. HANOI REPRESENTATIVE OFFICE', 'Hanwha International Vietnam Company LTD', 'SAMSUNG SDS', 'Hanwha', 'KOREAN AIR', 'POSCO ENGINEERING & CONSTRUCTION VIET NAM', 'REGIONAL OFFICE', 'LOTTE Chemical Vietnam', 'DOOSAN HANOI OFFICE', 'Hyundai', 'Hanwha Energy Corporation Vietnam Co. Ltd.', 'SK VIETNAM', 'LX Holdings', 'SAMSUNG ENGINEERING VIETNAM CO.', 'KỸ THUẬT HYUNDAI', 'HYUNDAI OILBANK VIETNAM', 'HECV', 'LTD. HANOI OFFICE', 'DOOSAN CÔNG NGHIỆP NẶNG & XÂY DỰNG', 'CJ', 'OCI VIETNAM CO.', 'SK', 'SK Việt Nam', 'Samsung Electronics', 'S.G CHEMICAL Co LTD', 'Hanwha Energy', 'LG', 'Tập Đoàn SK Việt Nam', 'LG Thương Mại', 'SAMSUNG ELECTRONICS VIETNAM', 'LTD. (MAIN OFFICE)', 'LOTTE Chemical Vietnam Co. Ltd.', 'SAMSUNG SDS VN CO.', 'HO CHI MINH CITY', 'UNID Global Corp Hochimin office', 'KOLON GLOBAL', 'SAMSUNG VINA ELECTRONICS CO.', 'KỸ THUẬT SAMSUNG', 'SK E&S', 'DOOSAN HEAVY INDUSTRIES & CONSTRUCTION', 'S.G CHEMICAL CO.', 'SK Innovation', 'KOREA EXPRESS-PACKSIMEX CO.', 'HYUNDAI KỸ THUẬT & XÂY DỰNG', 'CJ Korea Express', 'Doosan Vina', 'HYUNDAI ENGINERRING', 'THIẾT BỊ ĐIỆN TỬ SAMSUNG', 'Doosan Enerbility', 'SK', 'LTD. (REP. OFFICE)', 'Hyundai Thanh Cong Manufacturing Vietnam', 'GS ENGINEERING & CONSTRUCTION CORP', 'KOLON GLOBAL CORPORATION', 'HANWHA CORPORATION', 'PANTOS LOGISTICS VIETNAM CO.', 'LSCV', 'SKVietnamGroup', 'SKINNOVATIONHCMCOFFICE', 'SAMSUNGSDSVNCO.', 'NGÂNHÀNGDẦUHYUNDAI', 'Doosan', 'POSCOVIETNAM', 'NănglượngDoosan', 'LTD.HCMCBRANCH', 'TỰĐỘNGHÓAHANWHATECHWIN', 'LTD.(SECL)', 'Lotte', 'DOOSANENERBILITYVIETNAM', 'HYUNDAICONSTRUCTION', 'DOOSANHEAVYINDUSTRIES&CONSTRUCTIONCO.', 'POSCO', 'OCIViệtNam', 'HYUNDAITHANHCONGMANUFACTURINGVIETNAM', 'AsianaAirlinesHanoiBranch', 'HyundaiVietnam', 'Samsung', 'HanwhaCorporationHoChiMinh', 'KoreanAirHoChiMinh', 'LSCable&SystemVietnam', 'KOLONGLOBALCORPHANOIOFC', 'LSCABLE&SYSTEMVIETNAM', 'NĂNGLƯỢNGHANWHA', 'S.G', 'HYUNDAI', 'điệntửLS', 'Samsung', 'SAMSUNGSDSVNCO.', 'VietnamGSEnterpriseOne-MemberLCC', 'HANWHATECHWINAUTOMATIONVIETNAMCO.', 'HYUNDAIENGINEERINGCO.', 'SKInnovationHCMCOffice', 'HỆTHỐNGĐIỆN&NĂNGLƯỢNGHYUNDAI', 'HYUNDAIOILBANKCO.', 'HÓACHẤTLOTTE', 'KOLON', 'LSVina', 'POSCO-VST', 'HYUNDAIOILBANKVIETNAMREPRESENTATIVEOFFICE', 'SKĐỔIMỚI', 'CÔNGTYCỔPHẦNHANWHA', 'LOTTECHEMICALCORPORATION(REP.OFFICE)', 'HYUNDAIELECTRIC&ENERGYSYSTEMS', 'SAMSUNGSDS', 'LGINTERNATIONALCORP.(HCMOFFICE)', 'LTD.-HCMCBRANCH', 'LXPantos', 'LTD.', 'POSCOVIETNAM', 'SKEarthon', 'SKVietnamGroup', 'POSCOKỸTHUẬT&XÂYDỰNG', 'HyundaiOilbankRepresentativeOfficeinHanoi', 'ASIANAAIRLINES', 'POSCOVietnam', 'SAMSUNGELECTRONICSHCMCCECOMPLEXCO.', 'LSElectric', 'HYUNDAIMOBIS', 'SKInnovation', 'KOLONTOÀNCẦU', 'OCIVietnamCo.Ltd.', 'AsianaAirlinesHochiminh', 'S.GCHEMICAL', 'LSISCO.', 'KoreaExpress', 'LGCHEMCO.', 'HTMV', 'LSCÁP&HỆTHỐNG', 'GSKỸTHUẬT&XÂYDỰNG', 'HÓACHẤTSG', 'LTD.REPRESENTATIVEOFFICE', 'HYUNDAIENGINEERING&CONSTRUCTIONCO.', 'LSVINA', 'HYUNDAIENGINEERING&CONSTRUCTION', 'LoTTEChemical', 'HóachấtLG', 'LGChemical', 'HANWHATECHWINAUTOMATION\nVIETNAMCOMPANYLIMITED', 'LSCable&System', 'JSC', 'HIVN', 'KoreanAir', 'HÀNQUỐCEXPRESS-PACKSIMEX', 'KOLONGLOBALCORPHOCHIMINOFC', 'AsianaAirlinesHàNội', 'HYUNDAIMOBIS', 'GS', 'SamsungEngineering', 'GSConstruction', 'KOREANAIR', 'HanwhaCorporationManila', "Hyundai','HDHyundaiElectric", 'VietnamGSIndustryOne-MemberLCC', 'PANTOSLOGISTICSVIỆTNAM', 'HYUNDAIENGINEERING', 'HYUNDAIMOBISVIETNAM', 'POSCO-Vietnam', 'SKSEWINGMACHINE', '"LGElectronics', 'HANWHAENERGYCORP', 'LGChem', 'ĐIỆNTỬSAMSUNGVINA', 'LXInternational', 'LTD.HANOIREPRESENTATIVEOFFICE', 'HanwhaInternationalVietnamCompanyLTD', 'SAMSUNGSDS', 'Hanwha', 'KOREANAIR', 'POSCOENGINEERING&CONSTRUCTIONVIETNAM', 'REGIONALOFFICE', 'LOTTEChemicalVietnam', 'DOOSANHANOIOFFICE', 'Hyundai', 'HanwhaEnergyCorporationVietnamCo.Ltd.', 'SKVIETNAM', 'LXHoldings', 'SAMSUNGENGINEERINGVIETNAMCO.', 'KỸTHUẬTHYUNDAI', 'HYUNDAIOILBANKVIETNAM', 'HECV', 'LTD.HANOIOFFICE', 'DOOSANCÔNGNGHIỆPNẶNG&XÂYDỰNG', 'CJ', 'OCIVIETNAMCO.', 'SK', 'SKViệtNam', 'SamsungElectronics', 'S.GCHEMICALCoLTD', 'HanwhaEnergy', 'LG', 'TậpĐoànSKViệtNam', 'LGThươngMại', 'SAMSUNGELECTRONICSVIETNAM', 'LTD.(MAINOFFICE)', 'LOTTEChemicalVietnamCo.Ltd.', 'SAMSUNGSDSVNCO.', 'HOCHIMINHCITY', 'UNIDGlobalCorpHochiminoffice', 'KOLONGLOBAL', 'SAMSUNGVINAELECTRONICSCO.', 'KỸTHUẬTSAMSUNG', 'SKE&S', 'DOOSANHEAVYINDUSTRIES&CONSTRUCTION', 'S.GCHEMICALCO.', 'SKInnovation', 'KOREAEXPRESS-PACKSIMEXCO.', 'HYUNDAIKỸTHUẬT&XÂYDỰNG', 'CJKoreaExpress', 'DoosanVina', 'HYUNDAIENGINERRING', 'THIẾTBỊĐIỆNTỬSAMSUNG', 'DoosanEnerbility', 'SK', 'LTD.(REP.OFFICE)', 'HyundaiThanhCongManufacturingVietnam', 'GSENGINEERING&CONSTRUCTIONCORP', 'KOLONGLOBALCORPORATION', 'HANWHACORPORATION', 'PANTOSLOGISTICSVIETNAMCO.', 'Hansol', 'Hana', 'SI', 'Seoul', 'KMW', 'ACE', 'BSE', 'MIWON', 'Daesang', 'Hanil', 'AON', 'KUMHO', 'TAEKWANG', 'ILSHIN', 'DAEHA', 'Daewoo', 'Pangrim', 'IBC', 'Hwaseung', 'Youngone', 'Changshin', 'Asung', 'ORION', 'CS', 'Heesung', 'Hyosung', 'SDV', 'SEVT', 'SEMV', 'SEV', 'HANAMICRON', 'SIFLEX', 'ACEANTENNA', 'DIAMONDPLAZA', 'CSWIND']

Articles_vietnam_gov_Pattern = {'Company':[],'Title':[],'Link':[],'Content(RAW)':[]}
Articles_vietnam_local_Pattern = {'Company':[],'Title':[],'Link':[],'Content(RAW)':[]}

# 결과를 데이터프레임으로 변환
articles_df = pd.DataFrame(articles)
articles_local_df = pd.DataFrame(articles_local)

for index, row in articles_df.iterrows():
    for company in V_Top30_Name_list:
        # None이 아니면서 company가 'Title' 또는 'Content(RAW)'에 포함된 경우
        if row['Title'] and (company in row['Title']) or (row['Content(RAW)'] and company in row['Content(RAW)']):
            Articles_vietnam_gov_Pattern['Company'].append(company)
            Articles_vietnam_gov_Pattern['Title'].append(row['Title'])
            Articles_vietnam_gov_Pattern['Link'].append(row['Link'])
            Articles_vietnam_gov_Pattern['Content(RAW)'].append(row['Content(RAW)'])
            break

for index, row in articles_local_df.iterrows():
    for company in V_Top30_Name_list:
        # None이 아니면서 company가 'Title' 또는 'Content(RAW)'에 포함된 경우
        if row['Title'] and (company in row['Title']) or (row['Content(RAW)'] and company in row['Content(RAW)']):
            Articles_vietnam_local_Pattern['Company'].append(company)
            Articles_vietnam_local_Pattern['Title'].append(row['Title'])
            Articles_vietnam_local_Pattern['Link'].append(row['Link'])
            Articles_vietnam_local_Pattern['Content(RAW)'].append(row['Content(RAW)'])
            break
            
# 결과를 데이터프레임으로 변환
Articles_vietnam_gov_df = pd.DataFrame(Articles_vietnam_gov_Pattern)
Articles_vietnam_local_df = pd.DataFrame(Articles_vietnam_local_Pattern)

def split_and_apply_ner(text, nlp, max_length):
    # 토큰화
    tokens = tokenizer.tokenize(text)
    # 최대 길이에 맞춰 텍스트를 청크로 나눕니다.
    chunks = [
        tokens[i:i + max_length] for i in range(0, len(tokens), max_length)
    ]

    # 각 청크에 대한 NER 결과를 저장할 리스트
    ner_results = []

    # 각 청크에 대해 NER 수행
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunk_ner_results = nlp(chunk_text)
        ner_results.extend(chunk_ner_results)

    return ner_results

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

Articles_Select2_gov = {'Company':[], 'Title':[],'Link':[],'Content(RAW)':[]}

for i in range(len(Articles_vietnam_gov_df)):
  title = Articles_vietnam_gov_df['Title'][i]
  link = Articles_vietnam_gov_df['Link'][i]
  content = Articles_vietnam_gov_df['Content(RAW)'][i]

  combined_text = title + " " + content
  ner_results = split_and_apply_ner(combined_text, nlp, max_length=500)
  entities = [ent for ent in ner_results if ent['entity'] in ["I-ORGANIZATION", "B-ORGANIZATION"]]

  combined_entity = ''
  combined_entity2 = ''
  combined_entity3 = ''
  for ent in entities:
    if ent['entity'] == 'B-ORGANIZATION':
      combined_entity += ' ' + ent['word']
      combined_entity2 += ' ' + ent['word']
    else:
      if ent['word'].startswith('##'):
        combined_entity3 = combined_entity3.rstrip() + ent['word'].replace('##', '')
      else:
        combined_entity3 += ' ' + ent['word']
    
  combined_entity = combined_entity + combined_entity2 + combined_entity3
  combined_entity = combined_entity.replace('##', '')
  filtered_entities = [entity for entity in combined_entity.split() if entity in V_Top30_Name_list]

  if filtered_entities:
    Articles_Select2_gov['Company'].append(', '.join(set(filtered_entities)))
    Articles_Select2_gov['Title'].append(title)
    Articles_Select2_gov['Link'].append(link)
    Articles_Select2_gov['Content(RAW)'].append(content)

Articles_Select2_gov_df = pd.DataFrame(Articles_Select2_gov)

Articles_Select2_local = {'Company':[], 'Title':[],'Link':[],'Content(RAW)':[]}

for i in range(len(Articles_vietnam_local_df)):
  title = Articles_vietnam_local_df['Title'][i]
  link = Articles_vietnam_local_df['Link'][i]
  content = Articles_vietnam_local_df['Content(RAW)'][i]

  combined_text = title + " " + content
  ner_results = split_and_apply_ner(combined_text, nlp, max_length=500)
  entities = [ent for ent in ner_results if ent['entity'] in ["I-ORGANIZATION", "B-ORGANIZATION"]]

  combined_entity = ''
  combined_entity2 = ''
  combined_entity3 = ''
  for ent in entities:
    if ent['entity'] == 'B-ORGANIZATION':
      combined_entity += ' ' + ent['word']
      combined_entity2 += ' ' + ent['word']
    else:
      if ent['word'].startswith('##'):
        combined_entity3 = combined_entity3.rstrip() + ent['word'].replace('##', '')
      else:
        combined_entity3 += ' ' + ent['word']
    
  combined_entity = combined_entity + combined_entity2 + combined_entity3
  combined_entity = combined_entity.replace('##', '')
  filtered_entities = [entity for entity in combined_entity.split() if entity in V_Top30_Name_list]

  if filtered_entities:
    Articles_Select2_local['Company'].append(', '.join(set(filtered_entities)))
    Articles_Select2_local['Title'].append(title)
    Articles_Select2_local['Link'].append(link)
    Articles_Select2_local['Content(RAW)'].append(content)

Articles_Select2_local_df = pd.DataFrame(Articles_Select2_local)

########################################### <Select 3 : OpenAI> ##############################################
import openai

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

Articles_Final_gov = {'Company':[], 'Title':[],'Link':[],'Content(RAW)':[], 'Content(EN)':[]}
Articles_Final_local = {'Company':[], 'Title':[],'Link':[],'Content(RAW)':[], 'Content(EN)':[]}
Articles_Final = {'Company':[], 'Title':[],'Link':[],'Content(RAW)':[], 'Content(EN)':[]}

def translate_with_gpt(text):
    # Use GPT-3 to perform translation
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can choose a different engine
        prompt=f"Translate the following Vietnamese text to English: '{text}'",
        max_tokens=1000  # Adjust as needed
    )
    
    translation = response.choices[0].text.strip()
    return translation

def translate_long_text(text):
    # Split the text into chunks (adjust chunk_size as needed)
    chunk_size = 500
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Translate each chunk and concatenate the results
    translated_chunks = [translate_with_gpt(chunk) for chunk in text_chunks]
    translated_text = " ".join(translated_chunks)

    return translated_text

for i in range(len(Articles_Select2_gov_df)):
    company = Articles_Select2_gov_df['Company'][i]
    title = Articles_Select2_gov_df['Title'][i]
    link = Articles_Select2_gov_df['Link'][i]
    content_raw = Articles_Select2_gov_df['Content(RAW)'][i]

    prompt = f"""Article Title = {title} //
            Article Contents = {content_raw} //
            Company Name to Check = {company} //

            Based on the article titled '{title}' and its content,
            please analyze whether the term '{company}' refers to an actual company.
            If '[{company}' is related to a real company, output 'O'.
            If it is not related to a real company, output 'X'.
            """
    response = get_completion(prompt)

    if response == 'O':
        Articles_Final_gov['Company'].append(company)
        Articles_Final_gov['Title'].append(title)
        Articles_Final_gov['Link'].append(link)
        Articles_Final_gov['Content(RAW)'].append(content_raw)
        
        Articles_Final['Company'].append(company)
        Articles_Final['Title'].append(title)
        Articles_Final['Link'].append(link)
        Articles_Final['Content(RAW)'].append(content_raw)

        # Use GPT for translation
        content_en = translate_long_text(content_raw)
        
        Articles_Final_gov['Content(EN)'].append(content_en)
        Articles_Final['Content(EN)'].append(content_en)
        
for i in range(len(Articles_Select2_local_df)):
    company = Articles_Select2_local_df['Company'][i]
    title = Articles_Select2_local_df['Title'][i]
    link = Articles_Select2_local_df['Link'][i]
    content_raw = Articles_Select2_local_df['Content(RAW)'][i]

    prompt = f"""Article Title = {title} //
            Article Contents = {content_raw} //
            Company Name to Check = {company} //

            Based on the article titled '{title}' and its content,
            please analyze whether the term '{company}' refers to an actual company.
            If '[{company}' is related to a real company, output 'O'.
            If it is not related to a real company, output 'X'.
            """
    response = get_completion(prompt)

    if response == 'O':
        Articles_Final_local['Company'].append(company)
        Articles_Final_local['Title'].append(title)
        Articles_Final_local['Link'].append(link)
        Articles_Final_local['Content(RAW)'].append(content_raw)
        
        Articles_Final['Company'].append(company)
        Articles_Final['Title'].append(title)
        Articles_Final['Link'].append(link)
        Articles_Final['Content(RAW)'].append(content_raw)

        # Use GPT for translation
        content_en = translate_long_text(content_raw)
        
        Articles_Final_local['Content(EN)'].append(content_en)
        Articles_Final['Content(EN)'].append(content_en)

Articles_Final_gov_df = pd.DataFrame(Articles_Final_gov)
Articles_Final_local_df = pd.DataFrame(Articles_Final_local)
Articles_Final_gov_df = Articles_Final_gov_df.drop_duplicates(subset=['Title'])
Articles_Final_local_df = Articles_Final_local_df.drop_duplicates(subset=['Title'])
Articles_Final_gov_df.to_csv('V_Articles_GOV_TRANS' + datetime.now().strftime("_%y%m%d") + '.csv')
Articles_Final_local_df.to_csv('V_Articles_LOCAL_TRANS' + datetime.now().strftime("_%y%m%d") + '.csv')


Articles_Final_df = pd.DataFrame(Articles_Final)
Articles_Final_df = Articles_Final_df.drop_duplicates(subset=['Title'])
Articles_Final_df.to_csv('V_Final Articles' + datetime.now().strftime("_%y%m%d") + '.csv')
