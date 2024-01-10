# 라이브러리 import
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

def initialize_chrome_driver():
  # Chrome 옵션 설정 : USER_AGENT는 알아서 수정
  USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
  # 태준컴
  #USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.2 Safari/605.1.15'
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

# 날짜 통합 함수
def date_util(article_date):
  today = datetime.now().date()
  try:
    # Parse the date using dateutil.parser
    article_date = parser.parse(article_date).date()
  except ValueError:
    # If parsing fails, handle the relative dates
    article_date = article_date.lower()
    time_keywords = ["AM", "PM", "h", "hrs", "hr", "m", "s", "hours","hour", "minutes", "minute", "mins", "min", "seconds", "second", "secs", "sec"]
    if any(keyword in article_date for keyword in time_keywords):
      article_date = today
    elif "days" in article_date or "day" in article_date:
      # Find the number of days and subtract from today
      number_of_days = int(''.join(filter(str.isdigit, article_date)))
      article_date = today - timedelta(days=number_of_days)
    else:
      return None
  return article_date

# 에러 메시지 작성 함수
def Error_Message(message, add_error):
    if message is not str() : message += '/'
    message += add_error
    return message

# scraping할 때, context 긁는 함수
def scrap_context(url):
    wd = initialize_chrome_driver()
    wd.get(url)
    time.sleep(3)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    # 키워드가 포함된 모든 텍스트 블록 찾기
    context = str()
    # 대소문자 구분 없이 키워드와 정확히 일치하는 정규 표현식 패턴 생성
    pattern = re.compile('|'.join(r'(?<!\w)' + re.escape(keyword) + r'(?!\w)' for keyword in Top50_Name_list), re.IGNORECASE)
    #for element in soup.find_all(string=lambda text: keyword in text):
    for element in soup.find_all(string=pattern):
        context += (element.text + '\n')
    # 기업명 앞+뒤 문맥 파악할 수 있는 글이 포함되면, context return
    if context :
      return(context)
    else:
      return()

# 찾는 기업명 리스트(50개)
Top50_Name_list = ['SKLSI', 'Hana West Properties LLC', 'Samsung Electronics', 'Doosan Enerbility America', 
'Hyundai Motor', 'SeAH', 'LG Group', 'GOLFZON', 'GS Global USA', 'Doosan Group', 'ASIANA', 'GS Global', 
'POSCO America Corp', 'Medy-Tox', 'SK life science', 'COUPANG GLOBAL LLC', 'LG Corp', 'SAMSUNG SEMICONDUCTOR', 
'DSME', 'HYUNDAI', 'HL Mando America Corporation', 'Nongshim', 'Hanwha Corporation', 'LG Elec', 'Doosan Corp', 
'MANDO', "Samsung Semiconductor's US", 'Posco', 'SK hynix America', 'LG', 'SeAH America', 'SeAH USA', 
'POSCO International Corp', 'SK Energy', 'KIA MOTORS AMERICA', 'SEAH STEEL', 'LG Energy', 'LG Display', 
'POSCO AMERICA', 'Doosan Heavy Industries', 'Hanwha Corp', 'SM Entertainment USA', 'Kia Corp', 'LS Group', 
'SK E&S', 'LG ELECTRONICS USA', 'Mando America', 'HMGMA', 'POSCO America', 'Golfzon', 'MEDY-TOX', 'LG Energy Solution', 
'SK NETWORKS AMERICA', 'COUPANG GLOBAL', 'Medytox', 'SK Battery', 'Mando America Corp', 'LG Display America', 
'SK Hynix Semiconductor', 'Coupang Global LLC', 'Zenith Electronics LLC', 'Dsme', 'SK E&S America', 'Kia Motors', 
'ASIANA AIRLINES', 'SEAH STEEL USA LLC', 'GSG', 'Mando America Corporation', 'HL Mando America', 'Zenith Electronics', 
'DOOSAN HEAVY INDUSTRIES AMERICA LLC', 'SK Energy Americas', 'SM ENTERTAINMENT', 'POSCO International', 'SeAH Steel', 
'LG ENERGY SOLUTION', 'LS CABLE', 'Doosan Electro-Materials America', 'SM USA', 'LS Cable Systems', 
'Doosan Heavy Industries America Holdings', 'Nongshim America', 'Doosan Heavy Industries America', 'NONGSHIM AMERICA', 
'KOREAN AIR', 'KMMG', 'Doosan', 'LG Energy Solution Tech Center', 'SK E&S Energy', 'HANWHA', 'Samsung Electronics America', 
'HL Mando', 'Mando Corporation', 'Coupang', 'SK Life Science', 'DOOSAN MATERIAL HANDLING SOLUTION DOOSAN INDUSTRIAL VEHICLE AMERICA CORP', 
'Golfzon America Official', 'SK USA', 'Hanwha Q cells', 'Hyundais', 'LG Energy Michigan Tech Center', 'LS Cable & System USA', 
'Kia Georgia Plant', 'HANWHA HOLDINGS', 'DIVAC', 'HANWHA Q CELLS AMERICA', 'DOOSAN ELECTRO-MATERIALS AMERICA', 'MEDYTOX', 
'POSCO Holdings', 'Daewoo Shipbuilding and Marine Engineering', 'LG Electronics', 'Hanwha Holdings USA', 
'POSCO AMERICA CORPORATION', 'HANWHA CORP', 'Kia Motors Manufacturing Georgia', 'SM ENTERTAINMENT USA', 'KIA', 'SMtown USA', 
'SK HYNIX', 'Kias', 'Hanwha Holdings', 'Golfzon America', 'Mando', 'Hana West Properties', 'Samsung Biologics', 
'Hanwha Q CELLS America', 'SK E&S Co', 'Doosan Heavy Industries America LLC', 'POSCO America Corporation', 'Asiana Airlines', 
'Doosan Enerbility America Holdings', 'Kia Corporation', 'SK NETWORKS', 'DSME CO', 'Kia Georgia', 'SK HYNIX AMERICA', 
'Hanwha Group', 'Coupang Global', 'SK', 'Qcells', 'SAMSUNG', 'KIA GEORGIA', 'Hyundai', 'Doosan Material Handling Solutions', 
'SK Networks America', 'Hanwha Q CELLS', 'Samsung US', 'KIA MOTORS CORP', 'Samsung Semiconductor USA', 'LG ELECTRONICS', 
'SK Group Companies', 'SK GROUP', 'Samsung', 'SK Networks', 'Doosan Corporation', 'DSEM', 'SK E&S AMERICA', 'Posco America', 
'LGE', 'SK Hynix America', 'Samsung Electronics North America', 'Hyundai Motors', 'Hanwha USA', 'Kia', 'Samsung Semiconductor', 
'Doosan Industrial Vehicle', 'POSCO', 'LS Cable Systems America', 'Doosan Enerbility', 'LS CABLE SYSTEMS AMERICA', 
'LG ENERGY SOLUTION MICHIGAN TECH CENTER', 'SK ENERGY', 'SeAH US', 'Doosan Electronics', 'Kia America', 
'SAMSUNG ELECTRONICS AMERICA', 'NONGSHIM', 'Hyundai Motor Group', 'SK Hynix', 'GS GLOBAL USA', 'Kia Motors America', 
'LS CABLE & SYSTEM', 'SK hynix', 'KIA MOTORS', 'LG ELECTRONICS(ZENITH ELECTRONICS LLC)', 'SK ENERGY AMERICAS', 'DHI', 'Asiana', 
'GOLFZON AMERICA', 'HANWHA WEST PROPERTIES LLC', 'KOREAN AIRLINES', 'LG DISPLAY', 'LG DISPLAY AMERICA', 'SK LIFE SCIENCE', 
'Korean Air', 'Zenith Electronics Corp', 'Q CELLS', 'Doosan Industrial Vehicle America Corporation', 'Samsung Biologics America', 
'GSG Corp', 'SAMSUNG BIOLOGICS', 'SK Group', 'SM Entertainment', 'Seah Steel USA LLC', 'Doosan Industrial Vehicle America Corp', 
'Hanwha', 'HYUNDAI MOTORS', 'Hanwha OCEAN', 'Nongshim USA', 'SeAH Steel USA', 'LS Cable & System', 'SKLSI', 'HanaWestPropertiesLLC', 
'SamsungElectronics', 'DoosanEnerbilityAmerica', 'HyundaiMotor', 'SeAH', 'LGGroup', 'GOLFZON', 'GSGlobalUSA', 'DoosanGroup', 
'ASIANA', 'GSGlobal', 'POSCOAmericaCorp', 'Medy-Tox', 'SKlifescience', 'COUPANGGLOBALLLC', 'LGCorp', 'SAMSUNGSEMICONDUCTOR', 
'DSME', 'HYUNDAI', 'HLMandoAmericaCorporation', 'Nongshim', 'HanwhaCorporation', 'LGElec', 'DoosanCorp', 'MANDO', 
"SamsungSemiconductor'sUS", 'Posco', 'SKhynixAmerica', 'LG', 'SeAHAmerica', 'SeAHUSA', 'POSCOInternationalCorp', 'SKEnergy', 
'KIAMOTORSAMERICA', 'SEAHSTEEL', 'LGEnergy', 'LGDisplay', 'POSCOAMERICA', 'DoosanHeavyIndustries', 'HanwhaCorp', 
'SMEntertainmentUSA', 'KiaCorp', 'LSGroup', 'SKE&S', 'LGELECTRONICSUSA', 'MandoAmerica', 'HMGMA', 'POSCOAmerica', 'Golfzon', 
'MEDY-TOX', 'LGEnergySolution', 'SKNETWORKSAMERICA', 'COUPANGGLOBAL', 'Medytox', 'SKBattery', 'MandoAmericaCorp', 
'LGDisplayAmerica', 'SKHynixSemiconductor', 'CoupangGlobalLLC', 'ZenithElectronicsLLC', 'Dsme', 'SKE&SAmerica', 'KiaMotors', 
'ASIANAAIRLINES', 'SEAHSTEELUSALLC', 'GSG', 'MandoAmericaCorporation', 'HLMandoAmerica', 'ZenithElectronics', 
'DOOSANHEAVYINDUSTRIESAMERICALLC', 'SKEnergyAmericas', 'SMENTERTAINMENT', 'POSCOInternational', 'SeAHSteel', 'LGENERGYSOLUTION', 
'LSCABLE', 'DoosanElectro-MaterialsAmerica', 'SMUSA', 'LSCableSystems', 'DoosanHeavyIndustriesAmericaHoldings', 'NongshimAmerica',
'DoosanHeavyIndustriesAmerica', 'NONGSHIMAMERICA', 'KOREANAIR', 'KMMG', 'Doosan', 'LGEnergySolutionTechCenter', 'SKE&SEnergy', 
'HANWHA', 'SamsungElectronicsAmerica', 'HLMando', 'MandoCorporation', 'Coupang', 'SKLifeScience', 
'DOOSANMATERIALHANDLINGSOLUTIONDOOSANINDUSTRIALVEHICLEAMERICACORP', 'GolfzonAmericaOfficial', 'SKUSA', 'HanwhaQcells', 'Hyundais', 
'LGEnergyMichiganTechCenter', 'LSCable&SystemUSA', 'KiaGeorgiaPlant', 'HANWHAHOLDINGS', 'DIVAC', 'HANWHAQCELLSAMERICA', 
'DOOSANELECTRO-MATERIALSAMERICA', 'MEDYTOX', 'POSCOHoldings', 'DaewooShipbuildingandMarineEngineering', 'LGElectronics', 
'HanwhaHoldingsUSA', 'POSCOAMERICACORPORATION', 'HANWHACORP', 'KiaMotorsManufacturingGeorgia', 'SMENTERTAINMENTUSA', 'KIA', 
'SMtownUSA', 'SKHYNIX', 'Kias', 'HanwhaHoldings', 'GolfzonAmerica', 'Mando', 'HanaWestProperties', 'SamsungBiologics', 
'HanwhaQCELLSAmerica', 'SKE&SCo', 'DoosanHeavyIndustriesAmericaLLC', 'POSCOAmericaCorporation', 'AsianaAirlines', 
'DoosanEnerbilityAmericaHoldings', 'KiaCorporation', 'SKNETWORKS', 'DSMECO', 'KiaGeorgia', 'SKHYNIXAMERICA', 'HanwhaGroup', 
'CoupangGlobal', 'SK', 'Qcells', 'SAMSUNG', 'KIAGEORGIA', 'Hyundai', 'DoosanMaterialHandlingSolutions', 'SKNetworksAmerica', 
'HanwhaQCELLS', 'SamsungUS', 'KIAMOTORSCORP', 'SamsungSemiconductorUSA', 'LGELECTRONICS', 'SKGroupCompanies', 'SKGROUP', 'Samsung',
'SKNetworks', 'DoosanCorporation', 'DSEM', 'SKE&SAMERICA', 'PoscoAmerica', 'LGE', 'SKHynixAmerica', 
'SamsungElectronicsNorthAmerica', 'HyundaiMotors', 'HanwhaUSA', 'Kia', 'SamsungSemiconductor', 'DoosanIndustrialVehicle', 
'POSCO', 'LSCableSystemsAmerica', 'DoosanEnerbility', 'LSCABLESYSTEMSAMERICA', 'LGENERGYSOLUTIONMICHIGANTECHCENTER', 'SKENERGY', 
'SeAHUS', 'DoosanElectronics', 'KiaAmerica', 'SAMSUNGELECTRONICSAMERICA', 'NONGSHIM', 'HyundaiMotorGroup', 'SKHynix', 
'GSGLOBALUSA', 'KiaMotorsAmerica', 'LSCABLE&SYSTEM', 'SKhynix', 'KIAMOTORS', 'LGELECTRONICS(ZENITHELECTRONICSLLC)', 
'SKENERGYAMERICAS', 'DHI', 'Asiana', 'GOLFZONAMERICA', 'HANWHAWESTPROPERTIESLLC', 'KOREANAIRLINES', 'LGDISPLAY', 
'LGDISPLAYAMERICA', 'SKLIFESCIENCE', 'KoreanAir', 'ZenithElectronicsCorp', 'QCELLS', 'DoosanIndustrialVehicleAmericaCorporation', 
'SamsungBiologicsAmerica', 'GSGCorp', 'SAMSUNGBIOLOGICS', 'SKGroup', 'SMEntertainment', 'SeahSteelUSALLC', 
'DoosanIndustrialVehicleAmericaCorp', 'Hanwha', 'HYUNDAIMOTORS', 'HanwhaOCEAN', 'NongshimUSA', 'SeAHSteelUSA', 'LSCable&System']

# 지역언론
## Georgia
url_local_1 = 'https://www.ajc.com/news/'
url_local_2 = 'valdostadailytimes.com'
url_local_20 = 'https://www.savannahnow.com/news/local/'
## California
url_local_3 = 'https://www.mercurynews.com/latest-headlines/'
url_local_4 = 'https://www.latimes.com/california'
## Texas
url_local_5 = 'https://www.dallasnews.com/news/'
## New York
url_local_7 = 'https://nypost.com/news/'
## New Jersey
url_local_9 = 'https://www.northjersey.com/news/'
url_local_10 = 'https://www.nj.com/'
url_local_10_list = ['https://www.nj.com/starledger/', 'https://www.nj.com/atlantic/', 'https://www.nj.com/bergen/',
                     'https://nj.com/burlington/', 'https://www.nj.com/camden/', 'https://www.nj.com/cape-may-county/',
                     'https://www.nj.com/cumberland/', 'https://www.nj.com/essex/', 'https://www.nj.com/gloucester-county/',
                     'https://www.nj.com/hudson/', 'https://www.nj.com/hunterdon/', 'https://www.nj.com/mercer/',
                     'https://www.nj.com/middlesex/', 'https://www.nj.com/monmouth/', 'https://www.nj.com/morris/',
                     'https://www.nj.com/ocean/', 'https://www.nj.com/passaic-county/', 'https://www.nj.com/salem/',
                     'https://www.nj.com/somerset/', 'https://www.nj.com/sussex-county/', 'https://www.nj.com/union/',
                     'https://www.nj.com/warren/']
## North Carolina
url_local_11 = 'https://carolinapublicpress.org/recent-news/'
## District of Columbia
url_local_13 = 'https://www.washingtonpost.com/latest-headlines/'
url_local_14 = 'https://www.washingtontimes.com/news/world/'
## Virginia
url_local_15 = 'https://www.vpm.org/vpm-news'
url_local_16 = 'https://richmondmagazine.com/news/news'
## Maryland
url_local_17 = 'https://www.baltimoresun.com/latest-headlines/'
url_local_18 = 'https://www.fox5dc.com/tag/us/md'
url_local_19 = 'https://foxbaltimore.com/news/local'

# 데이터프레임
All_articles = pd.read_csv('US_All Articles' + datetime.now().strftime("_%y%m%d") + '.csv')
All_articles = All_articles.to_dict(orient='records')
All_error_list = pd.read_csv('US_ All Error List' + datetime.now().strftime("_%y%m%d") + '.csv')
All_error_list = All_error_list.to_dict(orient='records')
Local_articles = []
today_list = [date_util((datetime.now()- timedelta(days=1)).strftime("%Y-%m-%d")), date_util((datetime.now()).strftime("%Y-%m-%d"))]
today = date_util(datetime.now().strftime("%Y-%m-%d"))

################################################################ <지역언론> ###################################################################
###############################################<url_local_1>###############################################
#url_local_1 = "https://www.ajc.com/news/"
wd = initialize_chrome_driver()
wd.get(url_local_1)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[Georgia] '+url_local_1
try:
  news_blocks = soup.find_all('div',class_=['c-homeList', 'home-headline', 'headline', 'c-homeList left-photo','c-homeList no-photo','headline-box'])
  if not news_blocks: All_error_list.append({'Error Link': url_local_1, 'Error': "None News"})
  else:
    for block in news_blocks:
        date = block.find('span', class_='isTease article-timestamp')
        if not date: continue
        article_date = date_util(date.text.strip())
        if not article_date: error_message = Error_Message(error_message, "None Date")
        if article_date in today_list:
          link = block.find('a')['href']
          article_link = f'https://www.ajc.com{link}'
          if not article_link: error_message = Error_Message(error_message, "None Link")
          try:
              article = Article(article_link, language = 'en') # URL과 언어를 입력
              article.download()
              article.parse()
              title = article.title
              if not title: error_message = Error_Message(error_message, "None Title")
              content = article.text
              if not content: content = scrap_context(article_link)
              if error_message != str():
                    All_error_list.append({
                      'Error Link': url_local_1,
                      'Error': error_message
                    })
              else:
                  if content:
                    All_articles.append({
                      'Title': title,
                      'Link': article_link,
                      'Content(RAW)': content
                    })
                    Local_articles.append({
                      'Local Site': local,
                      'Title': title,
                      'Link': article_link,
                      'Content(RAW)': content
                    })
          except Exception as e:
              All_error_list.append({
                  'Error Link': article_link,
                  'Error': str(e)
                  })
except Exception as e:
  All_error_list.append({
      'Error Link': url_local_1,
      'Error': str(e)
      })
###############################################<url_local_3>###############################################
# url_local_3 = "https://www.mercurynews.com/latest-headlines/"
try:
    wd = initialize_chrome_driver()
    wd.get(url_local_3)
    time.sleep(5)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    error_message = str()
    local = '[California] '+url_local_3
    try:
        news_items = soup.find_all('a', class_='article-title')
        for item in news_items:
            link = item['href']
            if not link: error_message = Error_Message(error_message, "None Link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                article_date = date_util(str(article.publish_date))
                if not article_date : error_message = Error_Message(error_message, "None Date")
                else:
                  if article_date in today_list :
                      title = article.title
                      if not title : error_message = Error_Message(error_message, "None Title")
                      text = article.text
                      if not text: text = scrap_context(link)
                      if error_message != str():
                                All_error_list.append({
                                    'Error Link': url_local_3,
                                    'Error': error_message
                                })               
                      else:
                          if text:
                            All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
                            Local_articles.append({
                                'Local Site': local,
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
            except Exception as e:
                  All_error_list.append({
                      'Error Link': link,
                      'Error': str(e)
                      })
    except Exception as e:
      All_error_list.append({
          'Error Link': url_local_3,
          'Error': str(e)
          })
except Exception as e:
  All_error_list.append({
      'Error Link': url_local_3,
      'Error': str(e)
      })
###############################################<url_local_4>###############################################
#url_local_4 = 'https://www.latimes.com/california'
wd = initialize_chrome_driver()
wd.get(url_local_4)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[California] '+url_local_4
try:
    date_items = soup.find_all('p', class_='promo-timestamp')
    for item in date_items:
        date = date_util(item.text.strip())
        if date in today_list:
          link = item.find_parent().find('a')['href']
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              title = article.title
              if not title: error_message = Error_Message(error_message, "None Title")
              text = article.text
              if not text: text = scrap_context(link)
              if error_message != str():
                    All_error_list.append({
                        'Error Link': url_local_4,
                        'Error': error_message
                    })
              else:
                  if text:
                      All_articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': text
                      })
                      Local_articles.append({
                          'Local Site': local,
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': text
                      })
          except Exception as e:
              All_error_list.append({
                  'Error Link': link,
                  'Error': str(e)
                  })
except Exception as e:
  All_error_list.append({
      'Error Link': url_local_4,
      'Error': str(e)
      })
###############################################<url_local_5>###############################################
#url_local_5 = "https://www.dallasnews.com/news/"
'''
wd = initialize_chrome_driver()
wd.get(url_local_5)
time.sleep(5)
try:
    for _ in range(5):
        try:
            button = wd.find_element(By.ID, 'load-more')
            button.click()
            time.sleep(1)
        except ElementClickInterceptedException:
            # ElementClickInterceptedException 발생 시 무시하고 계속 진행
            pass
except Exception as e:
    All_error_list.append({
        'Error Link': url_local_5,
        'Error': str(e)
    })
  
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[Texas] '+url_local_5
try:
  news_blocks = soup.find_all('div', class_='dmnc_generic-cards-card-module__FcjA3 w-full py-6 flex border-gray-light relative')
  if not news_blocks: All_error_list.append({'Error Link': url_local_5, 'Error': "None News"})
  for item in news_blocks :
    date_tag = item.find('span', class_='dmnc_generic-article-elements-article-elements-module__l-ds8 secondaryRoman secondaryRoman-10 capitalize text-gray-medium')
    date_string = date_tag.text.strip()
    article_date = date_util(date_string)
    if not article_date : error_message = Error_Message(error_message, "None Date")
    if article_date in today_list:
      link = item.find('a')['href']
      link = f'https://www.dallasnews.com{link}'
      if not link: error_message = Error_Message(error_message, "None link")
      try:
          article = Article(link, language = 'en') # URL과 언어를 입력
          article.download()
          article.parse()
          title = article.title
          if not title: error_message = Error_Message(error_message, "None Title")
          content = article.text
          if not content: content = scrap_context(link)
          if error_message != str():
                    All_error_list.append({
                        'Error Link': url_local_5,
                        'Error': error_message
                        })
          else:
              if content:
                    All_articles.append({
                      'Title': title,
                      'Link': link,
                      'Content(RAW)': content
                      })
                    Local_articles.append({
                      'Local Site': local,
                      'Title': title,
                      'Link': link,
                      'Content(RAW)': content
                      })
      except Exception as e:
        All_error_list.append({
            'Error Link': link,
            'Error': str(e)
            })
except Exception as e:
  All_error_list.append({
      'Error Link': url_local_5,
      'Error': str(e)
  })
  '''
###############################################<url_local_7>###############################################
#url_local_7 = 'https://nypost.com/news/'
wd = initialize_chrome_driver()
wd.get(url_local_7)
time.sleep(5)
for _ in range(5):
  button = wd.find_element(By.CLASS_NAME, 'see-more')
  button.click()
  time.sleep(1)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[New York] '+url_local_7
try:
    date_items = soup.find_all('span', class_='meta meta--byline')
    for item in date_items:
        date = date_util(item.text.strip())
        if date in today_list:
          link = item.find_parent().find('a')['href']
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              title = article.title
              if not title: error_message = Error_Message(error_message, "None Title")
              text = article.text
              if not text: text = scrap_context(link)
              if error_message != str():
                    All_error_list.append({
                        'Error Link': link,
                        'Error': error_message
                    })
              else:
                  if text:
                      All_articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': text
                      })
                      Local_articles.append({
                          'Local Site': local,
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': text
                      })
          except Exception as e:
              All_error_list.append({
                  'Error Link': link,
                  'Error': str(e)
                  })
except Exception as e:
  All_error_list.append({
      'Error Link': url_local_7,
      'Error': str(e)
  })
###############################################<url_local_9>###############################################
# url_local_9 = 'https://www.northjersey.com/news/'
wd = initialize_chrome_driver()
wd.get(url_local_9)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[New Jersey] '+url_local_9
base_url = 'https://www.northjersey.com'
try:
    # 'gnt_x__nft gnt_m_flm_a' 클래스를 제외한 'gnt_m_flm_a' 클래스를 가진 모든 <a> 태그 찾기
    news_items = soup.find_all('a', class_='gnt_m_flm_a')
    for item in news_items:
        if 'gnt_x__nft' not in item['class']:
            link = base_url + item['href']
            if not link : error_message = Error_Message(error_message, "None Link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                article_date = date_util(str(article.publish_date))
                if not article_date : error_message = Error_Message(error_message, "None Date")
                else:
                    if article_date in today_list :
                        title = article.title
                        if not title : error_message = Error_Message(error_message, "None Title")
                        text = article.text
                        if not text: text = scrap_context(link)
                        if error_message != str():
                            All_error_list.append({
                                'Error Link': url_local_9,
                                'Error': error_message
                            })
                        else:
                            if text:
                                All_articles.append({
                                    'Title': title,
                                    'Link': link,
                                    'Content(RAW)': text
                                })
                                Local_articles.append({
                                    'Local Site': local,
                                    'Title': title,
                                    'Link': link,
                                    'Content(RAW)': text
                                })
            except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                    })
except Exception as e:
  All_error_list.append({
      'Error Link': url_local_9,
      'Error': str(e)
  })
###############################################<url_local_10>###############################################
# url_local_10 = 'https://www.nj.com/'
local = '[New Jersey] '+url_local_10
error_message = str()
date_items = []
try:
    for url in url_local_10_list:
      wd = initialize_chrome_driver()
      wd.get(url)
      time.sleep(5)
      html = wd.page_source
      soup = BeautifulSoup(html, 'html.parser')
      Newdate_items = soup.find_all('time')
      date_items += Newdate_items
except Exception as e:
  error_list.append({
      'Error Link': url_local_10,
      'Error': str(e)
  })
try:
    for item in date_items:
        date = date_util(item.text.strip())
        if date in today_list:
          link = item.find_parent().find_parent().find_parent()['href']
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              title = article.title
              if not title: error_message = Error_Message(error_message, "None Title")
              text = article.text
              if not text: text = scrap_context(link)
              if error_message != str():
                    All_error_list.append({
                        'Error Link': link,
                        'Error': error_message
                    })
              else:
                  if text:
                      All_articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': text
                      })
                      Local_articles.append({
                          'Local Site': local,
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': text
                      })
          except Exception as e:
              All_error_list.append({
                  'Error Link': link,
                  'Error': str(e)
              })
except Exception as e:
  All_error_list.append({
      'Error Link': url_local_10,
      'Error': str(e)
  })
###############################################<url_local_11>###############################################
# url_local_11 = 'https://carolinapublicpress.org/recent-news/'
wd = initialize_chrome_driver()
wd.get(url_local_11)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[North Carolina] ' + url_local_11
try:
    # 뉴스 아이템 추출
    news_items = soup.find_all('div', class_='entry-container')
    if not news_items:
        All_error_list.append({
            'Error Link': url_local_11,
            'Error': "None News"
        })
    else:
        for item in news_items:
          title_tag = item.find('h2', class_='entry-title')
          if not title_tag: 
            error_message = Error_Message(error_message, "None Title")
            continue
          title = title_tag.get_text().strip() 
          if not title: error_message = Error_Message(error_message, "None Title")
          link = title_tag.find('a')['href'] if title_tag and title_tag.find('a') else None
          if not link : error_message = Error_Message(error_message, "None Link")
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              article_date = date_util(str(article.publish_date))
              if not article_date:
                error_message = Error_Message(error_message, "None Date")
              else:
                if article_date in today_list :
                    text = article.text
                    if not text: content = scrap_context(article_link)
                    if error_message != str():
                        All_error_list.append({
                              'Error Link': url_local_11,
                              'Error': error_message
                              })
                    else:
                        if text:
                          All_articles.append({
                              'Title': title,
                              'Link': link,
                              'Content(RAW)': text
                              })
                          Local_articles.append({
                              'Local Site': local,
                              'Title': title,
                              'Link': link,
                              'Content(RAW)': text
                              })
          except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })  
except Exception as e:
    All_error_list.append({
        'Error Link': url_local_11,
        'Error': str(e)
    })  
###############################################<url_local_13>###############################################
# url_local_13 = 'https://www.washingtonpost.com/latest-headlines/'
wd = initialize_chrome_driver()
wd.get(url_local_13)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[District of Columbia] '+url_local_13
try:
    date_items = soup.find_all('span', class_='timestamp gray-dark')
    for item in date_items:
        date = date_util(item.text.strip())
        if date in today_list:
            link = item.find_parent().find_parent().find('a')['href']
            if not link: error_message = Error_Message(error_message, "None Link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                title = article.title
                if not title: error_message = Error_Message(error_message, "None Title")
                text = article.text
                if not text: text = scrap_context(link)
                if error_message != str():
                    All_error_list.append({
                        'Error Link': url_local_13,
                        'Error': error_message
                    })
                else:
                    if text:
                        All_articles.append({
                            'Local Site': local,
                            'Title': title,
                            'Link': link,
                            'Content(RAW)': text
                        })
                        Local_articles.append({
                            'Local Site': local,
                            'Title': title,
                            'Link': link,
                            'Content(RAW)': text
                        })
            except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })  
except Exception as e:
    All_error_list.append({
        'Error Link': url_local_13,
        'Error': str(e)
    })  
###############################################<url_local_14>###############################################
# url_local_14 = "https://www.washingtontimes.com/news/world/"
wd = initialize_chrome_driver()
wd.get(url_local_14)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[District of Columbia] '+url_local_14
base_url = "https://www.washingtontimes.com"
try:
    news_items = soup.find_all('h2', class_='article-headline')
    for item in news_items:
        relative_link = item.find('a')['href']
        link = base_url + relative_link
        try:
            article = Article(link, language='en')
            article.download()
            article.parse()
            article_date = date_util(str(article.publish_date))
            if not article_date : error_message = Error_Message(error_message, "None Date")
            else:
              if article_date in today_list :
                  title = article.title
                  if not title : error_message = Error_Message(error_message, "None Title")
                  text = article.text
                  if not text: text = scrap_context(link)
              if error_message != str():
                        All_error_list.append({
                            'Error Link': url_local_14,
                            'Error': error_message
                        })
              else:
                    if text:
                        All_articles.append({
                            'Title': title,
                            'Link': link,
                            'Content(RAW)': text
                        })
                        Local_articles.append({
                            'Local Site': local,
                            'Title': title,
                            'Link': link,
                            'Content(RAW)': text
                        })
        except Exception as e:
          All_error_list.append({
              'Error Link': link,
              'Error': str(e)
          })  
except Exception as e:
    All_error_list.append({
        'Error Link': url_local_14,
        'Error': str(e)
    })  
###############################################<url_local_15>###############################################
# url_local_15 = 'https://www.vpm.org/vpm-news'
wd = initialize_chrome_driver()
wd.get(url_local_15)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[Virginia] ' + url_local_15
try:
    # 뉴스 아이템 추출
    news_items = soup.find_all('div', class_='PromoA-content')
    if not news_items:
        All_error_list.append({
            'Error Link': url_local_15,
            'Error': "None News"
        })
    else:
        for item in news_items:
            title_tag = item.find('div', class_='PromoA-title').find('a')
            if not title_tag: 
                error_message = Error_Message(error_message, "None Title")
                continue
            title = title_tag.get_text().strip()
            if not title: 
                error_message = Error_Message(error_message, "None Title")
            link = title_tag['href'] if title_tag else None
            if not link : error_message = Error_Message(error_message, "None Link")
            try :
                article = Article(link, language='en')
                article.download()
                article.parse()
                article_date = date_util(str(article.publish_date))
                if not article_date: error_message = Error_Message(error_message, "None Date")
                if article_date in today_list :
                    text = article.text
                    if not text: text = scrap_context(link)
                    if error_message != str():
                        All_error_list.append({
                              'Error Link': url_local_15,
                              'Error': error_message
                              })
                    else:
                        if text:
                          All_articles.append({
                              'Title': title,
                              'Link': link,
                              'Content(RAW)': text
                              })
                          Local_articles.append({
                              'Local Site': local,
                              'Title': title,
                              'Link': link,
                              'Content(RAW)': text
                              })
            except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                  })  
except Exception as e:
      All_error_list.append({
          'Error Link': url_local_15,
          'Error': str(e)
      })  
###############################################<url_local_16>###############################################
# url_local_16 = 'https://richmondmagazine.com/news/news'
wd = initialize_chrome_driver()
wd.get(url_local_16)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[Virginia] ' + url_local_16
try:
    # 뉴스 아이템 추출
    news_items = soup.find_all('div', class_='mp-item-wrapper')
    if not news_items:
        All_error_list.append({
            'Error Link': url_local_16,
            'Error': "None News"
        })
    else:
        for item in news_items:
            title_tag = item.find('h3').find('a')
            if not title_tag: 
                error_message = Error_Message(error_message, "None Title")
                continue
            title = title_tag.get_text().strip()
            if not title: 
                error_message = Error_Message(error_message, "None Title")
            link = title_tag['href'] if title_tag else None
            if not link : error_message = Error_Message(error_message, "None Link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                article_date = date_util(str(article.publish_date))
                if not article_date:
                  error_message = Error_Message(error_message, "None Date")
                else:
                  if article_date in today_list :
                      text = article.text
                      if not text: text = scrap_context(link)
                      if error_message != str():
                          All_error_list.append({
                            'Error Link': url_local_16,
                            'Error': error_message
                            })
                      else:
                          if text:
                            All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                                })
                            Local_articles.append({
                                'Local Site': local,
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                                })
            except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })  
except Exception as e:
      All_error_list.append({
          'Error Link': url_local_16,
          'Error': str(e)
      })  
###############################################<url_local_17>###############################################
# url_local_17 = 'https://www.baltimoresun.com/latest-headlines/'
wd = initialize_chrome_driver()
wd.get(url_local_17)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[Maryland] ' + url_local_17
try:
    # 뉴스 아이템 추출
    news_items = soup.find_all('div', class_='article-info')
    if not news_items:
        All_error_list.append({
            'Error Link': url_local_17,
            'Error': "None News"
        })
    else:
        for item in news_items:
            link = item.find('a', class_='article-title')['href']
            if not link : error_message = Error_Message(error_message, "None Link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                article_date = date_util(str(article.publish_date))
                if not article_date: error_message = Error_Message(error_message, "None Date")
                else:
                  if article_date in today_list :
                      title = article.title
                      if not title : error_message = Error_Message(error_message, "None Title")
                      text = article.text
                      if not text: text = scrap_context(link)
                      if error_message != str():
                          All_error_list.append({
                              'Error Link': url_local_17,
                              'Error': error_message
                              })
                      else:
                          if text:
                            All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
                            Local_articles.append({
                                'Local Site': local,
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
            except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })
except Exception as e:
    All_error_list.append({
        'Error Link': url_local_17,
        'Error': str(e)
    })
###############################################<url_local_18>###############################################
# url_local_18 = 'https://www.fox5dc.com/tag/us/md'
wd = initialize_chrome_driver()
wd.get(url_local_18)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[Maryland] ' + url_local_18
base_url = 'https://www.fox5dc.com'
try:
    # 뉴스 아이템 추출
    news_items = soup.find_all('div', class_='info')
    if not news_items:
        All_error_list.append({
            'Error Link': url_local_18,
            'Error': "None News"
        })
    else:
        for item in news_items:
            date_tag = item.find('time', class_='time')
            date = date_tag['datetime'] if date_tag else None
            article_date = date_util(str(date))
            if not article_date:
                error_message = Error_Message(error_message, "None Date")
            title_tag = item.find('h3', class_='title').find('a')
            if not title_tag: 
                error_message = Error_Message(error_message, "None Title")
                continue
            title = title_tag.get_text().strip()
            if not title: 
                error_message = Error_Message(error_message, "None Title")
            relative_link = title_tag['href'] if title_tag else None
            link = base_url + relative_link if relative_link else None
            if not link : error_message = Error_Message(error_message, "None Link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                if article_date in today_list :
                    text = article.text
                    if not text: text = scrap_context(link)
                    if error_message != str():
                      All_error_list.append({
                            'Error Link': url_local_18,
                            'Error': error_message
                            })
                    else:
                        if text:
                            All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                                })
                            Local_articles.append({
                                'Local Site': local,
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                                })
            except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })  
except Exception as e:
      All_error_list.append({
          'Error Link': url_local_18,
          'Error': str(e)
      })  
###############################################<url_local_19>###############################################
# url_local_19 = 'https://foxbaltimore.com/news/local'
wd = initialize_chrome_driver()
wd.get(url_local_19)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
local = '[Maryland] ' + url_local_19
try:
    # 뉴스 아이템 추출
    news_items = soup.find_all('li', class_='teaserItem')
    if not news_items:
        All_error_list.append({
            'Error Link': url_local_19,
            'Error': "None News"
        })
    else:
        for item in news_items:
            date_tag = item.find('span', class_='publishedSince')
            date = date_tag.get_text() if date_tag else None
            article_date = date_util(str(date))
            if not article_date: 
                error_message = Error_Message(error_message, "None Date")
            title_tag = item.find('a')
            if not title_tag: 
                error_message = Error_Message(error_message, "None Title")
                continue
            base_url = 'https://foxbaltimore.com'
            relative_link = title_tag['href'] if title_tag else None
            link = base_url + relative_link if relative_link else None
            if not link : error_message = Error_Message(error_message, "None Link")
            if article_date in today_list :
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    title = article.title
                    if not title: error_message = Error_Message(error_message, "None Title")
                    text = article.text
                    if not text: text = scrap_context(link)
                    if error_message != str():
                      All_error_list.append({
                            'Error Link': url_local_19,
                            'Error': error_message
                            })
                    else:
                        if text:
                            All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                                })
                            Local_articles.append({
                                'Local Site': local,
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                                })
                except Exception as e:
                    All_error_list.append({
                        'Error Link': link,
                        'Error': str(e)
                    })
except Exception as e:
      All_error_list.append({
          'Error Link': url_local_19,
          'Error': str(e)
      })
###############################################<url_local_20>###############################################
local_news_sites = [url_local_20]
# search 검색어 & 링크 만들기
def generate_bing_news_url(site_keyword):
    search_q = f"site%3A{site_keyword}"
    url = f"https://www.bing.com/news/search?q={search_q}&qft=interval%3d%227%22&form=YFNR&setlang=en-US&setmkt=en-US&freshness=Day&sort=Date"
    return url
# local 변수 만들기
local_states = {
                'https://www.savannahnow.com/news/local/': '[Georgia] https://www.savannahnow.com/news/local/'
                }
bing_url_list = [] ; local_list = []
for local_news_site in local_news_sites:
    bing_news_url = generate_bing_news_url(local_news_site)
    bing_url_list.append(bing_news_url)
    local_list.append(local_states[local_news_site])
def scroll_down(driver):
    # Scroll down using the END key to trigger additional content loading
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(2)  # Adjust sleep time based on your page loading time
    # Function to check if the page has reached the end
def is_end_of_page(driver):
    try:
        # Check if there is a button indicating the end of the page
        driver.find_element(By.XPATH, '//button[text()="No more items to show."]')
        return True
    except NoSuchElementException:
        return False
        
# scrapping
for index, bing_url in enumerate(bing_url_list):
    wd = initialize_chrome_driver()
    wd.get(bing_url)
    time.sleep(3)
    # Scroll down a few times to load more articles
    for _ in range(50): # Adjust the number of scrolls based on your needs
        before_scroll_height = wd.execute_script("return document.body.scrollHeight")
        scroll_down(wd)
        time.sleep(2)  # Adjust sleep time based on your page loading time
        after_scroll_height = wd.execute_script("return document.body.scrollHeight")
        # Check if new content is being loaded
        if before_scroll_height == after_scroll_height:
            break
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    news_items = soup.find_all('div', class_='caption')
    for news_item in news_items:
        title = news_item.select_one('.title').text
        link = news_item.select_one('.title')['href']
        date = news_item.find('span', {'tabindex': '0'}).text.strip()
        article_date = date_util(date)
        local = local_list[index]
        if article_date in today_list:
            text = scrap_context(link)
            if text:
                All_articles.append({
                                    'Title': title,
                                    'Link': link,
                                    'Content(RAW)': text
                                })
                Local_articles.append({
                                    'Local Site': local,
                                    'Title': title,
                                    'Link': link,
                                    'Content(RAW)': text
                                })

########################################### <All Articles Print> ##############################################
articles = pd.DataFrame(All_articles)
articles = articles.drop_duplicates(subset=['Link'])
articles = articles.drop_duplicates(subset=['Title'])
articles.reset_index(drop=True, inplace=True)
articles.to_csv('US_All Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

articles4 = pd.DataFrame(Local_articles)
articles4 = articles4.drop_duplicates(subset=['Link'])
articles4 = articles4.drop_duplicates(subset=['Title'])
articles4.reset_index(drop=True, inplace=True)
articles4.to_csv('US_Local Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

error_list = pd.DataFrame(All_error_list)
error_list.to_csv('US_ All Error List' + datetime.now().strftime("_%y%m%d") + '.csv')
