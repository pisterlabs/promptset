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

# 방산업체 + NASA
url_32 = 'https://www.lockheedmartin.com/en-us/news.html'
url_33 = 'https://boeing.mediaroom.com/news-releases-statements'
url_34 = 'https://www.rtx.com/rss-feeds/news'   ### rss feed
url_35 = 'https://news.northropgrumman.com/news/releases'
url_36 = 'https://www.gd.com/news/news-feed?page=0&types=Press Release'
url_37 = 'https://www.baesystems.com/en/newsroom'
url_38 = 'https://www.l3harris.com/ko-kr/newsroom/search?size=n_10_n&sort-field%5Bname%5D=Publish%20Date&sort-field%5Bvalue%5D=created_date&sort-field%5Bdirection%5D=desc&sort-direction='
url_39 = 'https://investor.textron.com/rss/pressrelease.aspx'   ### rss feed
url_40 = 'https://www.nasa.gov/news/all-news/'

# 데이터프레임
All_articles = pd.read_csv('US_All Articles' + datetime.now().strftime("_%y%m%d") + '.csv')
All_articles = All_articles.to_dict(orient='records')
All_error_list = pd.read_csv('US_ All Error List' + datetime.now().strftime("_%y%m%d") + '.csv')
All_error_list = All_error_list.to_dict(orient='records')
DefenseIndustry_articles = []
today_list = [date_util((datetime.now()- timedelta(days=1)).strftime("%Y-%m-%d")), date_util((datetime.now()).strftime("%Y-%m-%d"))]
today = date_util(datetime.now().strftime("%Y-%m-%d"))

################################################################ <방산 업체> ###################################################################
########################################### <32> ##############################################
#url_32 = 'https://www.lockheedmartin.com/en-us/news.html'
wd = initialize_chrome_driver()
wd.get(url_32)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find('div', class_='newsListingContainer').find_all('div', class_='relatedItemDate')
  if not date_blocks: All_error_list.append({'Error Link': url_32, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.lockheedmartin.com' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language = 'en') # URL과 언어를 입력
            article.download()
            article.parse()
            title = article.title
            if not title: error_message = Error_Message(error_message, "None Title")
            content = article.text
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
              All_error_list.append({
                'Error Link': url_32,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              DefenseIndustry_articles.append({
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
      'Error Link': url_32,
      'Error': str(e)
      })
########################################### <33> ##############################################
#url_33 = 'https://boeing.mediaroom.com/news-releases-statements'
wd = initialize_chrome_driver()
wd.get(url_33)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date = None, None
try:
  date_blocks = soup.find_all('div', class_='wd_date')
  if not date_blocks: All_error_list.append({'Error Link': url_33, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        links = block.find_parent().find_parent().find_all('a')
        if len(links) > 1: article_link = links[2].get('href')
        else: article_link = block.find_parent().find_parent().find('a')['href']
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language = 'en') # URL과 언어를 입력
            article.download()
            article.parse()
            title = article.title
            if not title: error_message = Error_Message(error_message, "None Title")
            content = article.text
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
              All_error_list.append({
                'Error Link': url_33,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              DefenseIndustry_articles.append({
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
      'Error Link': url_33,
      'Error': str(e)
      })
########################################### <34> ##############################################
#url_34 = 'https://www.rtx.com/rss-feeds/news'    #RSS FEED
feed = feedparser.parse(url_34)
try:
  # 각 기사에 대한 정보 추출
  for entry in feed.entries:
      title = entry.title
      if not title: error_message = Error_Message(error_message, "None Title")
      link = entry.link
      if not link: error_message = Error_Message(error_message, "None Link")
      date = date_util(entry.published)
      if not date: error_message = Error_Message(error_message, "None date")
      if date in today_list:
        try:
            article = Article(link, language = 'en') # URL과 언어를 입력
            article.download()
            article.parse()
            content = article.text
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
                      All_error_list.append({
                          'Error Link': url_34,
                          'Error': error_message
                          })
            else:
                      All_articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
                      DefenseIndustry_articles.append({
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
        'Error Link': url_34,
        'Error': str(e)
    })
########################################### <35> ##############################################
# url_35 = 'https://news.northropgrumman.com/news/releases'
wd = initialize_chrome_driver()
wd.get(url_35)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('div', class_='index-item-info')
  if not date_blocks: All_error_list.append({'Error Link': url_35, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://news.northropgrumman.com' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language='en')
            article.download()
            article.parse()
            title = article.title
            if not title:
              error_message = Error_Message(error_message, "None Title")
            text = article.text
            if not text:
              error_message = Error_Message(error_message, "None Content")
            if error_message != str():
                            All_error_list.append({
                                'Error Link': url_35,
                                'Error': error_message
                            })     
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              DefenseIndustry_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
        except Exception as e:
            All_error_list.append({
                'Error Link': article_link,
                'Error': str(e)
            })
except Exception as e:
    All_error_list.append({
        'Error Link': url_35,
        'Error': str(e)
    })
########################################### <36> ##############################################
#url_36 = 'https://www.gd.com/news/news-feed?page=0&types=Press Release'
wd = initialize_chrome_driver()
wd.get(url_36)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find('div', class_='news-feed-target').find_all('div', class_='news__publish-date')
  if not date_blocks: All_error_list.append({'Error Link': url_36, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.contents[0].strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find('a')['href']
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language = 'en') # URL과 언어를 입력
            article.download()
            article.parse()
            title = article.title
            if not title: error_message = Error_Message(error_message, "None Title")
            content = article.text
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
              All_error_list.append({
                'Error Link': url_36,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              DefenseIndustry_articles.append({
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
      'Error Link': url_36,
      'Error': str(e)
      })
########################################### <37> ##############################################
#url_37 = 'https://www.baesystems.com/en/newsroom'
wd = initialize_chrome_driver()
wd.get(url_37)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('div', class_='cell__body')
  if not date_blocks:
    All_error_list.append({
            'Error Link': url_37,
            'Error': "Date Blocks"
          })
  else:
    for block in date_blocks:
      date_str = block.find_next('div',class_='searchresult-tags').text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find('a')['href']
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        title = block.find_parent().find('h3').text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        bodys = article_soup.find('div', class_='content-container').get_text().strip()
        if not bodys: error_message = Error_Message(error_message, "None Contents")
        if error_message != str():
          All_error_list.append({
            'Error Link': url_37,
            'Error': error_message
          })
        else:
          All_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
          DefenseIndustry_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
except Exception as e:
  All_error_list.append({
      'Error Link': url_37,
      'Error': str(e)
      })
########################################### <38> ##############################################
#url_38 = 'https://www.l3harris.com/ko-kr/newsroom/search?size=n_10_n&sort-field%5Bname%5D=Publish%20Date&sort-field%5Bvalue%5D=created_date&sort-field%5Bdirection%5D=desc&sort-direction='
wd = initialize_chrome_driver()
wd.get(url_38)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find('ul', class_='search-results-page__result-list').find_all('p', class_='d-block mb-4 mb-lg-8 subtitle')
  if not date_blocks: All_error_list.append({'Error Link': url_38, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.get_text().split('|')[1].strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find('a')['href']
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language='en')
            article.download()
            article.parse()
            title = article.title
            if not title:
              error_message = Error_Message(error_message, "None Title")
            text = article.text
            if not text:
              error_message = Error_Message(error_message, "None Content")
            if error_message != str():
                            All_error_list.append({
                                'Error Link': url_38,
                                'Error': error_message
                            })     
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              DefenseIndustry_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
        except Exception as e:
            All_error_list.append({
                'Error Link': article_link,
                'Error': str(e)
            })
except Exception as e:
    All_error_list.append({
        'Error Link': url_38,
        'Error': str(e)
    })
########################################### <39> ##############################################
feed = feedparser.parse(url_39)
try:
  # 각 기사에 대한 정보 추출
  for entry in feed.entries:
      title = entry.title
      if not title: error_message = Error_Message(error_message, "None Title")
      link = entry.link
      if not link: error_message = Error_Message(error_message, "None Link")
      date = date_util(entry.published)
      if not date: error_message = Error_Message(error_message, "None date")
      if date in today_list:
        try:
            article = Article(link, language = 'en') # URL과 언어를 입력
            article.download()
            article.parse()
            content = article.text
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
                      All_error_list.append({
                          'Error Link': url_39,
                          'Error': error_message
                          })
            else:
                      All_articles.append({
                          'Title': title,
                          'Link': link,
                          'Content(RAW)': content
                          })
                      DefenseIndustry_articles.append({
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
        'Error Link': url_39,
        'Error': str(e)
    })
########################################### <40> ##############################################
# url_40 = https://www.nasa.gov/news/all-news/
wd = initialize_chrome_driver()
wd.get(url_40)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('div', class_='hds-content-item-inner')
    if not news_items: All_error_list.append({'Error Link': url_40, 'Error': "Entire Error1"})
    for item in news_items:
        link_tag = item.find('a', class_='hds-content-item-heading')
        if not link_tag: error_message = Error_Message(error_message, "Entire Error2")
        title_tag = link_tag.find('h3')
        if not title_tag: error_message = Error_Message(error_message, "Entire Error3")
        link = link_tag.get('href')
        if not link: error_message = Error_Message(error_message, "None link")
        wd = initialize_chrome_driver()
        wd.get(link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        date_tag = article_soup.find('span', class_='heading-12 text-uppercase')
        if date_tag:
            date = date_util(date_tag.text)
            if date in today_list:
                title = title_tag.text.strip()
                if not title: error_message = Error_Message(error_message, "None title")
                paragraphs = article_soup.find_all('p')
                content_list = [p.text for p in paragraphs if p]
                content = '\n'.join(content_list)
                if not content: error_message = Error_Message(error_message, "None contents")
                if error_message != str():
                    All_error_list.append({
                    'Error Link': link,
                    'Error': error_message
                    })
                else:
                    All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                    DefenseIndustry_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
except Exception as e:
    All_error_list.append({
     'Error Link': url_40,
     'Error': str(e)
     })
  
########################################### <All Articles Print> ##############################################
articles = pd.DataFrame(All_articles)
articles = articles.drop_duplicates(subset=['Link'])
articles = articles.drop_duplicates(subset=['Title'])
articles.reset_index(drop=True, inplace=True)
articles.to_csv('US_All Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

articles3 = pd.DataFrame(DefenseIndustry_articles)
articles3 = articles3.drop_duplicates(subset=['Link'])
articles3 = articles3.drop_duplicates(subset=['Title'])
articles3.reset_index(drop=True, inplace=True)
articles3.to_csv('US_Defense Industry Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

error_list = pd.DataFrame(All_error_list)
error_list.to_csv('US_ All Error List' + datetime.now().strftime("_%y%m%d") + '.csv')
