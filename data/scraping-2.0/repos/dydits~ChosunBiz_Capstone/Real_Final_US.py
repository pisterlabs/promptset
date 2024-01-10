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
  # USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.2 Safari/605.1.15'
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

# 정부 사이트(연방정부 + 주정부)
url_1 = 'https://www.state.gov/press-releases/'
url_2 = 'https://www.state.gov/department-press-briefings/'
url_3 = 'https://home.treasury.gov/news/press-releases' 
url_4 = 'https://www.defense.gov/News/'
url_5 = 'https://www.army.mil/news'
url_6 = 'https://www.navy.mil/Press-Office/'
url_7 = 'https://www.af.mil/News/Category/22750/'
url_8 = 'https://www.nsa.gov/Press-Room/Press-Releases-Statements/'
url_9 = 'https://www.justice.gov/news'
url_10 = 'https://www.fbi.gov/news/press-releases'
url_11 = 'https://www.doi.gov/news'
url_12 = 'https://www.usda.gov/media/press-releases'
url_13 = 'https://www.ars.usda.gov/news-events/news-archive/'
url_14 = 'https://www.fs.usda.gov/news/releases'
url_15 = 'https://www.fas.usda.gov/newsroom/search'
url_16 = 'https://www.commerce.gov/news'
url_17 = 'https://www.dol.gov/newsroom/releases?agency=All&state=All&topic=All&year=all&page=0'
url_18 = 'https://www.hhs.gov/about/news/index.html'
url_19 = 'https://www.fda.gov/news-events/fda-newsroom/press-announcements'
url_20 = 'https://www.fda.gov/news-events/speeches-fda-officials'
url_21 = 'https://www.transportation.gov/newsroom/press-releases'
url_22 = 'https://www.transportation.gov/newsroom/speeches'
url_23 = 'https://www.energy.gov/newsroom'
url_24 = 'https://www.ed.gov/news/press-releases'
url_25 = 'https://www.ed.gov/news/speeches'
url_26 = 'https://news.va.gov/news/'
url_27 = 'https://www.dhs.gov/news-releases/press-releases'
url_28 = 'https://www.dhs.gov/news-releases/speeches'
url_29 = 'https://www.fema.gov/about/news-multimedia/press-releases'
url_30 = 'https://www.secretservice.gov/newsroom'
url_31 = 'https://www.epa.gov/newsreleases/search'
## Georgia
url_41 = 'https://sos.ga.gov/news/division/31?page=0'
url_42 = 'https://gov.georgia.gov/press-releases'
url_44 = 'https://www.georgia.org/press-releases'
url_45 = 'https://www.gachamber.com/all-news/'
## California
url_46 = 'https://www.sos.ca.gov/administration/news-releases-and-advisories/2023-news-releases-and-advisories'
url_48 = 'https://business.ca.gov/newsroom/'
url_49 = 'https://business.ca.gov/calosba-latest-news/'
url_50 = 'https://www.dir.ca.gov/dlse/DLSE_whatsnew2023.htm' ## 년도 바뀔 때마다 새로운 링크 찾아줘야함(https://www.dir.ca.gov/dlse/DLSE_whatsnew.htm)
url_51 = 'https://www.dir.ca.gov/dosh/DOSH_Archive.html'
url_52 = 'https://www.dir.ca.gov/mediaroom.html'
url_53 = 'https://news.caloes.ca.gov/'
url_54 = 'https://advocacy.calchamber.com/california-works/calchamber-members-in-the-news/'
## Texas
url_55 = 'https://gov.texas.gov/news'
url_56 = 'https://www.texasattorneygeneral.gov/news/releases'
url_57 = 'https://www.txdot.gov/about/newsroom/statewide.html'
url_58 = 'https://www.dps.texas.gov/news/press-releases'
url_59 = 'https://www.twc.texas.gov/news'
url_61 = 'https://comptroller.texas.gov/about/media-center/news//'
url_62 = 'https://www.tdi.texas.gov/news/2023/index.html'
url_63 = 'https://www.txbiz.org/chamber-news'
url_64 = 'https://www.txbiz.org/press-releases'
## New York
url_65 = 'https://www.governor.ny.gov/news'
url_67 = 'https://www.dot.ny.gov/news/press-releases/2023'
url_68 = 'https://ag.ny.gov/press-releases'
url_71 = 'https://chamber.nyc/news'
url_73 = 'https://www.nyserda.ny.gov/About/Newsroom/2023-Announcements'
## New Jersey
url_74 = 'https://www.njchamber.com/press-releases'
url_75 = 'https://www.nj.gov/health/news/'
url_76 = 'https://www.nj.gov/mvc/news/news.htm'
## North Carolina
url_78 = 'https://www.commerce.nc.gov/news/press-releases'
url_80 = 'https://www.ncdor.gov/news/press-releases'
url_81 = 'https://www.iprcenter.gov/news'
url_82 = 'https://edpnc.com/news-events/'
url_83 = 'https://ncchamber.com/category/chamber-updates/'
## Washington D.C.
url_84 = 'https://dc.gov/newsroom'
url_85 = 'https://dcchamber.org/posts/'
url_86 = 'https://planning.dc.gov/newsroom'
url_87 = 'https://dpw.dc.gov/newsroom'
## Virginia
url_88 = 'https://www.governor.virginia.gov/newsroom/news-releases/'
url_89 = 'https://www.vedp.org/press-releases'
url_90 = 'https://www.doli.virginia.gov/category/announcements/'
url_91 = 'https://vachamber.com/category/press-releases/'
## Maryland
url_92 = 'https://governor.maryland.gov/news/press/Pages/default.aspx?page=1'
url_93 = 'https://news.maryland.gov/mde/category/press-release/'
url_94 = 'https://commerce.maryland.gov/media/press-room'
url_95 = 'https://www.dllr.state.md.us/whatsnews/'
url_96 = 'https://www.mdchamber.org/news/'

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
All_articles = []
All_error_list = []
Govern_articles = []
DefenseIndustry_articles = []
Local_articles = []
today_list = [date_util((datetime.now()- timedelta(days=1)).strftime("%Y-%m-%d")), date_util((datetime.now()).strftime("%Y-%m-%d"))]
today = date_util(datetime.now().strftime("%Y-%m-%d"))

################################################################ <정부 사이트 + 방산 업체> ###################################################################
########################################### <1> ##############################################
# url_1 = 'https://www.state.gov/press-releases/'
wd = initialize_chrome_driver()
wd.get(url_1)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('li', class_='collection-result')
    for item in news_items:
        error_message = ''
        link = item.find('a', class_='collection-result__link')['href']
        if not link:
            error_message = Error_Message(error_message, "None Link")
        date_tag = item.find('div', class_='collection-result-meta').find('span', dir='ltr')
        extracted_date = date_tag.text.strip() if date_tag else 'No date found'
        article_date = date_util(extracted_date)
        if not date_tag:
            error_message = Error_Message(error_message, "None date")
        if article_date in today_list :
            try:
                # newspaper : 제목,본문 추출 
                article = Article(link, language='en')
                article.download()
                article.parse()
                title = article.title
                if not title:
                  error_message = Error_Message(error_message, "None Link")
                text = article.text
                if not text:
                  error_message = Error_Message(error_message, "None Content")
                if error_message != str():
                            All_error_list.append({
                                'Error Link': url_1,
                                'Error': error_message
                            })
                else:
                  All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
                  Govern_articles.append({
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
        'Error Link': url_1,
        'Error': str(e)
    })
########################################### <2> ##############################################
# url_2 = 'https://www.state.gov/department-press-briefings/'
wd = initialize_chrome_driver()
wd.get(url_2)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('li', class_='collection-result')
    for item in news_items:
        error_message = ''
        link = item.find('a', class_='collection-result__link')['href']
        if not link:
            error_message = Error_Message(error_message, "None Link")
        date_tag = item.find('div', class_='collection-result-meta').find('span', dir='ltr')
        extracted_date = date_tag.text.strip() if date_tag else 'No date found'
        article_date = date_util(extracted_date)
        if not date_tag:
            error_message = Error_Message(error_message, "None date")
        if article_date in today_list:
            # newspaper : 제목,본문 추출 
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                title = article.title
                if not title:
                  error_message = Error_Message(error_message, "None Link")
                text = article.text
                if not text:
                  error_message = Error_Message(error_message, "None Content")
                if error_message != str():
                            All_error_list.append({
                                'Error Link': url_2,
                                'Error': error_message
                            })
                else:
                  All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
                  Govern_articles.append({
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
        'Error Link': url_2,
        'Error': str(e)
    })
########################################### <3> ##############################################
# url_3 = 'https://home.treasury.gov/news/press-releases' 
wd = initialize_chrome_driver()
wd.get(url_3)
time.sleep(3)
html = wd.page_source
wd.quit()
soup = BeautifulSoup(html, 'html.parser')
try:
    news_items = soup.find_all('div', class_='content--2col__body')
    if not news_items:
        All_error_list.append({
            'Error Link': url_3,
            'Error': "None News"
        })
    else:
        for item in news_items:
            error_message = ''
            link_tag = item.find('a', href=True)
            link = 'https://home.treasury.gov' + link_tag['href']
            if not link:
              error_message = Error_Message(error_message, "None Link")
            date_tag = item.find('time', class_='datetime')
            extracted_date = date_tag['datetime'].split('T')[0]
            article_date = date_util(extracted_date)
            if not date_tag:
              error_message = Error_Message(error_message, "None Date")
            if article_date in today_list:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    title = article.title
                    if not title:
                      error_message = Error_Message(error_message, "None Title")
                    text = article.text
                    if not link:
                      error_message = Error_Message(error_message, "None Content")
                    if error_message != str():
                            All_error_list.append({
                                'Error Link': url_3,
                                'Error': error_message
                            })
                    else:
                      All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
                      Govern_articles.append({
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
        'Error Link': url_3,
        'Error': str(e)
    })
########################################### <4> ##############################################
#url_4 = 'https://www.defense.gov/News/'
wd = initialize_chrome_driver()
wd.get(url_4)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date = None, None
try:
  date_blocks = soup.find_all('time')
  if not date_blocks: All_error_list.append({'Error Link': url_4, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block['data-dateap']
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find_parent().find('a').find_next('a')['href']
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        title = article_soup.find('h1', class_='maintitle').text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        # 기사 본문을 찾습니다.
        body = [] ; bodys = str()
        # 조건에 맞는 요소를 찾는 함수
        def custom_filter(tag):
          # 'p' 태그는 클래스에 상관없이 모두 선택
          if tag.name == 'p':
            return True
          # 'div' 태그는 'ast-glance' 클래스만 선택
          if tag.name == 'div' and 'ast-glance' in tag.get('class', []):
            return True
          # 그 외의 경우는 선택하지 않음
          return False
        paragraphs = article_soup.find_all(custom_filter)
        for p in paragraphs: body.append(p.get_text().strip())
        for i in range(3,len(body)-2): bodys += str(body[i]).strip()
        if not bodys: error_message = Error_Message(error_message, "None Contents")
        if error_message != str():
          All_error_list.append({
            'Error Link': url_4,
            'Error': error_message
          })
        else:
          All_articles.append({
              'Title': title,
              'Link': article_link,
              'Content(RAW)': bodys
          })
          Govern_articles.append({
              'Title': title,
              'Link': link,
              'Content(RAW)': bodys
          })
except Exception as e:
  All_error_list.append({
      'Error Link': url_4,
      'Error': str(e)
      })
########################################### <5> ##############################################
# url_5 = 'https://www.army.mil/news'
wd = initialize_chrome_driver()
wd.get(url_5)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    # 뉴스 아이템 가져오기
    news_items = soup.find_all('div', class_='main-content')
    if not news_items:
        All_error_list.append({
            'Error Link': url_5,
            'Error': "None News"
        })
    else:
        for item in news_items:
            # 링크 추출 
            link_tag = item.find('p', class_='title').find('a', href=True)
            link = link_tag['href'] if link_tag else None
            link = 'https://www.army.mil' + link if link.startswith('/') else link
            if not link: error_message = Error_Message(error_message, "None Link")
            # 날짜 추출 
            date_tag = item.find('p', class_='date')
            date = date_tag.text.strip() if date_tag else None
            if not date_tag:
              error_message = Error_Message(error_message, "None Date")
            if article_date in today_list :
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()       
                    title = article.title
                    if not title:
                      error_message = Error_Message(error_message, "None Title")
                    text = article.text
                    if not link:
                      error_message = Error_Message(error_message, "None Content")
                    if error_message != str():
                            All_error_list.append({
                                'Error Link': url_5,
                                'Error': error_message
                            })
                    else:
                      All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
                      Govern_articles.append({
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
        'Error Link': url_5,
        'Error': str(e)
    })
########################################### <6> ##############################################
#url_6 = 'https://www.navy.mil/Press-Office/'
wd = initialize_chrome_driver()
wd.get(url_6)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
news_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  news_blocks = soup.find_all('div', class_='col-12 col-sm-8')
  if not news_blocks: All_error_list.append({'Error Link': url_6, 'Error': "None News"})
  else:
    for article in news_blocks:
      date_str = article.find('h6').text
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = article.find('h2').find('a')['href']
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        title = article.find('h2').get_text().strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        # 기사 본문을 찾습니다.
        body = [] ; bodys = str()
        paragraphs = article_soup.find('div', class_='acontent-container').find_all('p')
        for p in paragraphs: body.append(p.get_text().strip())
        for i in range(len(body)): bodys += str(body[i]).strip()
        if not bodys: error_message = Error_Message(error_message, "None Contents")
        if error_message != str():
          All_error_list.append({
            'Error Link': url_6,
            'Error': error_message
          })
        else:
          All_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
          Govern_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
except Exception as e:
  All_error_list.append({
      'Error Link': url_6,
      'Error': str(e)
      })
########################################### <7> ##############################################
# url_7 = 'https://www.af.mil/News/Category/22750/'
wd = initialize_chrome_driver()
wd.get(url_7)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try :
  #뉴스 아이템 가져오기
  news_items = soup.select('article[class^="article-listing-item"]')
  if not news_items:
    All_error_list.append({
        'Error Link': url_7,
        'Error': "None News"
        })
  else :
    for item in news_items :
      link = item.find('a')['href']
      if not link: error_message = Error_Message(error_message, "None Link")
      date_tag = item.find('time')
      if date_tag:
        date_string = date_tag.text.strip()
        article_date = date_util(date_string)
        if article_date in today_list:
          try:
              article = Article(link, language = 'en') # URL과 언어를 입력
              article.download()
              article.parse()
              title = article.title
              if not title: error_message = Error_Message(error_message, "None Title")
              content = article.text
              if not content: error_message = Error_Message(error_message, "None Contents")
              if error_message != str():
                        All_error_list.append({
                            'Error Link': url_7,
                            'Error': error_message
                            })
              else:
                        All_articles.append({
                            'Title': title,
                            'Link': link,
                            'Content(RAW)': content
                            })
                        Govern_articles.append({
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
        'Error Link': url_7,
        'Error': str(e)
    })
########################################### <8> ##############################################
# url_8 = 'https://www.nsa.gov/Press-Room/Press-Releases-Statements/'
wd = initialize_chrome_driver()
wd.get(url_8)
time.sleep(3)
html = wd.page_source
wd.quit()
soup = BeautifulSoup(html, 'html.parser')
try:
    # 뉴스 아이템 가져오기
    news_items = soup.find_all('div', class_='item')
    if not news_items:
      All_error_list.append({
          'Error Link': url_8,
          'Error': "None News"
          })
    else :
      for item in news_items :
        title_tag = item.find('div', class_='title').find('a')
        title = title_tag.text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        link = title_tag['href'].strip()
        if not link: error_message = Error_Message(error_message, "None Link")
        date_tag = item.find('span', class_='date')
        if date_tag:
          date_string = date_tag.text.strip()
          article_date = date_util(date_string)
          if article_date in today_list:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    text = article.text
                    if not link:
                      error_message = Error_Message(error_message, "None Content")
                    if error_message != str():
                            All_error_list.append({
                                'Error Link': url_8,
                                'Error': error_message
                            })
                    else:
                      All_articles.append({
                                'Title': title,
                                'Link': link,
                                'Content(RAW)': text
                            })
                      Govern_articles.append({
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
        'Error Link': url_8,
        'Error': str(e)
    })
########################################### <9> ##############################################
# url_9 = 'https://www.justice.gov/news'
wd = initialize_chrome_driver()
wd.get(url_9)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try :
  #뉴스 아이템 가져오기
  news_items = soup.find_all('div', class_='views-row')
  if not news_items:
    All_error_list.append({
        'Error Link': url_9,
        'Error': "None News"
        })
  else :
    for item in news_items :
      link_tag = item.find('a')
      if link_tag:
        relative_link = link_tag['href']
        # 상대 링크를 절대 링크로 변환
        base_url = url_9.rsplit('/', 1)[0]  # 'https://www.justice.gov'를 얻기 위해 /news 제거
        full_link = base_url + relative_link
      date_tag = item.find('time')
      if date_tag :
          date_string = date_tag.text.strip()
          article_date = date_util(date_string)
          if article_date in today_list:
            try:
                article = Article(full_link, language = 'en') # URL과 언어를 입력
                article.download()
                article.parse()
                title = article.title
                if not title: error_message = Error_Message(error_message, "None Title")
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                          All_error_list.append({
                              'Error Link': url_9,
                              'Error': error_message
                              })
                else:
                          All_articles.append({
                            'Title': title,
                            'Link': full_link,
                            'Content(RAW)': content
                            })
                          Govern_articles.append({
                            'Title': title,
                            'Link': full_link,
                            'Content(RAW)': content
                            })
            except Exception as e:
                All_error_list.append({
                    'Error Link': full_link,
                    'Error': str(e)
                })
except Exception as e:
    All_error_list.append({
        'Error Link': url_9,
        'Error': str(e)
    })
########################################### <10> ##############################################
# url_10 = 'https://www.fbi.gov/news/press-releases'
wd = initialize_chrome_driver()
wd.get(url_10)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
try:
  date_blocks = soup.find_all('p',class_='date')
  if not date_blocks: All_error_list.append({'Error Link': url_10, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list :
        article_link = block.find_parent().find_parent().find('a')['href']
        if not article_link:
                  error_message = Error_Message(error_message, "None Content")
        try:
            article = Article(article_link, language='en')
            article.download()
            article.parse()
            text = article.text
            if not text:
              error_message = Error_Message(error_message, "None Content")
              if error_message != str():
                      All_error_list.append({
                          'Error Link': url_10,
                          'Error': error_message
                      })
              else:
                      All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
                      Govern_articles.append({
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
        'Error Link': url_10,
        'Error': str(e)
    })
########################################### <11> ##############################################
#url_11 = 'https://www.doi.gov/news'    ### 유지보수중
wd = initialize_chrome_driver()
wd.get(url_11)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('div', class_='publication-date')
  if not date_blocks: All_error_list.append({'Error Link': url_11, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent()['href']
        if article_link[0] =='/': article_link = 'https://www.doi.gov' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        title = article_soup.find('h1',class_='section-page-title').text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        # 기사 본문을 찾습니다.
        body = [] ; bodys = str()
        body = article_soup.find('div',class_='field field--name-body field--type-text-with-summary field--label-hidden').text.strip()
        for i in range(len(body)): bodys += body[i]
        if not bodys: error_message = Error_Message(error_message, "None Contents")
        if error_message != str():
          All_error_list.append({
            'Error Link': url_11,
            'Error': error_message
          })
        else:
          All_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
          Govern_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
except Exception as e:
  All_error_list.append({
      'Error Link': url_11,
      'Error': str(e)
      })
########################################### <12> ##############################################
#url_12 = 'https://www.usda.gov/media/press-releases'
wd = initialize_chrome_driver()
wd.get(url_12)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
news_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  news_blocks = soup.find_all('li', class_='news-releases-item')
  if not news_blocks: All_error_list.append({'Error Link': url_12, 'Error': "None News"})
  else:
    for block in news_blocks:
      date_str = block.find('div',class_='news-release-date').text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.usda.gov' + article_link
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
                'Error Link': url_12,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_12,
      'Error': str(e)
      })
########################################### <13> ##############################################
# url_13 = 'https://www.ars.usda.gov/news-events/news-archive/'
wd = initialize_chrome_driver()
wd.get(url_13)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
base_url = 'https://www.ars.usda.gov'
try:
    # 뉴스 아이템 가져오기
    news_items = soup.find_all('a', id=lambda x: x and x.startswith('anch_'))
    if not news_items:
        All_error_list.append({
            'Error Link': url_13,
            'Error': "None News"
        })
    else:
        for item in news_items:
            error_message = ''
            title_link = item
            if title_link:
                title = title_link.text.strip()
                if not title:
                    error_message = Error_Message(error_message, "None Title")
                relative_link = item['href']
                if relative_link.startswith('/'):
                    full_link = base_url + relative_link
                else:
                    full_link = base_url + '/' + relative_link
                if not full_link:
                    error_message = Error_Message(error_message, "None Link")
                date_tag = item.find_next_sibling('td', class_='ars-tablecell-space')
                if date_tag:
                    date_string = date_tag.text.strip()
                    article_date = date_util(date_string)
                if article_date in today_list :
                  try:
                      article = Article(full_link, language='en')
                      article.download()
                      article.parse()
                      text = article.text
                      if not text:
                        error_message = Error_Message(error_message, "None Content")
                      if error_message != str():
                            All_error_list.append({
                                'Error Link': url_13,
                                'Error': error_message
                            })
                      else:
                        All_articles.append({
                                  'Title': title,
                                  'Link': full_link,
                                  'Content(RAW)': text
                              })
                        Govern_articles.append({
                                  'Title': title,
                                  'Link': full_link,
                                  'Content(RAW)': text
                              })
                  except Exception as e:
                      All_error_list.append({
                          'Error Link': full_link,
                          'Error': str(e)
                      })
except Exception as e:
    All_error_list.append({
        'Error Link': url_13,
        'Error': str(e)
    })
########################################### <14> ##############################################
# url_14 = 'https://www.fs.usda.gov/news/releases'
wd = initialize_chrome_driver()
wd.get(url_14)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
     news_items = soup.find_all('div', class_='margin-bottom-2 views-row')
     if not news_items:
         All_error_list.append({
             'Error Link': url_14,
             'Error': "None News"
         })
     else:
         for item in news_items:
             date_tag = item.find('time', class_='text-base-darker')
             date_str = date_tag.get_text(strip=True)
             article_date = date_util(date_str)
             if not article_date:
                 error_message = Error_Message(error_message, "None Date")
             if article_date in today_list:
                 article_link = f"https://www.fs.usda.gov{item.a['href']}"
                 if not article_link:
                     error_message = Error_Message(error_message, "None Link")
                 try:
                     article = Article(article_link, language = 'en') # URL과 언어를 입력
                     article.download()
                     article.parse()
                     title = article.title
                     if not title: error_message = Error_Message(error_message, "None Title")
                     content = article.text
                     if not content: error_message = Error_Message(error_message, "None Contents")
                     if error_message != "":
                         All_error_list.append({
                             'Error Link': url_14,
                             'Error': error_message
                         })
                     else:
                         All_articles.append({
                             'Title': title,
                             'Link': article_link,
                             'Content(RAW)': content
                         })
                         Govern_articles.append({
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
         'Error Link': url_14,
         'Error': str(e)
     })
########################################### <15> ##############################################
# url_15 = 'https://www.fas.usda.gov/newsroom/search'
wd = initialize_chrome_driver()
wd.get(url_15)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
base_url = 'https://fas.usda.gov'
try:
    # 뉴스 아이템 가져오기
    news_items = soup.find_all('div', class_='card__header')
    if not news_items:
        All_error_list.append({
            'Error Link': url_15,
            'Error': "None News"
        })
    else:
        for item in news_items:
            error_message = ''
            link_tag = item.find('a', class_='card__url')
            if link_tag:
                title = link_tag.text.strip()
                if not title:
                    error_message = Error_Message(error_message, "None Title")
                relative_link = link_tag['href']
                full_link = base_url + relative_link
                if not full_link:
                    error_message = Error_Message(error_message, "None Link")
                date_tag = item.find('time')
                if date_tag:
                    date_time_str = date_tag['datetime']
                    date_str = date_time_str.split('T')[0]
                    article_date = date_util(date_str)
                    if article_date in today_list :
                      try:
                          article = Article(full_link, language='en')
                          article.download()
                          article.parse()
                          text = article.text
                          if not text:
                            error_message = Error_Message(error_message, "None Content")
                          if error_message != str():
                                  All_error_list.append({
                                      'Error Link': url_15,
                                      'Error': error_message
                                  })
                          else:
                              All_articles.append({
                                      'Title': title,
                                      'Link': full_link,
                                      'Content(RAW)': text
                                  })
                              Govern_articles.append({
                                      'Title': title,
                                      'Link': full_link,
                                      'Content(RAW)': text
                                  })
                      except Exception as e:
                          All_error_list.append({
                              'Error Link': full_link,
                              'Error': str(e)
                          })
except Exception as e:
    All_error_list.append({
        'Error Link': url_15,
        'Error': str(e)
    })
########################################### <16> ##############################################
# url_16 = 'https://www.commerce.gov/news'
wd = initialize_chrome_driver()
wd.get(url_16)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    # 뉴스 아이템 가져오기
    news_items = soup.find_all('div', class_='views-row')
    if not news_items:
        All_error_list.append({
            'Error Link': url_16,
            'Error': "None News"
        })
    else:
        for item in news_items:
            error_message = ''
            date_tag = item.find('time', class_='datetime')
            date_string = date_tag['datetime'].split('T')[0]  # 'YYYY-MM-DD' 형식으로 추출
            article_date = date_util(date_string)
            if not article_date: error_message = Error_Message(error_message, "None Date")
            if article_date in today_list:
                link_tag = item.find('a', href=True)
                link = link_tag['href'] if link_tag else '#'
                full_link = f"https://www.commerce.gov{link}"  # 절대 URL 생성
                if not full_link:
                    error_message = Error_Message(error_message, "None Link")
                try:
                    article = Article(full_link, language = 'en') # URL과 언어를 입력
                    article.download()
                    article.parse()
                    title = article.title
                    if not title: error_message = Error_Message(error_message, "None Title")
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                    if error_message != str():
                        All_error_list.append({
                            'Error Link': url_16,
                            'Error': error_message
                        })
                    else:
                        All_articles.append({
                            'Title': title,
                            'Link': full_link,
                            'Content(RAW)': content
                        })
                        Govern_articles.append({
                            'Title': title,
                            'Link': full_link,
                            'Content(RAW)': content
                        })
                except Exception as e:
                    All_error_list.append({
                        'Error Link': full_link,
                        'Error': str(e)
                    })
except Exception as e:
    All_error_list.append({
        'Error Link': url_16,
        'Error': str(e)
    })
########################################### <17> ##############################################
# url_17 = 'https://www.dol.gov/newsroom/releases?agency=All&state=All&topic=All&year=all&page=0'
wd = initialize_chrome_driver()
wd.get(url_17)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('p', class_='dol-date-text')
  if not date_blocks: All_error_list.append({'Error Link': url_17, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list :
        article_link = block.find_parent().find('a')['href']
        if article_link[0] ==' ': article_link = ('https://www.dol.gov' + article_link[4:]).strip()
        if article_link.endswith('s') : article_link = article_link[:-1]
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
                                'Error Link': url_17,
                                'Error': error_message
                            })        
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              Govern_articles.append({
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
        'Error Link': url_17,
        'Error': str(e)
    })
########################################### <18> ##############################################
#url_18 = 'https://www.hhs.gov/about/news/index.html'
wd = initialize_chrome_driver()
wd.get(url_18)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('time')
  if not date_blocks: All_error_list.append({'Error Link': url_18, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.hhs.gov' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        title = article_soup.find('h1').text
        if not title: error_message = Error_Message(error_message, "None Title")
        # 기사 본문을 찾습니다.
        body = [] ; bodys = str()
        bodys = article_soup.find('div',class_='field__item usa-prose').text.strip()
        if not bodys: error_message = Error_Message(error_message, "None Contents")
        if error_message is not str():
          All_error_list.append({
            'Error Link': url_18,
            'Error': error_message
          })
        else:
          All_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
          Govern_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
except Exception as e:
  All_error_list.append({
      'Error Link': url_18,
      'Error': str(e)
      })
########################################### <19> ##############################################
#url_19 = 'https://www.fda.gov/news-events/fda-newsroom/press-announcements'
wd = initialize_chrome_driver()
wd.get(url_19)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('time')
  if not date_blocks: All_error_list.append({'Error Link': url_19, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      if block.find_parent().get('class') == None:
        date_str = block.text.strip()
        article_date = date_util(date_str)
        article_link, title, bodys = None, None, None
        if article_date in today_list:
          article_link = block.find_parent().find_parent().find('a')['href']
          if article_link[0] =='/': article_link = 'https://www.fda.gov' + article_link
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
                  'Error Link': url_19,
                  'Error': error_message
                })
              else:
                All_articles.append({
                  'Title': title,
                  'Link': article_link,
                  'Content(RAW)': content
                })
                Govern_articles.append({
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
      'Error Link': url_19,
      'Error': str(e)
      })
########################################### <20> ##############################################
#url_20 = 'https://www.fda.gov/news-events/speeches-fda-officials'
wd = initialize_chrome_driver()
wd.get(url_20)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('time')
  if not date_blocks: All_error_list.append({'Error Link': url_20, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.fda.gov' + article_link
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
                'Error Link': url_20,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_20,
      'Error': str(e)
      })
########################################### <21> ##############################################
# url_21 = 'https://www.transportation.gov/newsroom/press-releases'
wd = initialize_chrome_driver()
wd.get(url_21)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('time', class_='datetime')
  if not date_blocks: All_error_list.append({'Error Link': url_21, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list :
        article_link = block.find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.transportation.gov' + article_link
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
                                'Error Link': url_21,
                                'Error': error_message
                            })          
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              Govern_articles.append({
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
        'Error Link': url_21,
        'Error': str(e)
    })
########################################### <22> ##############################################
# url_22 = 'https://www.transportation.gov/newsroom/speeches'
wd = initialize_chrome_driver()
wd.get(url_22)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try :
  date_blocks = soup.find_all('td', class_='views-field views-field-field-effective-date')
  if not date_blocks: All_error_list.append({'Error Link': url_22, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list :
        article_link = block.find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.transportation.gov' + article_link
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
                                'Error Link': url_22,
                                'Error': error_message
                            })       
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              Govern_articles.append({
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
        'Error Link': url_22,
        'Error': str(e)
    })
########################################### <23> ##############################################
#url_23 = 'https://www.energy.gov/newsroom'
wd = initialize_chrome_driver()
wd.get(url_23)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('div', class_='search-result-display-date')
  if not date_blocks: All_error_list.append({'Error Link': url_23, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.find('p').text
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find('a', class_="search-result-title")['href']
        if article_link[0] =='/': article_link = 'https://www.energy.gov' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        title = article_soup.find('h1', class_='page-title').text.replace('\n','').strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        # 기사 본문을 찾습니다.
        body = [] ; bodys = str()
        bodys = article_soup.find('div',class_='block block-system block-system-main-block').text.strip()
        if not bodys: error_message = Error_Message(error_message, "None Contents")
        if error_message != str():
          All_error_list.append({
            'Error Link': url_23,
            'Error': error_message
          })
        else:
          All_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
          Govern_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
except Exception as e:
  All_error_list.append({
      'Error Link': url_23,
      'Error': str(e)
      })
########################################### <24> ##############################################
# url_24 = 'https://www.ed.gov/news/press-releases'
wd = initialize_chrome_driver()
wd.get(url_24)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try :
  date_blocks = soup.find_all('span', class_='field-content date-display-single')
  if not date_blocks: All_error_list.append({'Error Link': url_24, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.ed.gov' + article_link
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
                                'Error Link': url_24,
                                'Error': error_message
                            })        
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              Govern_articles.append({
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
        'Error Link': url_24,
        'Error': str(e)
    })
########################################### <25> ##############################################
# url_25 = 'https://www.ed.gov/news/speeches'
wd = initialize_chrome_driver()
wd.get(url_25)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('span', class_='field-content date-display-single')
  if not date_blocks: All_error_list.append({'Error Link': url_25, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.text
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.ed.gov' + article_link
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
                                'Error Link': url_25,
                                'Error': error_message
                            })          
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              Govern_articles.append({
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
        'Error Link': url_25,
        'Error': str(e)
    })
########################################### <26> ##############################################
#url_26 = 'https://news.va.gov/news/'
wd = initialize_chrome_driver()
wd.get(url_26)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
article_links, article_date, article_link, title, bodys = None, None, None, None, None
try:
  article_links = soup.find_all('a', class_='awb-custom-text-color awb-custom-text-hover-color')
  if not article_links: All_error_list.append({'Error Link': url_26, 'Error': "None Links"})
  else:
    links = [link.get('href') for link in article_links]
    for link in links:
      try:
          article = Article(link, language = 'en') # URL과 언어를 입력
          article.download()
          article.parse()
          article_date = (article.publish_date).date()
          if not article_date: error_message = Error_Message(error_message, "None Date")
          if article_date in today_list:
            title = article.title
            if not title: error_message = Error_Message(error_message, "None Title")
            content = article.text
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
              All_error_list.append({
                'Error Link': url_26,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_26,
      'Error': str(e)
      })
########################################### <27> ##############################################
#url_27 = 'https://www.dhs.gov/news-releases/press-releases'
wd = initialize_chrome_driver()
wd.get(url_27)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('time')
  if not date_blocks: All_error_list.append({'Error Link': url_27, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.get('datetime')
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.dhs.gov' + article_link
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
                'Error Link': url_27,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_27,
      'Error': str(e)
      })
########################################### <28> ##############################################
#url_28 = 'https://www.dhs.gov/news-releases/speeches'
wd = initialize_chrome_driver()
wd.get(url_28)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('time')
  if not date_blocks: All_error_list.append({'Error Link': url_28, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.get('datetime')
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.dhs.gov' + article_link
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
                'Error Link': url_28,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_28,
      'Error': str(e)
      })
########################################### <29> ##############################################
#url_29 = 'https://www.fema.gov/about/news-multimedia/press-releases'
wd = initialize_chrome_driver()
wd.get(url_29)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('div', class_='views-listing views-row')
  if not date_blocks: All_error_list.append({'Error Link': url_29, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block.find('time').text
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.fema.gov' + article_link
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
                'Error Link': url_29,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_29,
      'Error': str(e)
      })
########################################### <30> ##############################################
# url_30 = 'https://www.secretservice.gov/newsroom'
wd = initialize_chrome_driver()
wd.get(url_30)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('div', class_='field-content usss-news-blk-date-y')
  if not date_blocks: All_error_list.append({'Error Link': url_30, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      day = block.find('p', class_='usss-news-date-d').text
      month = block.find('p', class_='usss-news-date-m').text
      year = block.find('p', class_='usss-news-date-y').text
      date_str = f"{month} {day}, {year}"
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.secretservice.gov' + article_link
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
                                'Error Link': url_30,
                                'Error': error_message
                            })     
            else:
              All_articles.append({
                                'Title': title,
                                'Link': article_link,
                                'Content(RAW)': text
                            })
              Govern_articles.append({
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
        'Error Link': url_30,
        'Error': str(e)
    })
########################################### <31> ##############################################
#url_31 = 'https://www.epa.gov/newsreleases/search'
wd = initialize_chrome_driver()
wd.get(url_31)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find_all('time')
  if not date_blocks:
    All_error_list.append({
            'Error Link': url_31,
            'Error': "Date Blocks"
          })
  else:
    for block in date_blocks:
      date_str = block.text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find('a', class_='usa-link')['href']
        if article_link[0] =='/': article_link = 'https://www.epa.gov' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(3)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        title = article_soup.find('h1', class_='page-title').text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        body = []
        paragraphs = article_soup.find_all('p')
        for p in paragraphs:
          if p and not p.get('class'):  
            body.append(p.get_text())
        bodys = ''.join(body[4:])
        if not bodys: error_message = Error_Message(error_message, "None Contents")
        if error_message != str():
          All_error_list.append({
            'Error Link': url_31,
            'Error': error_message
          })
        else:
          All_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
          Govern_articles.append({
            'Title': title,
            'Link': article_link,
            'Content(RAW)': bodys
          })
except Exception as e:
  error_list.append({
      'Error Link': url_31,
      'Error': str(e)
      })
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
########################################### <41> ##############################################
#url_41 = 'https://sos.ga.gov/news/division/31?page=0'
wd = initialize_chrome_driver()
wd.get(url_41)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
     news_items = soup.select(".card__content")
     if not news_items:
         All_error_list.append({
             'Error Link': url_41,
             'Error': "None News"
         })
     else:
         for item in news_items:
             date_str = item.select_one('.card__date').get_text(strip=True)
             article_date = date_util(date_str)
             if not article_date:
                 error_message = Error_Message(error_message, "None Date")
             if article_date in today_list :
                 title = item.select_one('.heading__link').get_text(strip=True)
                 if not title:
                     error_message = Error_Message(error_message, "None Title")
                 article_link = f"https://sos.ga.gov{item.a['href']}"
                 if not article_link: error_message = Error_Message(error_message, "None Link")
                 try:
                     article = Article(article_link, language='en')
                     article.download()
                     article.parse()
                     text = article.text
                     if not text:
                       error_message = Error_Message(error_message, "None Content")
                     if error_message != str():
                                     All_error_list.append({
                                         'Error Link': url_41,
                                         'Error': error_message
                                     })     
                     else:
                         All_articles.append({
                                         'Title': title,
                                         'Link': article_link,
                                         'Content(RAW)': text
                                     })
                         Govern_articles.append({
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
        'Error Link': url_41,
        'Error': str(e)
    })
########################################### <42> ##############################################
#url_42 = 'https://gov.georgia.gov/press-releases'
wd = initialize_chrome_driver()
wd.get(url_42)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
     news_items = soup.select(".news-teaser")
     if not news_items:
         All_error_list.append({
             'Error Link': url_42,
             'Error': "None News"
         })
     else:
         for item in news_items:
             date_str = item.select_one('.global-teaser__description').get_text(strip=True)
             article_date = date_util(date_str)
             if not article_date:
                 error_message = Error_Message(error_message, "None Date")
             if article_date in today_list:
                 article_link = f"https://gov.georgia.gov{item.a['href']}"
                 if not article_link: error_message = Error_Message(error_message, "None Link")
                 try:
                     article = Article(article_link, language = 'en') # URL과 언어를 입력
                     article.download()
                     article.parse()
                     title = article.title
                     if not title: error_message = Error_Message(error_message, "None Title")
                     content = article.text
                     if not content: error_message = Error_Message(error_message, "None Contents")
                     if error_message != "":
                         All_error_list.append({
                             'Error Link': url_42,
                             'Error': error_message
                         })
                     else:
                         All_articles.append({
                             'Title': title,
                             'Link': article_link,
                             'Content(RAW)': content
                         })
                         Govern_articles.append({
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
         'Error Link': url_42,
         'Error': str(e)
     })
########################################### <44> ##############################################
# url_44 = 'https://www.georgia.org/press-releases' 
wd = initialize_chrome_driver()
wd.get(url_44)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
     news_items = soup.select("[class=info]")
     if not news_items:
         All_error_list.append({
             'Error Link': url_44,
             'Error': "None News"
         })
     else:
         for item in news_items:
             date_str = item.select_one('.date').get_text(strip=True)
             article_date = date_util(date_str)
             if not article_date:
                 error_message = Error_Message(error_message, "None Date")
             if article_date in today_list :
                 a_tag = item.find('a')
                 if not a_tag: error_message = Error_Message(error_message, "a_tag 못찾음")
                 all_text = a_tag.get_text(strip=True)
                 not_title = a_tag.find('span', class_='date').get_text(strip=True)
                 title = all_text.replace(not_title, '').strip('" ')
                 if not title:
                     error_message = Error_Message(error_message, "None Title")
                 article_link = f"https://www.georgia.org{item.a['href']}"
                 if not article_link: error_message = Error_Message(error_message, "None Link")
                 try:
                     article = Article(article_link, language='en')
                     article.download()
                     article.parse()  
                     text = article.text
                     if not text:
                       error_message = Error_Message(error_message, "None Content")
                     if error_message != str():
                                     All_error_list.append({
                                         'Error Link': url_44,
                                         'Error': error_message
                                     })     
                     else:
                       All_articles.append({
                                         'Title': title,
                                         'Link': article_link,
                                         'Content(RAW)': text
                                     })
                       Govern_articles.append({
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
        'Error Link': url_44,
        'Error': str(e)
    })
########################################### <45> ##############################################
#url_45 = 'https://www.gachamber.com/all-news/'
wd = initialize_chrome_driver()
wd.get(url_45)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
     news_items = soup.select(".fl-post-feed-post")
     if not news_items:
         All_error_list.append({
             'Error Link': url_45,
             'Error': "None News"
         })
     else:
         for item in news_items:
             meta_tag = item.find('div', class_='fl-post-meta')
             parts = meta_tag.get_text(strip=True).split('·')
             date_str = parts[-1].strip()
             article_date = date_util(date_str)
             if not article_date:
                 error_message = Error_Message(error_message, "None Date")
             if article_date in today_list:
                 title = item.select_one('a').get_text(strip=True)
                 if not title:
                     error_message = Error_Message(error_message, "None Title")
                 article_link = item.find('h2').find('a')['href']
                 if not article_link:
                     error_message = Error_Message(error_message, "None Link")
                 wd = initialize_chrome_driver()
                 wd.get(article_link)
                 article_html = wd.page_source
                 article_soup = BeautifulSoup(article_html, 'html.parser')
                 parent_div = article_soup.find('div', class_="fl-module fl-module-fl-post-content fl-node-5cb515ec1b22a")
                 paragraphs = parent_div.select(".fl-module-content.fl-node-content p") #if parent_div else []
                 article_body = ' '.join(p.get_text(strip=True) for p in paragraphs)
                 if not article_body:
                     error_message = Error_Message(error_message, "None Contents")
                 if error_message != "":
                     All_error_list.append({
                         'Error Link': url_45,
                         'Error': error_message
                     })
                 else:
                     All_articles.append({
                         'Title': title,
                         'Link': article_link,
                         'Content(RAW)': article_body
                     })
                     Govern_articles.append({
                         'Title': title,
                         'Link': article_link,
                         'Content(RAW)': article_body
                     })
except Exception as e:
     All_error_list.append({
         'Error Link': url_45,
         'Error': str(e)
     })
########################################### <46> ##############################################
# url_46 = 'https://www.sos.ca.gov/administration/news-releases-and-advisories/2023-news-releases-and-advisories'
wd = initialize_chrome_driver()
wd.get(url_46)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('tr')
    if not news_items: All_error_list.append({'Error Link': url_46, 'Error': "Entire Error1"})
    for item in news_items[1:]:  
        td_tags = item.find_all('td')
        if not td_tags: error_list.append({'Error Link': url_46, 'Error': "Entire Error2"})
        if len(td_tags) > 1:
            date_text = td_tags[1].get_text(strip=True)
            article_date = date_util(date_text)
            if not article_date: error_message = Error_Message(error_message, "None Date")
            if article_date in today_list:
              a_tag = td_tags[1].find('a')
              link = a_tag['href']
              if not link: error_message = Error_Message(error_message, "None Link")
              try:
                  article = Article(link, language='en')
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
                                      'Error Link': url_46,
                                      'Error': error_message
                                  })     
                  else:
                    All_articles.append({
                                      'Title': title,
                                      'Link': link,
                                      'Content(RAW)': text
                                  })
                    Govern_articles.append({
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
        'Error Link': url_46,
        'Error': str(e)
    })
########################################### <48> ##############################################
# url_48 = 'https://business.ca.gov/newsroom/'
wd = initialize_chrome_driver()
wd.get(url_48)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    date_span_list = soup.find_all('span', class_='date')
    if not date_span_list: All_error_list.append({'Error Link': url_48, 'Error': "Entire Error"})
    for date_span in date_span_list:
        date_text = date_span.get_text(strip=True)
        if not date_text: error_message = error_list.append({'Error Link': url_48, 'Error': "None Date"})
        article_date = date_util(date_text)
        if article_date in today_list:
            news_items = soup.find_all('li', class_= 'listing-item')
            if not news_items: error_message = error_list.append({'Error Link': url_48, 'Error': "None news_items"})
            for item in news_items:
                soup = BeautifulSoup(str(item), 'html.parser')
                a_tag = soup.find('a')
                link = a_tag['href']
                if not link: error_message = Error_Message(error_message, "None link")
                try:
                    article = Article(link, language='en')
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
                                        'Error Link': url_48,
                                        'Error': error_message
                                    })     
                    else:
                      All_articles.append({
                                        'Title': title,
                                        'Link': link,
                                        'Content(RAW)': text
                                    })
                      Govern_articles.append({
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
        'Error Link': url_48,
        'Error': str(e)
    })
########################################### <49> ##############################################
# url_49 = 'https://business.ca.gov/calosba-latest-news/'
wd = initialize_chrome_driver()
wd.get(url_49)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    date_span_list = soup.find_all('span', class_='date')
    if not date_span_list: All_error_list.append({'Error Link': url_49, 'Error': "Entire Error"})
    for date_span in date_span_list:
        date_text = date_span.get_text(strip=True)
        if not date_text: error_message = error_list.append({'Error Link': url_49, 'Error': "None Date"})
        article_date = date_util(date_text)
        if article_date in today_list :
            news_items = soup.find_all('li', class_= 'listing-item')
            if not news_items: error_message = error_list.append({'Error Link': url_49, 'Error': "None news_items"})
            for item in news_items:
                soup = BeautifulSoup(str(item), 'html.parser')
                a_tag = soup.find('a')
                link = a_tag['href']
                if not link: error_message = Error_Message(error_message, "None link")
                try:
                    article = Article(link, language='en')
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
                                        'Error Link': url_49,
                                        'Error': error_message
                                    })     
                    else:
                      All_articles.append({
                                        'Title': title,
                                        'Link': link,
                                        'Content(RAW)': text
                                    })
                      Govern_articles.append({
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
        'Error Link': url_49,
        'Error': str(e)
    })
########################################### <50> ##############################################
#url_50 = 'https://www.dir.ca.gov/dlse/DLSE_whatsnew2023.htm'
wd.get(url_50)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    tbody = soup.find('tbody', class_='latest_news')
    selected_rows = []
    for tr in tbody.find_all('tr'):
        if tr.find('td', class_='nowrap'):
            selected_rows.append(tr)
    if not selected_rows: All_error_list.append({'Error Link': url_50, 'Error': "Entire Error4"})
    for row in selected_rows:
        td_tags = row.find('td', class_ = "nowrap")
        date_text = td_tags.get_text(strip=True)
        article_date = date_util(date_text)
        if article_date in today_list:
            a_tag = td_tags.find_parent().find('a')
            link = f"https://www.dir.ca.gov{a_tag['href']}"
            if not link: error_message = Error_Message(error_message, "None link")
            try:
                article = Article(link, language = 'en') # URL과 언어를 입력
                article.download()
                article.parse()
                title = article.title
                if not title: error_message = Error_Message(error_message, "None Title")
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                    All_error_list.append({
                    'Error Link': url_50,
                    'Error': error_message
                    })
                else:
                    All_articles.append({
                        'Title': title,
                        'Link': link,
                        'Content(RAW)': content
                        })
                    Govern_articles.append({
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
     'Error Link': url_50,
     'Error': str(e)
     })
########################################### <51> ##############################################
# url_51 = 'https://www.dir.ca.gov/dosh/DOSH_Archive.html'
wd = initialize_chrome_driver()
wd.get(url_51)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    soup = soup.find('tbody')
    if not soup: error_list.append({'Error Link': url_51, 'Error': "Entire Error1"})
    if not soup.find_all('tr'): All_error_list.append({'Error Link': url_51, 'Error': "Entire Error2"})
    for item in soup.find_all('tr'):
        td_tags = item.find_all('td')
        if td_tags:
            article_date = date_util(td_tags[0].text)
            if article_date in today_list :
                a_tag = item.find('a')
                if not a_tag: error_list.append({'Error Link': url_51, 'Error': "Entire Error3"})
                if a_tag and 'href' in a_tag.attrs:
                    link = "https://www.dir.ca.gov" + a_tag.attrs['href']
                    if not link: error_message = Error_Message(error_message, "None link")
                    try:
                        article = Article(link, language='en')
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
                                            'Error Link': url_51,
                                            'Error': error_message
                                        })     
                        else:
                          All_articles.append({
                                            'Title': title,
                                            'Link': link,
                                            'Content(RAW)': text
                                        })
                          Govern_articles.append({
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
        'Error Link': url_51,
        'Error': str(e)
    })
########################################### <52> ##############################################
#url_52 = 'https://www.dir.ca.gov/mediaroom.html'
wd = initialize_chrome_driver()
wd.get(url_52)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    selected_rows = []
    for tr in soup.find_all('tr'):
          if tr.find('td', class_='nowrap'):
              selected_rows.append(tr)
    if not selected_rows: All_error_list.append({'Error Link': url_52, 'Error': "Entire Error1"})
    for item in selected_rows:
        td_tags = item.find_all('td')
        if not td_tags: error_list.append({'Error Link': url_52, 'Error': "Entire Error2"})
        if td_tags:
            article_date = date_util(td_tags[0].text)
            if article_date in today_list:
                a_tag = item.find('a')
                title = a_tag.attrs.get('aria-label').strip()
                if not title: error_message = Error_Message(error_message, "None title")
                link = "https://www.dir.ca.gov" + a_tag.attrs['href']
                if not link: error_message = Error_Message(error_message, "None link")
                if '.pdf' in link: content = 'pdf file'
                else:
                      try:
                        article = Article(link, language = 'en') # URL과 언어를 입력
                        article.download()
                        article.parse()
                        content = article.text
                        if not content: error_message = Error_Message(error_message, "None Contents")
                      except Exception as e:
                        All_error_list.append({
                         'Error Link': link,
                         'Error': str(e)
                         })
                if error_message != str():
                    All_error_list.append({
                    'Error Link': url_52,
                    'Error': error_message
                    })
                else:
                    All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                    Govern_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
except Exception as e:
    All_error_list.append({
     'Error Link': url_52,
     'Error': str(e)
     })
########################################### <53> ##############################################
# url_53 = 'https://news.caloes.ca.gov/'
wd = initialize_chrome_driver()
wd.get(url_53)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('article')
    if not news_items: All_error_list.append({'Error Link': url_53, 'Error': "Entire Error1"})
    for article in news_items:
        link = article.find('a', class_='entry-featured-image-url')['href']
        if not link: error_message = Error_Message(error_message, "None link")
        date = article.find('span', class_='published').get_text(strip=True)
        if not date: error_message = Error_Message(error_message, "None date")
        author = article.find('a', rel='author').get_text(strip=True)
        article_date = date_util(date)
        if article_date in today_list :
          try:
              article = Article(link, language='en')
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
                    'Error Link': url_53,
                    'Error': error_message
                    })     
              else:
                All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': text
                    })
                Govern_articles.append({
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
        'Error Link': url_53,
        'Error': str(e)
    })
########################################### <54> ##############################################
# url_54 = 'https://advocacy.calchamber.com/california-works/calchamber-members-in-the-news/'
wd = initialize_chrome_driver()
wd.get(url_54)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    first_h2 = soup.find('h2')
    if not first_h2: All_error_list.append({'Error Link': url_54, 'Error': "Entire Error1"})
    else:
        first_ul_after_h2 = first_h2.find_next('ul')
        if not first_ul_after_h2: All_error_list.append({'Error Link': url_54, 'Error': "Entire Error2"})
        else: 
            news_items = first_ul_after_h2.find_all('li')
            if not news_items: All_error_list.append({'Error Link': url_54, 'Error': "Entire Error3"})
    for article in news_items: 
        em_tag = article.find('em')
        if not em_tag: error_message = Error_Message(error_message, "Entire Error4")
        em_text = em_tag.get_text().strip()
        newspaper, date_str = em_text.split(',', 1)  
        date_str = date_str.strip() 
        date = date_util(date_str)
        if date in today_list:
            link = article.a['href']
            if not link: error_message = Error_Message(error_message, "None link")
            try:
                article = Article(link, language='en')
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
                      'Error Link': url_54,
                      'Error': error_message
                      })     
                else:
                  All_articles.append({
                      'Title': title,
                      'Link': link,
                      'Content(RAW)': text
                      })
                  Govern_articles.append({
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
        'Error Link': url_54,
        'Error': str(e)
    })
########################################### <55> ##############################################
# url_55 = 'https://gov.texas.gov/news'
wd = initialize_chrome_driver()
wd.get(url_55)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    h3_tags = soup.find_all('h3', class_='h2')
    if not h3_tags: All_error_list.append({'Error Link': url_55, 'Error': "Entire Error1"})
    else:
        month_s = soup.find_all('span', class_='date-month')
        if not month_s: All_error_list.append({'Error Link': url_55, 'Error': "Entire Error2"})
        else:
          day_s= soup.find_all('span', class_='date-day')
          if not day_s: All_error_list.append({'Error Link': url_55, 'Error': "Entire Error3"})
    i = 0
    for month in month_s:
        date_str = f"{month.text} {day_s[i].text}"
        article_date = date_util(date_str)
        i+=1
        if article_date in today_list:
            link = h3_tags[i-1].a['href']
            if not link: error_message = Error_Message(error_message, "None link")
            try:
                article = Article(link, language='en')
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
                      'Error Link': url_55,
                      'Error': error_message
                      })     
                else:
                  All_articles.append({
                      'Title': title,
                      'Link': link,
                      'Content(RAW)': text
                      })
                  Govern_articles.append({
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
        'Error Link': url_55,
        'Error': str(e)
    })
########################################### <56> ##############################################
#url_56 = 'https://www.texasattorneygeneral.gov/news/releases'
wd = initialize_chrome_driver()
wd.get(url_56)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('h4', class_='h4-sans')
    if not news_items:
        All_error_list.append({'Error Link': url_56, 'Error': "Entire Error1"})
    for article in news_items:
        date_str = article.find_next_sibling('p', class_='meta').get_text()
        if not date_str:
            error_message = Error_Message(error_message, "None date")
        article_date = date_util(date_str.split('|')[0].strip())
        # print(article_date)
        if article_date in today_list:
            a_tag = article.find('a')
            if not a_tag:
                error_message = Error_Message(error_message, "None title")
            title = a_tag.text.strip()
            link = 'https://www.texasattorneygeneral.gov' + a_tag['href']
            if not link: error_message = Error_Message(error_message, "None link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                  All_error_list.append({
                      'Error Link': url_56,
                      'Error': error_message
                      }) 
                else:
                  All_articles.append({
                      'Title': title,
                      'Link': link,
                      'Content(RAW)': content
                      })
                  Govern_articles.append({
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
     'Error Link': url_56,
     'Error': str(e)
     })
########################################### <57> ##############################################
#url_57 = "https://www.txdot.gov/about/newsroom/statewide.html"
wd = initialize_chrome_driver()
wd.get(url_57)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('tr')
    if not news_items:
        All_error_list.append({'Error Link': url_57, 'Error': "Entire Error1"})
    for item in news_items:
        date_tags = item.find_all('td')
        if len(date_tags) < 2:
          continue
        date_str = date_tags[1].get_text(strip=True)
        # print(date_str)
        if not date_str:
            error_message = Error_Message(error_message, "None date")
        article_date = date_util(date_str)
        # print(article_date)
        if article_date in today_list:
            a_tag = item.find('a')
            if not a_tag:
                error_message = Error_Message(error_message, "None title")
            title = a_tag.text.strip()
            link = 'https://www.txdot.gov' + a_tag['href']
            if not link: error_message = Error_Message(error_message, "None link")
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                  All_error_list.append({
                      'Error Link': url_57,
                      'Error': error_message
                      }) 
                else:
                  All_articles.append({
                      'Title': title,
                      'Link': link,
                      'Content(RAW)': content
                      })
                  Govern_articles.append({
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
     'Error Link': url_57,
     'Error': str(e)
     })
########################################### <58> ##############################################
#url_58 = 'https://www.dps.texas.gov/news/press-releases'
wd = initialize_chrome_driver()
wd.get(url_58)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    soup = soup.find('div', class_ = 'item-list')
    if not soup: All_error_list.append({'Error Link': url_58, 'Error': "Entire Error1"})
    news_items = soup.find_all('li')
    if not news_items: All_error_list.append({'Error Link': url_58, 'Error': "Entire Error2"})
    for item in news_items:
        date = item.select_one('.views-field-created').get_text(strip=True)
        if not date: error_message = Error_Message(error_message, "None date")
        news_date = date_util(date)
        if news_date in today_list:
          title = item.select_one('.views-field-title a').get_text(strip=True)
          if not title: error_message = Error_Message(error_message, "None title")
          alink = item.select_one('.views-field-title a')['href']
          link = f"https://www.dps.texas.gov{alink}"
          if not link: error_message = Error_Message(error_message, "None link")
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              content = article.text
              if not content: error_message = Error_Message(error_message, "None Contents")
              if error_message != str():
                All_error_list.append({
                    'Error Link': url_58,
                    'Error': error_message
                    }) 
              else:
                All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                Govern_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
          except Exception as e: # 코드상 에러가 생김
              All_error_list.append({
               'Error Link': link,
               'Error': str(e)
               })
except Exception as e: # 코드상 에러가 생김
    All_error_list.append({
     'Error Link': url_58,
     'Error': str(e)
     })
########################################### <59> ##############################################
#url_59 = 'https://www.twc.texas.gov/news'
wd = initialize_chrome_driver()
wd.get(url_59)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('div', class_='views-row')
    if not news_items: All_error_list.append({'Error Link': url_59, 'Error': "Entire Error"})
    for item in news_items:
        date = item.find('div', class_='news-date').get_text(strip=True)
        if not date: error_message = Error_Message(error_message, "None date")
        news_date = date_util(date)
        if news_date in today_list:
          title = item.find('div', class_='news-title').get_text(strip=True)
          if not title: error_message = Error_Message(error_message, "None title")
          link = 'https://www.twc.texas.gov' + item.find('div', class_='news-title').a['href']
          if not link:
              error_message = Error_Message(error_message, "None link")
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              content = article.text
              if not content: error_message = Error_Message(error_message, "None Contents")
              if error_message != str():
                All_error_list.append({
                    'Error Link': url_59,
                    'Error': error_message
                    }) 
              else:
                All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                Govern_articles.append({
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
     'Error Link': url_59,
     'Error': str(e)
     })
########################################### <61> ##############################################
#url_61 = 'https://comptroller.texas.gov/about/media-center/news//'
wd = initialize_chrome_driver()
wd.get(url_61)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_containers = soup.find_all('div', class_="medium-12 small-12 columns")
    if not news_containers: All_error_list.append({'Error Link': url_61, 'Error': "Entire Error1"})
    for container in news_containers:
        link = container.find('a')['href']
        if not link: error_message = Error_Message(error_message, "None link")
        article_date = date_util(container.find('br').next_sibling.strip())
        if not date: error_message = Error_Message(error_message, "None date")
        if article_date in today_list:
            try:
                article = Article(link, language = 'en') # URL과 언어를 입력
                article.download()
                article.parse()
                title = article.title
                if not title: error_message = Error_Message(error_message, "None Title")
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                    All_error_list.append({
                    'Error Link': url_61,
                    'Error': error_message
                    })
                else:
                    All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                    Govern_articles.append({
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
     'Error Link': url_61,
     'Error': str(e)
     })
########################################### <62> ##############################################
#url_62 = 'https://www.tdi.texas.gov/news/2023/index.html' ######################문제발생
wd = initialize_chrome_driver()
wd.get(url_62)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = []
    for p_tag in soup.find_all('p'):
        if p_tag.find('strong'):
            news_items.append(p_tag)
    if not news_items: All_error_list.append({'Error Link': url_62, 'Error': "Entire Error1"})
    for item in news_items:
        strong_tag = item.find('strong')
        if not strong_tag: error_message = Error_Message(error_message, "Entire Error2")
        date_str = strong_tag.get_text().rstrip(':')
        if not date_str: error_message = Error_Message(error_message, "None date")
        article_date = date_util(date_str)
        if article_date in today_list:
            a_tag = item.find('a')
            if not a_tag: error_message = Error_Message(error_message, "Entire Error3")
            title = a_tag.get_text().strip()
            if not title: error_message = Error_Message(error_message, "None title")
            link = f'https://www.tdi.texas.gov/news/2023/{item.a["href"]}'
            if not link: error_message = Error_Message(error_message, "None link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                     'Error Link': link,
                     'Error': str(e)
                     })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_62,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
     'Error Link': url_62,
     'Error': str(e)
     })
########################################### <63> ##############################################
#url_63 = 'https://www.txbiz.org/chamber-news'
wd = initialize_chrome_driver()
wd.get(url_63)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try :
    news_items = soup.find_all('div', class_='gallery-item-common-info-outer')
    if not news_items: All_error_list.append({'Error Link': url_63, 'Error': "Entire Error"})
    for article in news_items:
        link = article.find('a', class_='O16KGI')['href']
        if not link: error_message = Error_Message(error_message, "None link")
        date = article.find('span', class_='post-metadata__date').text
        if not date: error_message = Error_Message(error_message, "None date")
        news_date = date_util(date)
        if news_date in today_list:
            try:
                article = Article(link, language = 'en') # URL과 언어를 입력
                article.download()
                article.parse()
                title = article.title
                if not title: error_message = Error_Message(error_message, "None Title")
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                    All_error_list.append({
                    'Error Link': url_63,
                    'Error': error_message
                    })
                else:
                    All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                    Govern_articles.append({
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
     'Error Link': url_63,
     'Error': str(e)
     })
########################################### <64> ##############################################
#url_64 = 'https://www.txbiz.org/press-releases'
wd = initialize_chrome_driver()
wd.get(url_64)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('div', class_='gallery-item-common-info-outer')
    if not news_items: All_error_list.append({'Error Link': url_64, 'Error': "Entire Error"})
    for article in news_items:
        link = article.find('a', class_='O16KGI')['href']
        if not link: error_message = Error_Message(error_message, "None link")
        date = article.find('span', class_='post-metadata__date').text
        if not date: error_message = Error_Message(error_message, "None date")
        news_date = date_util(date)
        if news_date in today_list:
            try:
                article = Article(link, language = 'en') # URL과 언어를 입력
                article.download()
                article.parse()
                title = article.title
                if not title: error_message = Error_Message(error_message, "None Title")
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                    All_error_list.append({
                    'Error Link': url_64,
                    'Error': error_message
                    })
                else:
                    All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                    Govern_articles.append({
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
     'Error Link': url_64,
     'Error': str(e)
     })
########################################### <65> ##############################################
#url_65 = 'https://www.governor.ny.gov/news'
wd = initialize_chrome_driver()
wd.get(url_65)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.select('.content-right')
    if not news_items: All_error_list.append({'Error Link': url_65, 'Error': "Entire Error1"})
    for item in news_items:
        date_str = item.select_one('.text-proxima.text-extra-bold').get_text().strip()
        article_date = date_util(date_str)
        if not article_date: error_message = Error_Message(error_message, "None Date")
        if article_date in today_list:
            title = item.select_one(".field--name-title").get_text(strip=True)
            if not title: error_message = Error_Message(error_message, "None title")
            link = f"https://www.governor.ny.gov/{item.a['href']}"
            if not link: error_message = Error_Message(error_message, "None link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                     'Error Link': link,
                     'Error': str(e)
                     })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_65,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
     'Error Link': url_65,
     'Error': str(e)
     })
########################################### <67> ##############################################
#url_67 = 'https://www.dot.ny.gov/news/press-releases/2023'
wd = initialize_chrome_driver()
wd.get(url_67)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
seen_news_set = set()
try:
    news_items = soup.find_all('tr', align="LEFT")
    if not news_items: All_error_list.append({'Error Link': url_67, 'Error': "Entire Error1"})
    for item in news_items:
        date_cell = item.find('td', style="width:130px;")
        if date_cell == None:
          continue
        date_str = date_cell.text.strip()
        article_date = date_util(date_str)
        if article_date in today_list:
            a_tag = item.find('a', href=True, target="_blank")
            if a_tag:
              title = a_tag.text.strip()
              if not title: error_message = Error_Message(error_message, "None title")
              link = a_tag['href']
              if not link: error_message = Error_Message(error_message, "None link")
              else:
                  try:
                      article = Article(link, language='en')
                      article.download()
                      article.parse()
                      content = article.text
                      if not content: error_message = Error_Message(error_message, "None Contents")
                  except Exception as e:
                      All_error_list.append({
                       'Error Link': link,
                       'Error': str(e)
                       })
              if error_message != str():
                All_error_list.append({
                    'Error Link': url_67,
                    'Error': error_message
                    }) 
              else:
                All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                Govern_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
            else:
              title_cell = item.find('td', style=None)
              if not title_cell: error_message = Error_Message(error_message, "None title cell")
              title = title_cell.get_text(strip=True)
              link = item.a['href']
              if not link:
                error_message = Error_Message(error_message, "None link2")
              else:
                  try:
                      article = Article(link, language='en')
                      article.download()
                      article.parse()
                      content = article.text
                      if not content: error_message = Error_Message(error_message, "None Contents")
                  except Exception as e:
                      All_error_list.append({
                       'Error Link': link,
                       'Error': str(e)
                       })
              if error_message != str():
                All_error_list.append({
                    'Error Link': url_67,
                    'Error': error_message
                    }) 
              else:
                All_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
                Govern_articles.append({
                    'Title': title,
                    'Link': link,
                    'Content(RAW)': content
                    })
except Exception as e:
    All_error_list.append({
     'Error Link': url_67,
     'Error': str(e)
     })
########################################### <68> ##############################################
#url_68 = 'https://ag.ny.gov/press-releases'
wd = initialize_chrome_driver()
wd.get(url_68)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('div', class_='views-row')
    if not news_items:
        All_error_list.append({
            'Error Link': url_68,
            'Error': "None News"
        })
    for item in news_items:
        date_str = item.select_one(".field-content").get_text(strip=True)
        article_date = date_util(date_str)
        if not article_date:
            error_message = Error_Message(error_message, "None Date")
        if article_date in today_list:
            title = item.select_one('a').get_text(strip=True)
            if not title:
                error_message = Error_Message(error_message, "None Title")
            link = f"https://ag.ny.gov{item.a['href']}"
            if not link:
                error_message = Error_Message(error_message, "None Link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                        'Error Link': link,
                        'Error': str(e)
                    })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_68,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
        'Error Link': url_68,
        'Error': str(e)
    })
########################################### <71> ##############################################
#url_71 = 'https://chamber.nyc/news'
wd = initialize_chrome_driver()
wd.get(url_71)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
news_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  news_blocks = soup.find_all('div', class_='post-style2 wow fadeIn')
  if not news_blocks: All_error_list.append({'Error Link': url_71, 'Error': "None News"})
  else:
    for block in news_blocks:
      date_str = block.find('li').text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find('a')['href']
        if article_link[0] !='h' : article_link = 'https://chamber.nyc/' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language = 'en') # URL과 언어를 입력 
            article.download() 
            article.parse() 
            content = article.text 
            title = content.split('\n')[0]
            if not title: error_message = Error_Message(error_message, "None Title")
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
              All_error_list.append({
                'Error Link': url_71,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_71,
      'Error': str(e)
      })
########################################### <73> ##############################################
#url_73 = https://www.nyserda.ny.gov/About/Newsroom/2023-Announcements
wd = initialize_chrome_driver()
wd.get(url_73)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find('div', class_='content-area').find_all('li')
    if not news_items: All_error_list.append({'Error Link': url_73, 'Error': "Entire Error"})
    for article in news_items:
        news_date = date_util(article.find('span',class_='announce-date').text.strip())
        l = article.find('a')['href']
        link = f'https://www.nyserda.ny.gov{l}'
        if not link: error_message = Error_Message(error_message, "None link")
        if news_date in today_list:
            try:
                article = Article(link, language = 'en') # URL과 언어를 입력
                article.download()
                article.parse()
                title = article.title
                if not title: error_message = Error_Message(error_message, "None Title")
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
                if error_message != str():
                    All_error_list.append({
                    'Error Link': url_73,
                    'Error': error_message
                    })
                else:
                    All_articles.append({
                      'Title': title,
                      'Link': link,
                      'Content(RAW)': content
                      })
                    Govern_articles.append({
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
     'Error Link': url_73,
     'Error': str(e)
     })
########################################### <74> ##############################################
#url_74 = "https://www.njchamber.com/press-releases"
wd = initialize_chrome_driver()
wd.get(url_74)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.find_all('div', class_='g-array-item-title')
    if not news_items: All_error_list.append({'Error Link': url_74, 'Error': "Entire Error1"})
    for item in news_items:
        date_tag = item.find_next_sibling('div', class_='g-array-item-details').find('span', class_='g-array-item-date')
        date_str = date_tag.get_text(strip=True) if date_tag else None
        article_date = date_util(date_str)
        if not article_date: error_message = Error_Message(error_message, "None Date")
        if article_date in today_list:
            title_tag = item.find('a')
            title = title_tag.get_text(strip=True)
            if not title: error_message = Error_Message(error_message, "None title")
            link = "https://www.njchamber.com" + title_tag['href']
            if not link: error_message = Error_Message(error_message, "None link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                     'Error Link': link,
                     'Error': str(e)
                     })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_74,
                  'Error': error_message
                  }) 
            else:
                All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
                Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
     'Error Link': url_74,
     'Error': str(e)
     })
########################################### <75> ##############################################
#url_75 = "https://www.nj.gov/health/news/"
wd = initialize_chrome_driver()
wd.get(url_75)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
news_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  news_blocks = soup.find_all('li', class_='news_item_wrapper noDesc archive')
  if not news_blocks: All_error_list.append({'Error Link': url_75, 'Error': "None News"})
  else:
    for block in news_blocks:
      date_str = block.find('span', class_='news_item_date').text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find('a')['href']
        article_link = f'https://www.nj.gov{article_link}'
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language = 'en') # URL과 언어를 입력 
            article.download() 
            article.parse() 
            content = article.text 
            title = article.title
            if not title: error_message = Error_Message(error_message, "None Title")
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
              All_error_list.append({
                'Error Link': url_75,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_75,
      'Error': str(e)
      })
########################################### <76> ##############################################
#url_76 = 'https://www.nj.gov/mvc/news/news.htm'
wd = initialize_chrome_driver()
wd.get(url_76)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  news_blocks = soup.find('div', class_='tab-pane active').find_all('hr')
  if not news_blocks: All_error_list.append({'Error Link': url_76, 'Error': "None News"})
  else:
    for block in news_blocks:
      block = block.find_next('p')
      article_link = None
      if any(date.strftime('%B') in block.text for date in today_list):
        article_link = block.find('a')['href'].replace('..','')
        if article_link[0] =='/': article_link = 'https://www.nj.gov/mvc' + article_link
        if not article_link: error_message = Error_Message(error_message, "None Link")
        wd = initialize_chrome_driver()
        wd.get(article_link)
        time.sleep(5)
        article_html = wd.page_source
        article_soup = BeautifulSoup(article_html, 'html.parser')
        finddate = article_soup.find_all('td')
        for i in range(len(finddate)):
          if i == 2: date_str = finddate[i].text.strip()
        article_date = date_util(date_str)
        title, bodys = None, None
        if article_date in today_list:
          article = article_soup.find_all('p')
          entire = []
          for p in article: entire.append(p.get_text().strip())  # 텍스트 내용을 출력
          title = entire[1]
          if not title: error_message = Error_Message(error_message, "None Title")
          try:
              article = Article(article_link, language = 'en') # URL과 언어를 입력 
              article.download() 
              article.parse() 
              content = article.text 
              if not content: error_message = Error_Message(error_message, "None Contents")
              if error_message != str():
                All_error_list.append({
                  'Error Link': url_76,
                  'Error': error_message
                })
              else:
                All_articles.append({
                  'Title': title,
                  'Link': article_link,
                  'Content(RAW)': content
                })
                Govern_articles.append({
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
      'Error Link': url_76,
      'Error': str(e)
      })
########################################### <78> ##############################################
#url_78 = 'https://www.commerce.nc.gov/news/press-releases'
wd = initialize_chrome_driver()
wd.get(url_78)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
date_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  date_blocks = soup.find('div', class_='view-content row').find_all('span', class_='date-display-single')
  if not date_blocks: All_error_list.append({'Error Link': url_78, 'Error': "Date Blocks"})
  else:
    for block in date_blocks:
      date_str = block['content'].strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find_parent().find_parent().find_parent().find_parent().find_parent().find('a')['href']
        if article_link[0] =='/': article_link = 'https://www.commerce.nc.gov' + article_link
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
                'Error Link': url_78,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_78,
      'Error': str(e)
      })
########################################### <80> ##############################################
#url_80 = 'https://www.ncdor.gov/news/press-releases'
wd = initialize_chrome_driver()
wd.get(url_80)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.views-row')
  if not news_items: All_error_list.append({'Error Link': url_80, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.datetime').text.strip()
    article_date = date_util(date_str)
    if article_date in today_list:
      title = item.select_one('a').get_text(strip=True)
      if not title:
          error_message = Error_Message(error_message, "None Title")
      link = f"https://www.ncdor.gov{item.a['href']}"
      if not link:
          error_message = Error_Message(error_message, "None Link")
      else:
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              content = article.text
              if not content: error_message = Error_Message(error_message, "None Contents")
          except Exception as e:
              All_error_list.append({
                  'Error Link': link,
                  'Error': str(e)
                  })
      if error_message != str():
              All_error_list.append({
                  'Error Link': url_80,
                  'Error': error_message
                  }) 
      else:
        All_articles.append({
            'Title': title,
            'Link': link,
            'Content(RAW)': content
            })
        Govern_articles.append({
            'Title': title,
            'Link': link,
            'Content(RAW)': content
            })
except Exception as e:
    All_error_list.append({
        'Error Link': url_80,
        'Error': str(e)
    })
########################################### <81> ##############################################
#url_81 = "https://www.iprcenter.gov/news"
wd = initialize_chrome_driver()
wd.get(url_81)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = []
    for p_tag in soup.select("[class=row]"):
        if p_tag.find('p', class_='date'):
            news_items.append(p_tag)
    if not news_items:
        All_error_list.append({'Error Link': url_81, 'Error': "None News"})
    for item in news_items:
        date_str = item.select_one('.date').text.strip()
        article_date = date_util(date_str)
        if not article_date:
            All_error_list.append({'Error Link': url_81, 'Error': "None Date"})
        if article_date in today_list:
            title = item.select_one('a').get_text(strip=True)
            if not title:
                error_message = Error_Message(error_message, "None Title")
            link = item.a['href']
            if not link:
                error_message = Error_Message(error_message, "None Link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                        'Error Link': link,
                        'Error': str(e)
                    })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_81,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
        'Error Link': url_81,
        'Error': str(e)
    })
########################################### <82> ##############################################
#url_82 = "https://edpnc.com/news-events/"
wd = initialize_chrome_driver()
wd.get(url_82)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  news_items = soup.select('.four')
  if not news_items: All_error_list.append({'Error Link': url_82, 'Error': "None News"})
  for item in news_items:
    date_str = item.select_one('.blog_date').text.strip()
    article_date = date_util(date_str)
    if not article_date: error_list.append({'Error Link': url_82, 'Error': "None Date"})
    if article_date in today_list:
      title = item.find('h3').get_text(strip=True)
      if not title: error_message = Error_Message(error_message, "None Title")
      link = item.a['href']
      if not link: error_message = Error_Message(error_message, "None Link")
      else:
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              content = article.text
              if not content: error_message = Error_Message(error_message, "None Contents")
          except Exception as e:
             All_error_list.append({
                 'Error Link': link,
                 'Error': str(e)
                 })
      if error_message != str():
              All_error_list.append({
                  'Error Link': url_82,
                  'Error': error_message
                  }) 
      else:
        All_articles.append({
            'Title': title,
            'Link': link,
            'Content(RAW)': content
            })
        Govern_articles.append({
            'Title': title,
            'Link': link,
            'Content(RAW)': content
            })
except Exception as e:
 All_error_list.append({
     'Error Link': url_82,
     'Error': str(e)
     })
########################################### <83> ##############################################
#url_83 = "https://ncchamber.com/category/chamber-updates/"
wd = initialize_chrome_driver()
wd.get(url_83)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
     news_items = soup.select(".entry-article")
     if not news_items:
         All_error_list.append({
             'Error Link': url_83,
             'Error': "None News"
         })
     else:
         for item in news_items:
             date_str = item.select_one('time[datetime]').get_text(strip=True)
             article_date = date_util(date_str)
             if not article_date:
                 error_message = Error_Message(error_message, "None Date")
             if article_date in today_list:
                 title = item.select_one('h2 a').get_text(strip=True)
                 if not title:
                     error_message = Error_Message(error_message, "None Title")
                 article_link = f"{item.a['href']}"
                 if not article_link:
                     error_message = Error_Message(error_message, "None Link")
                 wd = initialize_chrome_driver()
                 wd.get(article_link)
                 article_html = wd.page_source
                 article_soup = BeautifulSoup(article_html, 'html.parser')
                 content = article_soup.select_one(".container-post")
                 paragraphs = content.find_all("p")
                 article_body = ' '.join(p.get_text(strip=True) for p in paragraphs)
                 if not article_body:
                     error_message = Error_Message(error_message, "None Contents")
                 if error_message != "": 
                     All_error_list.append({
                         'Error Link': url_83,
                         'Error': error_message
                     })
                 else:
                     All_articles.append({
                         'Title': title,
                         'Link': article_link,
                         'Content(RAW)': article_body
                     })
                     Govern_articles.append({
                         'Title': title,
                         'Link': article_link,
                         'Content(RAW)': article_body
                     })
except Exception as e:
    All_error_list.append({
        'Error Link': url_83,
        'Error': str(e)
    })
########################################### <84> ##############################################
#url_84 = "https://dc.gov/newsroom"
wd = initialize_chrome_driver()
wd.get(url_84)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.select('.views-row')
    if not news_items: All_error_list.append({'Error Link': url_84, 'Error': "None News"})
    for item in news_items:
      date_str = item.select_one('.date-display-single').text.strip()
      article_date = date_util(date_str)
      if not article_date: All_error_list.append({'Error Link': url_84, 'Error': "None Date"})
      if article_date in today_list:
        a_tag = item.find('a')
        title = a_tag.text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        link = f"https://dc.gov{a_tag['href']}"
        if not link: error_message = Error_Message(error_message, "None Link")
        else:
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
            except Exception as e:
                All_error_list.append({
                    'Error Link': link,
                    'Error': str(e)
                })
        if error_message != str():
              All_error_list.append({
                  'Error Link': url_84,
                  'Error': error_message
                  }) 
        else:
          All_articles.append({
              'Title': title,
              'Link': link,
              'Content(RAW)': content
              })
          Govern_articles.append({
              'Title': title,
              'Link': link,
              'Content(RAW)': content
              })
except Exception as e:
    All_error_list.append({
        'Error Link': url_84,
        'Error': str(e)
    })
########################################### <85> ##############################################
#url_85 = 'https://dcchamber.org/posts/'
wd = initialize_chrome_driver()
wd.get(url_85)
time.sleep(5)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
news_blocks, article_date, article_link, title, bodys = None, None, None, None, None
try:
  news_blocks = soup.find_all('tr', {'style':'display: table-row;'})
  if not news_blocks: All_error_list.append({'Error Link': url_85, 'Error': "None News"})
  else:
    for block in news_blocks[1:]:
      date_str = block.find('td').text.strip()
      article_date = date_util(date_str)
      article_link, title, bodys = None, None, None
      if article_date in today_list:
        article_link = block.find('a')['href']
        if not article_link: error_message = Error_Message(error_message, "None Link")
        try:
            article = Article(article_link, language = 'en') # URL과 언어를 입력 
            article.download() 
            article.parse() 
            content = article.text 
            title = article.title
            if not title: error_message = Error_Message(error_message, "None Title")
            if not content: error_message = Error_Message(error_message, "None Contents")
            if error_message != str():
              All_error_list.append({
                'Error Link': url_85,
                'Error': error_message
              })
            else:
              All_articles.append({
                'Title': title,
                'Link': article_link,
                'Content(RAW)': content
              })
              Govern_articles.append({
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
      'Error Link': url_85,
      'Error': str(e)
      })
########################################### <86> ##############################################
#url_86 = "https://planning.dc.gov/newsroom"
wd = initialize_chrome_driver()
wd.get(url_86)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = []
    for p_tag in soup.select(".views-row"):
        if p_tag.select('.date-display-single'):
            news_items.append(p_tag)
    if not news_items: All_error_list.append({'Error Link': url_86, 'Error': "None News"})
    for item in news_items:
      date_str = item.select_one('.date-display-single').text.strip()
      article_date = date_util(date_str)
      if not article_date: All_error_list.append({'Error Link': url_86, 'Error': "None Date"})
      if article_date in today_list:
        a_tag = item.find('a')
        title = a_tag.text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        link = f"https://planning.dc.gov{a_tag['href']}"
        if not link: error_message = Error_Message(error_message, "None Link")
        else:
            try:
                article = Article(link, language='en')
                article.download()
                article.parse()
                content = article.text
                if not content: error_message = Error_Message(error_message, "None Contents")
            except Exception as e:
               All_error_list.append({
                   'Error Link': link,
                   'Error': str(e)
                   })
        if error_message != str():
              All_error_list.append({
                  'Error Link': url_86,
                  'Error': error_message
                  }) 
        else:
          All_articles.append({
              'Title': title,
              'Link': link,
              'Content(RAW)': content
              })
          Govern_articles.append({
              'Title': title,
              'Link': link,
              'Content(RAW)': content
              })
except Exception as e:
 All_error_list.append({
     'Error Link': url_86,
     'Error': str(e)
     })
########################################### <87> ##############################################
#url_87 = "https://dpw.dc.gov/newsroom"
wd = initialize_chrome_driver()
wd.get(url_87)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = []
    for p_tag in soup.select(".views-row"):
        if p_tag.select('.date-display-single'):
            news_items.append(p_tag)
    if not news_items: All_error_list.append({'Error Link': url_87, 'Error': "None News"})
    for item in news_items:
      date_str = item.select_one('.date-display-single').text.strip()
      article_date = date_util(date_str)
      if not article_date: All_error_list.append({'Error Link': url_87, 'Error': "None Date"})
      if article_date in today_list:
        a_tag = item.find('a')
        title = a_tag.text.strip()
        if not title: error_message = Error_Message(error_message, "None Title")
        link = f"https://dpw.dc.gov{a_tag['href']}"
        if not link: error_message = Error_Message(error_message, "None Link")
        else:
            try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              content = article.text
              if not content: error_message = Error_Message(error_message, "None Contents")
            except Exception as e:
               All_error_list.append({
                   'Error Link': link,
                   'Error': str(e)
                   })
        if error_message != str():
              All_error_list.append({
                  'Error Link': url_87,
                  'Error': error_message
                  }) 
        else:
          All_articles.append({
              'Title': title,
              'Link': link,
              'Content(RAW)': content
              })
          Govern_articles.append({
              'Title': title,
              'Link': link,
              'Content(RAW)': content
              })
except Exception as e:
 All_error_list.append({
     'Error Link': url_87,
     'Error': str(e)
     })
########################################### <88> ##############################################
#url_88 = 'https://www.governor.virginia.gov/newsroom/news-releases/'
wd = initialize_chrome_driver()
wd.get(url_88)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.select(".col-md-12")
    if not news_items: All_error_list.append({'Error Link': url_88, 'Error': "None News"})
    for item in news_items:
        date_str = item.select_one(".date").text.strip().strip('"')
        article_date = date_util(date_str)
        if not article_date: error_message = Error_Message(error_message, "None Date")
        if article_date in today_list:
            title = item.select_one(".va-icon--arrow-ext").text
            if not title: error_message = Error_Message(error_message, "None Title")
            link = f"https://www.governor.virginia.gov{item.a['href']}"
            if not link: error_message = Error_Message(error_message, "None Link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                        'Error Link': link,
                        'Error': str(e)
                    })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_88,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
        'Error Link': url_88,
        'Error': str(e)
    })
########################################### <89> ##############################################
#url_89 = "https://www.vedp.org/press-releases"
wd = initialize_chrome_driver()
wd.get(url_89)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
  date_items = soup.select("[class=date]")
  if not news_items: All_error_list.append({'Error Link': url_89, 'Error': "None News"})
  for datezz in date_items:
    date_str = datezz.text.strip()
    article_date = date_util(date_str)
    if not article_date: error_message = Error_Message(error_message, "None Date")
    if article_date in today_list:
      h3_tag = datezz.find_previous_sibling('h3', class_='field-content')
      if not h3_tag: error_message = Error_Message(error_message, "None h3_tag")
      a_tag = h3_tag.find('a')
      if not a_tag: error_message = Error_Message(error_message, "None a_tag")
      title = a_tag.text.strip()
      if not title: error_message = Error_Message(error_message, "None Title")
      link = f"https://www.vedp.org{a_tag['href']}"
      if not link: error_message = Error_Message(error_message, "None Link")
      else:
          try:
              article = Article(link, language='en')
              article.download()
              article.parse()
              content = article.text
              if not content: error_message = Error_Message(error_message, "None Contents")
          except Exception as e:
             All_error_list.append({
                 'Error Link': link,
                 'Error': str(e)
                 })
      if error_message != str():
              All_error_list.append({
                  'Error Link': url_89,
                  'Error': error_message
                  }) 
      else:
        All_articles.append({
            'Title': title,
            'Link': link,
            'Content(RAW)': content
            })
        Govern_articles.append({
            'Title': title,
            'Link': link,
            'Content(RAW)': content
            })
except Exception as e:
 All_error_list.append({
     'Error Link': url_89,
     'Error': str(e)
     })
########################################### <90> ##############################################
#url_90 = "https://www.doli.virginia.gov/category/announcements/"
wd = initialize_chrome_driver()
wd.get(url_90)
time.sleep(3)
error_message = str()
try:
      def get_article_list(url, filter_date):
          wd.get(url_90)  
          html = wd.page_source 
          soup = BeautifulSoup(html, 'html.parser')
          articles_info = []
          error_message = ""
          blocks = soup.find_all('article')
          if not blocks:
              All_error_list.append({
                  'Error Link': url_90,
                  'Error': "None News"
              })
          for article in blocks:
              error_message = ""
              title_tag = article.find('h3') 
              link_tag = article.find('a', rel='bookmark')
              date_tag = article.find('time', class_='updated')
              if title_tag:
                  title = title_tag.get_text(strip=True)
              else:
                  title = None
                  error_message = Error_Message(error_message, "None Title")
              if link_tag and 'href' in link_tag.attrs:
                  link = link_tag['href']
              else:
                  link = None
                  error_message = Error_Message(error_message, "None Link")
              if date_tag and 'datetime' in date_tag.attrs:
                  date = date_tag['datetime']
                  date_parsed = datetime.strptime(date, '%Y-%m-%d').date()
              else:
                  date_parsed = None
                  error_message = Error_Message(error_message, "None Date")
              if date_parsed in filter_date and not error_message:
                  articles_info.append({'title': title, 'link': link})
              elif error_message:
                  error_list.append({
                      'Error Link': url,
                      'Error': error_message
                  })
          return articles_info
      def get_article_content(article_url):
          wd = initialize_chrome_driver()
          wd.get(url_90)
          article_html = wd.page_source 
          article_soup = BeautifulSoup(article_html, 'html.parser')
          error_message = ""
          content_div = article_soup.find('div', {'class': 'proclamation'})
          if not content_div:
              error_message = "None Contents"
              return "", error_message 
          paragraphs = content_div.find_all('p') if content_div else []
          content = ' '.join(paragraph.text.strip() for paragraph in paragraphs)
          return content, error_message
      articles_list = get_article_list(url_90, today_list)
      for article in articles_list:
          full_url = article['link']
          content, content_error_message = get_article_content(full_url)
          if content_error_message:
              All_error_list.append({
                  'Error Link': url_90,
                  'Error': content_error_message
              })
          else:
              All_articles.append({
                  'Title': article['title'],
                  'Link': full_url,
                  'Content(RAW)': content
              })
              Govern_articles.append({
                  'Title': article['title'],
                  'Link': full_url,
                  'Content(RAW)': content
              })
except Exception as e:
 All_error_list.append({
     'Error Link': url_90,
     'Error': str(e)
     })
########################################### <91> ##############################################
#url_91 = 'https://vachamber.com/category/press-releases/'
wd = initialize_chrome_driver()
wd.get(url_91)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.select(".archiveitem")
    if not news_items: All_error_list.append({'Error Link': url_91, 'Error': "None News"})
    for item in news_items:
        date_str = item.select_one('.item_meta').get_text(strip=True)
        article_date = date_util(date_str)
        if not article_date: error_message = Error_Message(error_message, "None Date")
        if article_date in today_list:
            title = item.select_one('h4 a').get_text(strip=True)
            if not title: error_message = Error_Message(error_message, "None Title")
            link = f"{item.a['href']}"
            if not link: error_message = Error_Message(error_message, "None Link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                        'Error Link': link,
                        'Error': str(e)
                    })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_91,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
        'Error Link': url_91,
        'Error': str(e)
    })
########################################### <92> ##############################################
#url_92 = "https://governor.maryland.gov/news/press/Pages/default.aspx?page=1"
wd = initialize_chrome_driver()
wd.get(url_92)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try :
    div_content = soup.find('div', class_ = 'mdgov-twoCol-col2 order-1')
    if not div_content: All_error_list.append({'Error Link': url_92, 'Error': "Entire Error1"})
    else:
      tags = div_content.find_all('div', class_ = 'col')
      if not tags: All_error_list.append({'Error Link': url_92, 'Error': "Entire Error2"})
    for div_tag in tags:
        date_tag = div_tag.find('p', class_='font-weight-bold')
        date_str = date_tag.text.split(': ')[-1]
        if not date_str: error_message = Error_Message(error_message, "None date")
        article_date = date_util(date_str)
        if article_date in today_list:
            title_tag = div_tag.find('h2')
            if not title_tag: error_message = Error_Message(error_message, "None title")
            title = title_tag.text
            linkzz = title_tag.find_parent('a')['href']
            link = f'https://governor.maryland.gov{linkzz}'
            if not link: error_message = Error_Message(error_message, "None Link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                     'Error Link': link,
                     'Error': str(e)
                     })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_92,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
     'Error Link': url_92,
     'Error': str(e)
     })
########################################### <93> ##############################################
#url_93 = 'https://news.maryland.gov/mde/category/press-release/'
wd = initialize_chrome_driver()
wd.get(url_93)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    news_items = soup.select(".type-post")
    if not news_items: All_error_list.append({'Error Link': url_93, 'Error': "None News"})
    for item in news_items:
        li_tag = item.select_one('li')
        title_str = li_tag.select_one('a').text.strip()
        date_str = li_tag.text.replace(title_str, '').strip()
        article_date = date_util(date_str)
        if not article_date: error_message = Error_Message(error_message, "None Date")
        if article_date in today_list:
            title = item.select_one('a').get_text(strip=True)
            if not title: error_message = Error_Message(error_message, "None Title")
            link = f"{item.a['href']}"
            if not link: error_message = Error_Message(error_message, "None Link")
            else:
                try:
                    article = Article(link, language='en')
                    article.download()
                    article.parse()
                    content = article.text
                    if not content: error_message = Error_Message(error_message, "None Contents")
                except Exception as e:
                    All_error_list.append({
                        'Error Link': link,
                        'Error': str(e)
                    })
            if error_message != str():
              All_error_list.append({
                  'Error Link': url_93,
                  'Error': error_message
                  }) 
            else:
              All_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
              Govern_articles.append({
                  'Title': title,
                  'Link': link,
                  'Content(RAW)': content
                  })
except Exception as e:
    All_error_list.append({
        'Error Link': url_93,
        'Error': str(e)
    })
########################################### <94> ##############################################
# url_94 = 'https://commerce.maryland.gov/media/press-room'
wd = initialize_chrome_driver()
wd.get(url_94)
base_url = "https://commerce.maryland.gov"
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    # 뉴스 아이템 가져오기
    news_item = soup.find('table', id='speakerTable')
    if not news_item:
        All_error_list.append({
            'Error Link': url_94,
            'Error': "None News"
        })
    else:
        for row in news_item.find_all('tr'):
            error_message = ''
            # 날짜와 제목을 추출
            date_tag = row.find('nobr')
            title_tag = row.find('a')
            if date_tag and title_tag:
                date_string = date_tag.text.strip()
                article_date = date_util(date_string)
                title = title_tag.text.strip()
                link = title_tag['href']
                full_link = base_url + link if link.startswith('/') else link
                if article_date in today_list:
                    wd = initialize_chrome_driver()
                    wd.get(full_link)
                    time.sleep(3)
                    article_html = wd.page_source
                    article_soup = BeautifulSoup(article_html, 'html.parser')
                    content = ''
                    content_blocks = article_soup.find('div', id='ctl00_PlaceHolderMain_ctl06__ControlWrapper_RichHtmlField')
                    if content_blocks:
                        spans = content_blocks.find_all('span', style=lambda value: value and 'font-family' in value)
                        for span in spans:
                            content += span.get_text(strip=True) + ' '
                    if not content:
                        error_message = Error_Message(error_message, "None Contents")
                    if error_message != str():
                        All_error_list.append({
                            'Error Link': url_94,
                            'Error': error_message
                        })
                    else:
                        All_articles.append({
                            'Title': title,
                            'Link': full_link,
                            'Content(RAW)': content
                        })
                        Govern_articles.append({
                            'Title': title,
                            'Link': full_link,
                            'Content(RAW)': content
                        })
except Exception as e:
    All_error_list.append({
        'Error Link': url_94,
        'Error': str(e)
    })
########################################### <95> ##############################################
#url_95 = 'https://www.dllr.state.md.us/whatsnews/'
wd = initialize_chrome_driver()
wd.get(url_95)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try :
    div_content = soup.find('div', {'id': 'ui-id-2'})
    if not div_content: All_error_list.append({'Error Link': url_95, 'Error': "None div_content"})    
    li_tags = div_content.select('ul li')
    if not li_tags: All_error_list.append({'Error Link': url_95, 'Error': "None li_tags"})    
    for li in li_tags:
        a_tag = li.find('a')
        date_str = li.text.split(' - ')[-1]
        if not date_str: error_message = Error_Message(error_message, "None date")
        article_date = date_util(date_str)        
        if article_date in today_list:
          title = a_tag.text
          if not title: error_message = Error_Message(error_message, "None Title")
          link = f"https://www.dllr.state.md.us/whatsnews/{a_tag['href']}"
          if not link: error_message = Error_Message(error_message, "None Link")
          else:
              try:
                  article = Article(link, language='en')
                  article.download()
                  article.parse()
                  content = article.text
                  if not content: error_message = Error_Message(error_message, "None Contents")
              except Exception as e:
                  All_error_list.append({
                   'Error Link': link,
                   'Error': str(e)
                   })
          if error_message != str():
              All_error_list.append({
                  'Error Link': url_95,
                  'Error': error_message
                  }) 
          else:
            All_articles.append({
                'Title': title,
                'Link': link,
                'Content(RAW)': content
                })
            Govern_articles.append({
                'Title': title,
                'Link': link,
                'Content(RAW)': content
                })
except Exception as e:
    All_error_list.append({
     'Error Link': url_95,
     'Error': str(e)
     })
########################################### <96> ##############################################
url_96 = "https://www.mdchamber.org/news/"
wd = initialize_chrome_driver()
wd.get(url_96)
time.sleep(3)
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
error_message = str()
try:
    items = soup.find_all('div', class_='fl-post-feed-header')
    if not items:
        All_error_list.append({
            'Error Link': url_96,
            'Error': "None News"
        })
    else:
        for item in items:
            link_tag = item.find('a', href=True)
            link = link_tag['href'] if link_tag else None
            date_tag = item.find('span', class_='fl-post-feed-date')
            date = date_tag.get_text(strip=True) if date_tag else None
            if article_date in today_list :
              try:
                  article = Article(link, language='en')
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
                        'Error Link': url_96,
                        'Error': error_message
                        })     
                  else:
                    All_articles.append({
                        'Title': title,
                        'Link': link,
                        'Content(RAW)': text
                        })
                    Govern_articles.append({
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
        'Error Link': url_96,
        'Error': str(e)
    })

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
wd = initialize_chrome_driver()
wd.get(url_local_5)
time.sleep(5)
for _ in range(5):
  button = wd.find_element(By.ID, 'load-more')
  button.click()
  time.sleep(1)
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
if All_articles == []:
  print("No news for today!")

articles = pd.DataFrame(All_articles)
articles = articles.drop_duplicates(subset=['Link'])
articles = articles.drop_duplicates(subset=['Title'])
articles.reset_index(drop=True, inplace=True)
articles.to_csv('US_All Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

articles2 = pd.DataFrame(Govern_articles)
articles2 = articles2.drop_duplicates(subset=['Link'])
articles2 = articles2.drop_duplicates(subset=['Title'])
articles2.reset_index(drop=True, inplace=True)
articles2.to_csv('US_Government Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

articles3 = pd.DataFrame(DefenseIndustry_articles)
articles3 = articles3.drop_duplicates(subset=['Link'])
articles3 = articles3.drop_duplicates(subset=['Title'])
articles3.reset_index(drop=True, inplace=True)
articles3.to_csv('US_Defense Industry Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

articles4 = pd.DataFrame(Local_articles)
articles4 = articles4.drop_duplicates(subset=['Link'])
articles4 = articles4.drop_duplicates(subset=['Title'])
articles4.reset_index(drop=True, inplace=True)
articles4.to_csv('US_Local Articles' + datetime.now().strftime("_%y%m%d") + '.csv')

error_list = pd.DataFrame(All_error_list)
error_list.to_csv('US_ All Error List' + datetime.now().strftime("_%y%m%d") + '.csv')

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
