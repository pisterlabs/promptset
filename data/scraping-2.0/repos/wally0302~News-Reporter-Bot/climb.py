import json
import os
import requests
from bs4 import BeautifulSoup
import datetime
import openai
import telepot
#Aws
import boto3
import logging

#docker
bot_api_key = os.environ.get('bot_API_KEY')


botID = telepot.Bot(bot_api_key)


def enter_the_venturebeat(url):

    index=2
    r = requests.get(url) 
    soup = BeautifulSoup(r.text,"html.parser")

    # 標題
    title = soup.find('h1',class_='article-title')
    # 內文
    context = soup.find('div',class_='article-content')

    
    #創建一個 key-value 的 dictionary
    data = {}
    #將 title 加進 dictionary
    data['title'] = title.text
    #將 context 加進 dictionary
    data['context'] = context.text
    #將 url 加進 dictionary
    data['url'] = url

    #將 data 傳給 chatGpt3
    get_outline(data,data['context'],index)



#進入 technews
def enter_the_technews(url):
    index=1
    r = requests.get(url) 
    soup = BeautifulSoup(r.text,"html.parser")

    # 標題
    title = soup.find('h1',class_='entry-title')
    # 內文
    context = soup.find('div',class_='indent')

    
    #創建一個 key-value 的 dictionary
    data = {}
    #將 title 加進 dictionary
    data['title'] = title.text
    #將 context 加進 dictionary
    data['context'] = context.text
    #將 url 加進 dictionary
    data['url'] = url

    #將 data 傳給 chatGpt3
    get_outline(data,data['context'],index)







#進入 hackernews
def enter_the_hackernews(url):
    index=0
    r = requests.get(url) #將網頁資料GET下來
    soup = BeautifulSoup(r.text,"html.parser") #將網頁資料以html.parser

    #該網站的標題
    title = soup.find('h1',class_='story-title')

    #存文章內容的 array
    context=[]
    #取得該網站的 p tag
    total = soup.find_all('p')
    #取得該網站的 p tag 的文字，最後一個p tag是廣告，所以不要
    for i in range(0,len(total)-1):
        context.append(total[i].text)

    
    #創建一個 key-value 的 dictionary
    data = {}
    #將 title 加進 dictionary
    data['title'] = title.text
    #將 context 加進 dictionary
    data['context'] = context
    #將 url 加進 dictionary
    data['url'] = url

    #將 data['context'] 轉成 string
    articles = ''.join(data['context'])
    
    #將 data 傳給 chatGpt3
    get_outline(data,articles,index)




#把 dictionary 丟給chatGpt3 整理成大綱
def get_outline(data,articles,index):
    
    news=index#判別是哪個新聞網站
    #docker 專用
    openAI_API_KEY = os.environ.get('openAI_API_KEY')
    openai.api_key =openAI_API_KEY



    #跟 gpt 溝通
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "將以下內容用項目符號條列式整理重點，以台灣最常用的正體中文表達:"},
        {"role": "user", "content": articles},
      ]
    )
    #gpt 回傳的結果
    ans=completion.choices[0].message



    print('標題 : '+data['title'])
    print("\n")
    print('網址 : '+data['url'])
    print("\n")
    print('大綱 : '+ans['content'])
    print("\n")
    print("--------------------------------------------------")
    send_daily_message(data,ans,news)
    
# bot 發送訊息
def send_daily_message(data,ans,news):
    if news==0: #TheHackerNews
        with open('user_TheHackerNews.txt', 'r') as file:
            user_ids = file.read().splitlines()
    elif news==1: #technews
        with open('user_TechNews.txt', 'r') as file:
            user_ids = file.read().splitlines()
    elif news==2: #VentureBeat
        with open('user_VentureBeat.txt', 'r') as file:
            user_ids = file.read().splitlines()
    
    message = '標題 : ' + data['title'] + '\n\n' \
              '網址 : ' + data['url']+ '\n\n' \
              '大綱 : ' + ans['content'] + '\n\n' \
    
    for user_id in user_ids:
        botID.sendMessage(user_id, message)

    save_output_as_json(data, ans['content'],news)
    
def save_output_as_json(data,ans,news):

    if news==0:
        name='TheHackerNews'
    elif news==1:
        name='TechNews'
    elif news==2:
        name='VentureBeat'

    # 檔案名稱
    json_file = 'article_' + name + '_'+str(datetime.date.today()) + '.json'  # article_TheHackerNews_2023-06-05.json

    try:
        # 嘗試讀取現有資料
        with open(json_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        # 若檔案不存在，創建一個空的資料列表
        original_data = []

    # 新資料
    output = {
        '標題': data['title'],
        '網址': data['url'],
        '大綱': ans
    }

    # 將新資料附加到原有資料後面
    original_data.append(output)

    # 將資料寫入 JSON 檔案
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)
    
    result_upload = upload_file(json_file, "wallys3demo", json_file)
    if result_upload :
        print("bucket file uploaded successfully..!")
    else:
        print("bucket file upload failed..!")

#s3 服務
def upload_file(file_name, bucket, object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except Exception as e:
        logging.error(e)
        return False
    return True

def get_hackernews():
    r = requests.get("https://thehackernews.com/") #將網頁資料GET下來
    soup = BeautifulSoup(r.text,"html.parser") #將網頁資料以html.parser

    #今天日期
    today = datetime.date.today()
    formatted_date = today.strftime("%b %d, %Y")# Jun 05, 2023

    #取得class="body-post clear"的tag
    total = soup.find_all('div',class_='body-post clear')

    #array
    titles=[]
    links=[]
    dates=[]

    for i in range(len(total)):
    #取得class="body-post clear"底下的 title、link、date
        #標題
        title = total[i].find('h2',class_='home-title')
        #加進去array
        titles.append(title.text)
        #超連結
        link = total[i].find('a',class_='story-link')
        links.append(link.get('href'))
        #日期，如果找不到日期則顯示None
        if(total[i].find('span',class_='h-datetime') == None):
            dates.append(None)
        else:
            #去除第一個符號
            dates.append(total[i].find('span',class_='h-datetime').text[1:])

    # enter_the_hackernews(links[0])

    NoNews=0
    # 如果日期是今天，就進去該網站，如果沒有則寄送當日無新聞
    for i in range(len(dates)):
        if(dates[i]==formatted_date):
            enter_the_hackernews(links[i])
            NoNews=1

    if(NoNews==0):
        with open('user_TheHackerNews.txt', 'r') as file:
            user_ids = file.read().splitlines()
        for user_id in user_ids:
            botID.sendMessage(user_id, '當日無新聞')


def get_technews():
    headers = {'Cache-Control': 'no-cache'}

    r = requests.get("https://technews.tw/category/ai/", headers=headers) #將網頁資料GET下來
    soup = BeautifulSoup(r.text,"html.parser") #將網頁資料以html.parser
    #今天日期
    today = datetime.date.today()

    total = soup.find_all('header',class_='entry-header')

    #array
    titles=[]
    links=[]
    dates=[]

    for i in range(len(total)):
        #標題
        title = total[i].find('h1',class_='entry-title')
        titles.append(title.text)
        #超連結
        link = total[i].find('a')['href']
        links.append(link)
        #日期
        date=total[i].find_all('span',class_='body') #本來格式為 "2023 年 06 月 13 日 8:30" ->  "2023-06-13"
        date_obj=datetime.datetime.strptime((date[1].text.strip()), "%Y 年 %m 月 %d 日 %H:%M").strftime("%Y-%m-%d")
        dates.append(date_obj)

    NoNews=0
    # enter_the_technews(links[0])

    # 如果日期是今天，就進去該網站
    for i in range(len(dates)):
        if(dates[i]==str(today)):
            enter_the_technews(links[i])
            NoNews=1

    if(NoNews==0):
        with open('user_TechNews.txt', 'r') as file:
            user_ids = file.read().splitlines()
        for user_id in user_ids:
            botID.sendMessage(user_id, '當日無新聞')



def get_venturebeat():
    headers = {'Cache-Control': 'no-cache'}

    r = requests.get("https://venturebeat.com/", headers=headers) #將網頁資料GET下來
    soup = BeautifulSoup(r.text,"html.parser") #將網頁資料以html.parser

    #今天日期
    today = datetime.date.today()
    formatted_date = today.strftime("%b %d, %Y")# Jun 05, 2023

    total = soup.find_all('header',class_='ArticleListing__body')
    
    #array
    titles=[]
    links=[]
    dates=[]

    for i in range(len(total)):
        #標題
        title = total[i].find('h2',class_='ArticleListing__title')
        # print(title.text)
        titles.append(title.text)
        #超連結
        link = total[i].find('a')['href']
        # print(link)
        links.append(link)
        #日期
        date = total[i].find('time', class_='ArticleListing__time')
        if date is not None:
            date_obj = datetime.datetime.strptime(date.text.strip(), "%B %d, %Y %I:%M %p")
            formatted_date = date_obj.strftime("%b %d, %Y")
            dates.append(formatted_date)
        else:
            dates.append(None)
    NoNews=0
    
    # enter_the_venturebeat(links[0])

    # 如果日期是今天，就進去該網站
    for i in range(len(dates)):
        if(dates[i]==str(today)):
            enter_the_venturebeat(links[i])
            NoNews=1
    if(NoNews==0):
        with open('user_VentureBeat.txt', 'r') as file:
            user_ids = file.read().splitlines()
        for user_id in user_ids:
            botID.sendMessage(user_id, '當日無新聞')
