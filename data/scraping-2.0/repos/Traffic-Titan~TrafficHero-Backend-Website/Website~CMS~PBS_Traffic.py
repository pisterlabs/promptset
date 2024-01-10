from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import Service.Token as Token
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
from PIL import Image
import os
from Main import MongoDB # 引用MongoDB連線實例
import Service.TDX as TDX
import Website.CMS.MainContent as CMS_MainContent
from datetime import datetime, timedelta
from urllib import request
import openai
import json


router = APIRouter(tags=["3.即時訊息推播(Website)"],prefix="/Website/CMS")

@router.put("/PBS", summary="【Update】即時訊息推播-警廣路況")
async def getPBS_TrafficAPI(token: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    Token.verifyToken(token.credentials,"admin") # JWT驗證

    return await getPBS_Traffic()

async def getPBS_Traffic():
    documents = []

    collection = await MongoDB.getCollection("traffic_hero","cms_main_car") # 取得MongoDB的collection

    pbsRequest = json.load(request.urlopen("https://od.moi.gov.tw/MOI/v1/pbs"))

    # 指定時間格式
    timeFormatter = "%Y-%m-%d %H:%M:%S"
    count = 0
    index = 0

    for data in pbsRequest:
        
        if(count != 1000):
            timeDate = datetime.strptime(data['happendate']+" "+ str(data['happentime']).replace(".0000000",""), timeFormatter)

            # 判斷字數小於25字以下的短行內容才進行處理
            if(len(data['comment']) <= 25 and timeDate > datetime.now() - timedelta(hours=1)):

                
                detail = processData(data['roadtype']+" "+data['areaNm']+" "+data['comment']+" "+str(timeDate), data['roadtype'],data['region'],float(data['y1']),float(data['x1']),datetime.strptime(data['happendate']+" "+ str(data['happentime']).replace(".0000000",""), timeFormatter))
                # documents.append(detail)
            
                CMS_MainContent.create("car", detail)
                
                count = count + 1
                  
        else:
            break
    
    return {"message": f"更新成功，總筆數:{await collection.count_documents({})}"}

def processData(Description:str,type:str,direction:str,latitude:float,longitude:float,happenedTime:datetime):

    # 指定輸出格式
    user = Description +  """
    \n輸出格式範例
    (輸出內容不要顯示格式外的文字也不要顯示任何括號)
    (以下內容已台灣為本位，使用台灣用語與機關名稱):
    一-一.事件發生的縣市名稱(格式:XX縣/市，若資料不明確則顯示無，如:雲林縣):
    一-二.事件發生的行政區名稱(格式:鄉/鎮/市/區，若資料不明確則顯示無，如:斗六市):
    二.事件發生的開始時間(格式如:2023/10/18 22:00):
    三.結束時間(格式如:2023/10/18 22:00，若無明顯結束時間，則設定成開始時間後30分鐘):
    四.模擬路況事件口播稿(模擬警察廣播電台的播報員播報路況的口播稿，簡化開頭詞句與結尾詞句，只要把「事件內容整理清楚」並「完整呈現」即可，如:蘇澳地區的外側和外路肩將進行施工封閉。):
    五.模擬資訊可變標誌CMS顯示內容(從事件中整理出重點文句，每一句不得超過8個字，每一句用/////隔開(代表換行)，只能使用一次或兩次/////分隔符號，若超過以上條件會無法正常顯示，範例格式一:西螺服務區/////狀態:未滿，範例格式二:西螺服務區/////狀態:未滿/////尚有30格停車位，以上範例僅供參考，請勿直接使用此文字，請根據input事件整理，也請確定字數、行數限制，只要把事件內容完整呈現即可)
    
    輸出內容範例:
    一-一.雲林縣
    一-二.西螺鎮
    二.2023/11/24 22:00
    三.2023/11/24 22:30
    四.前方西螺服務區，目前還有150格停車位，停車位未滿
    五.西螺服務區/////狀態:未滿/////尚有150格停車位
    """
    # 指定OpenAI_API KEY
    openai.api_key = os.getenv('OpenAI_Key')

    if user :
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
                {"role": "system", "content": "我需要用繁體中文輸出"},
                {"role": "user", "content": user},
            ]
        )
        # print("\nChatGPT:")
        result = response['choices'][0]['message']['content']
        # print(result)

        CMSDetailArray = []
        contentArray = []
        voiceContent = ""
        try:
            # 分割出語音需要的字串
            if("四." in result):
                voiceArray = result.split("四.")
                voiceArray = voiceArray[1].split("五.")

                # 語音訊息
                voiceContent = voiceArray[0]

            # 分割出CMS顯示的字串
            if("五." in result):
                CMSArray = result.split("五.")
                
                # 判斷斜線的次數
                slushTimes = CMSArray[1].count("/////")

                # 將分割完/////的內容放進陣列
                for data in CMSArray[1].split("/////"):
                    CMSDetailArray.append(data)

            # 根據/////的次數生成對應次數的content，並存進陣列
            for data in range(0,slushTimes):
                content = {
                    "text": [CMSDetailArray[data]],
                    "color": ["#FFFFFF"]
                }
                contentArray.append(content)
        except:
            pass
        
        
        
        content = {
                "type": type,
                "icon": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Zeichen_123_-_Baustelle%2C_StVO_1992.svg/150px-Zeichen_123_-_Baustelle%2C_StVO_1992.svg.png",
                "content": contentArray,
                "voice": f"前方{voiceContent}",
                "location": {
                    "longitude": longitude,
                    "latitude": latitude
                },
                "direction": direction,
                "distance": 2.5,
                "priority": "Demo", # Demo
                "start": happenedTime,
                "end": happenedTime + timedelta(minutes = 10),
                "active": True,
                "id": "string"
            }
        
    return content