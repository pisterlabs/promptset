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
import math

router = APIRouter(tags=["3.即時訊息推播(Website)"],prefix="/Website/CMS")
@router.put("/Sidebar_TrafficAPI", summary="【Update】即時訊息推播-路肩開放情況")
async def getSidebar_TrafficAPI(token: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    Token.verifyToken(token.credentials,"user") # JWT驗證

    collection = await MongoDB.getCollection("traffic_hero","sidebar_car_testing") # 取得MongoDB的collection

    documents = []

    sidebarRequest = json.load(request.urlopen("https://1968.freeway.gov.tw/api/getShoulder?freewayid=0&expresswayid=0"))
    happenedTime = datetime.now()

    for data in sidebarRequest['response']:
        # 開放路肩資料
        status = data['description']
        contentDetail = data['content'] 

        roadName = f"國道{data['freewayid']}號"
        if(data['from_milepost']!=0):
            lengthOfMilePost = len(str(data['from_milepost']))
            startMile =  str(data['from_milepost']).split(str(data['from_milepost'])[math.ceil(lengthOfMilePost/2)-1])[0]+".+"+ str(data['from_milepost'])[lengthOfMilePost-3] + str(data['from_milepost'])[lengthOfMilePost-2]+ str(data['from_milepost'])[lengthOfMilePost-1]
            
            
        if(data['end_milepost']!=0):
            lengthOfMilePost = len(str(data['end_milepost']))
            endMile =  str(data['end_milepost']).split(str(data['end_milepost'])[math.ceil(lengthOfMilePost/2)-1])[0]+".+"+ str(data['end_milepost'])[lengthOfMilePost-3] + str(data['end_milepost'])[lengthOfMilePost-2]+ str(data['end_milepost'])[lengthOfMilePost-1]
            
        else:
            endMile = data['content'].split(data['content'][data['content'].find("-")])[1]
            
        try:
            url = f"https://tdx.transportdata.tw/api/basic/v2/Road/Link/RoadClass/1/Mileage/{roadName}/0/{startMile}/to/{endMile}?%24format=JSON"
            # print(url)
            # roadInfo = TDX.getData(url)
            
        except Exception as e:
            print(e)
            

        content = {
            "type": "路肩開放",
            "icon": "",
            "content": [
                {
                    "text": ["路肩開放"],
                    "color": ["#FFFFFF"]
                }
            ],
            "voice": "",
            "location": {
                "longitude": data['longitude'],
                "latitude": data['latitude']
            },
            "direction": "",
            "distance": 2.5,
            "priority": "Demo", # Demo
            "start": happenedTime,
            "end": happenedTime + timedelta(minutes = 10),
            "active": True,
            "id": "string"
        }
        documents.append(content)
    return documents
        