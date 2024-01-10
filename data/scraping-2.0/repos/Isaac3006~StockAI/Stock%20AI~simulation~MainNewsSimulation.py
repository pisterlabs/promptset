import sys
sys.path.insert(0, r"C:\Users\isaac\OneDrive\Desktop\Todos\Codes\Real world projects\Stock AI")
import requests
import os
import openai
import time
import datetime
import json 
from googletrans import Translator
import AI
import google.generativeai as palm
import asyncio
import yfinance as yf
from polygon import Polygon
import newsSim
import main


# delayBetweenChecks= 1

# MICROSECOND_TO_SECOND = 1000000

recordFile = r"C:\Users\isaac\OneDrive\Desktop\Todos\Codes\Real world projects\Stock AI\simulation\records.csv"

howOldCanNewsBe = 15

brokerage = Polygon()


translator = Translator()

today = datetime.datetime(2023, 6, 30) # 2023-05-17


seen = set()

prof = 0

# messages=[
#         {"role": "user", "content": "hi"}

#     ]

async def main():
    global today
    global brokerage
    global seen
    global prof
 
    # while True:

    file = open(recordFile, "a")
    for _ in range(5):

        


        # start = datetime.datetime.now()

        news = newsSim.getNews(today)

        for title, date in news:
            if not isNew(date) or title in seen:
                continue

            seen.add(title)



            title = f""" \" {title} \" """

            gpt, bard = await AI.analyze(title)

            stocksAffected = {}


            recordTransactions(title, gpt, bard, stocksAffected, None, file)

        # end = datetime.datetime.now()


        # wait = delayBetweenChecks - (end - start).microseconds / MICROSECOND_TO_SECOND

    file.close()

        
    
    today += datetime.timedelta(days=1)
        # time.sleep(max(wait * 60, 0))


def recordTransactions(title, gptResponse, bardResponse, stocksAffected, profit, file):

    title = title[3:-3]

    

    try:
        trans = translator.translate(title)
        print(f"{profit},{trans.text},{stocksAffected},{gptResponse},{bardResponse}\n")
        file.write(f"{profit},{trans.text},{stocksAffected},{gptResponse},{bardResponse}\n")
    except:
        pass


    try:
        print(f"{profit},{title},{stocksAffected},{gptResponse},{bardResponse}\n")
        file.write(f"{profit},{title},{stocksAffected},{gptResponse},{bardResponse}\n")
    except:
        pass
def isNew(news):

    try:


        released = newsSim.convertStringToDateTime(news)




        return (today - released).total_seconds() // 60 < howOldCanNewsBe
    
    except:

        return False
    
def actions(gptResponse, bardResponse):

    res = []


    for stock, change in gptResponse.items():

        if stock in bardResponse and bardResponse[stock] == change:

           res.append([stock, change])


    

    return res


asyncio.run(main())