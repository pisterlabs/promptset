import openai
import os

#這些是LINE官方開放的套件組合透過import來套用這個檔案上
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import *

 
def gptapi():
    # 设置OpenAI API密钥
# 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
    api_key = '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'  #os.environ['OPENAI_API_KEY']
    print(api_key)
    openai.api_key = api_key
# 撰寫輸入的提示語句
    prompt = "請列舉出糖尿病患者的飲食禁忌："
    prompt = "請列舉出MyDAILYCHOICE 公司的產品項目："

    # 設定要求模型輸出的最大字數
    max_tokens = 256

    # 設定輸出結果的格式
    model_engine = "text-davinci-002"

    # 使用 OpenAI API 輸出模型預測的結果
    output = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.5,
        n=1,    
        stop=None,
    )

# 從輸出結果中取得模型預測的文本內容
    result = output.choices[0].text.strip()
    print(result)  
    return output.choices[0].text

#TemplateSendMessage - ButtonsTemplate (按鈕介面訊息)
