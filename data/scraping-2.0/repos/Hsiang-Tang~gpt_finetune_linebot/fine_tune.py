# gpt-3.5_fine_tune

#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import TextMessage, MessageEvent, TextSendMessage
import os
import openai
import tempfile
import datetime
import time
import string







import os

# 升級 openai 庫
os.system('pip install openai --upgrade')

# 使用 curl 下載 clinic_qa.json 文件
os.system('curl -o clinic_qa.json -L https://github.com/Hsiang-Tang/gpt-3.5_fine_tune/raw/main/clinic_qa.json')

import openai

# 設置您的 OpenAI API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")

# 創建 fine-tune 文件
openai.File.create(
  file=open("clinic_qa.json", "rb"),
  purpose='fine-tune'
)

# 列出文件
openai.File.list()

# 創建 fine-tuning 作業
openai.FineTuningJob.create(training_file="file-BhA5o5gmQRCx15KsK4zq97WI", model="gpt-3.5-turbo")

# 列出 fine-tuning 作業
openai.FineTuningJob.list(limit=10)

# 檢索 fine-tuning 作業事件
openai.FineTuningJob.retrieve("ftjob-x9NGckMPlxEXEXGSDtjy4Uc0")

# 列出 fine-tuning 作業事件
openai.FineTuningJob.list_events(id="ftjob-x9NGckMPlxEXEXGSDtjy4Uc0", limit=10)

# 創建聊天完成
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "您現在扮演一個專業的醫生"},
    {"role": "user", "content": "我感覺動一動就很累，老是很疲倦"}
  ]
)

print(completion.choices[0].message.content)

# 創建帶有 fine-tuned 模型的聊天完成
completion2 = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:personal::7wllb3DZ",
  messages=[
    {"role": "system", "content": "您現在扮演一個專業的醫生"},
    {"role": "user", "content": "我感覺動一動就很累，老是很疲倦"}
  ]
)

print(completion2.choices[0].message.content)


def GPT_response(text):
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0613:personal::7wllb3DZ",
        messages=[
            {"role": "system", "content": "您現在扮演一個專業的醫生"},
            {"role": "user", "content": text}
        ]
    )

    answer = response.choices[0].message.content

    # 去除回复文本中的標點符號
    answer = answer.translate(str.maketrans('', '', string.punctuation))

    return answer