from celery import shared_task

from django.utils import timezone
from django.conf import settings

from .models import chat_record, chat_record_mention

import time, datetime
import re

from linebot import LineBotApi
from linebot.models import MessageEvent, TextSendMessage, TextMessage
from linebot.exceptions import LineBotApiError
Line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)

# 放在全域變數，會出大事，openai和azure的api有重複的參數名稱，待處理
# 在其他檔案全域宣告version會被讀到，openai版不支援此參數
import openai

# OpenAI API
# openai.api_type = 'open_ai'
# openai.api_key = settings.OPENAI_OPENAI_API_KEY

# Azure OpenAI API
# openai.api_type = 'azure'
# openai.api_key = settings.AZURE_OPENAI_API_KEY
# openai.api_base = settings.AZURE_OPENAI_ENDPOINT
# openai.api_version = '2023-05-15' # this may change in the future
# AZURE_OPENAI_DEPLOYMENT_NAME = settings.AZURE_OPENAI_DEPLOYMENT_NAME

@shared_task
def async_func():
    # 非同步操作
    print('start sleep')
    time.sleep(5)
    print('end sleep')
    return 'Hello, world!'

@shared_task
def save_chat_record(event_dict):
    mention = bool(event_dict['message'].get('mention', []))

    if str.startswith(event_dict['message']['text'], 'http'):
        # 網址訊息，不存入不處理
        return False

    # 處理訊息
    filtered_message = event_dict['message']['text']

    # # 刪除從@字元開頭到空格結尾的文字 ，方法太粗暴已棄用
    # regex = r"@\S+\s?"
    # filtered_message = re.sub(regex, "", event_dict['message']['text']).strip()

    # 處理 @標註
    if mention:
        mentionees = event_dict['message']['mention']['mentionees']
        filtered_message_list = list(filtered_message)

        for mentionee in mentionees:
            start_index = mentionee['index']
            end_index = start_index + mentionee['length']

            # 將被提及用戶名替換為空字串
            for i in range(start_index, end_index):
                filtered_message_list[i] = " "

        filtered_message = "".join(filtered_message_list).strip()
        # 訊息是:@名字 444 @名字 555
        # 輸出是:444     555

    # 處理表情貼
    regex_emoji = r"\([\w\s]+\)"
    filtered_message = re.sub(regex_emoji, "", filtered_message).strip()

    # Line給的timestamp是以毫秒表示的，要轉換成datetime
    timestamp = datetime.datetime.fromtimestamp(event_dict['timestamp'] / 1000.0)
    # 掛上Django設定的位置時區
    aware_timestamp = timezone.make_aware(timestamp)

    created_chat_record = chat_record.objects.create(
        userId=event_dict['source']['userId'],
        groupId=event_dict['source']['groupId'],
        message=event_dict['message']['text'],
        filtered_message=filtered_message if filtered_message else None,
        mention=mention,
        timestamp=aware_timestamp
    )

    if mention:
        for event_mention in event_dict['message']['mention']['mentionees']:
            chat_record_mention.objects.create(
                chat_record=created_chat_record,
                mentioned_userId=event_mention.get('userId', None)
            )
    return "存入資料庫"

@shared_task
def OpenAI_API_text_embedding(event_dict, model="text-embedding-ada-002"):

    openai.api_type = 'open_ai'
    openai.api_base = 'https://api.openai.com/v1'
    openai.api_key = settings.OPENAI_OPENAI_API_KEY
    print(openai.api_base)

    response = openai.Embedding.create(
        model=model,   
        input=event_dict['message']['text'],
    )
    Line_bot_api.reply_message(event_dict['replyToken'], TextSendMessage(text= response['model'] + "，總token: " + str(response['usage']['total_tokens'])))
    return "消耗token: " + str(response['usage']['total_tokens'])

@shared_task
def Azure_openAI_gpt35turbo(event_dict):

    openai.api_type = 'azure'
    openai.api_key = settings.AZURE_OPENAI_API_KEY
    openai.api_base = settings.AZURE_OPENAI_ENDPOINT
    openai.api_version = '2023-05-15' # this may change in the future
    response = openai.ChatCompletion.create(
        engine= settings.AZURE_OPENAI_DEPLOYMENT_NAME, # deployment_name
        max_tokens = 30,
        temperature = 0.2,
        messages=[
            {'role': 'system', 'content': '你是一位負責審查言論的分析人員，負責解析收到的文本，其中可能包含色情、或辱罵言語，你將會提取這些詞語出來並用python字典格式表示。如果文本不存在這些字詞或你不確定該詞語是否不適當，則回答：沒有粗鄙言論。'},
            {'role': 'user', "content": '"你幫我素完再幫我打手槍"'},
            {'role': 'assistant', 'content': '{"幫我素", "打手槍"}'},
            {'role': 'user', "content": '"要先把楊芷昀叫進來"'},
            {'role': 'assistant', 'content': '沒有粗鄙言論。'},
            {'role': 'user', "content": '"他媽的祖墳燒起來了"'},
            {'role': 'assistant', 'content': '{"他媽的", "祖墳"}'},
            {'role': 'user', "content": '"你直播吃ㄐㄐ比較快"'},
            {'role': 'assistant', 'content': '{"ㄐㄐ"}'},
            {'role': 'user', 'content': '"' + event_dict['message']['text'] + '"'}
        ]
    )

    if response.choices[0].finish_reason != 'stop':
        Line_bot_api.reply_message(event_dict['replyToken'], TextSendMessage(text="Azure OpenAI API失敗，原因: " + response.choices[0].finish_reason))
        return response.choices[0].finish_reason

    Line_bot_api.reply_message(event_dict['replyToken'], TextSendMessage(text= "原文: " + event_dict['message']['text'] + "，識別: "+ response.choices[0].message.content))
    return "消耗token: " + str(response['usage']['total_tokens'])