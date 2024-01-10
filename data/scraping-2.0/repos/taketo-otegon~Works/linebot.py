from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)

from time import time
from os import environ
import boto3
from openai import OpenAI
import json
import logging

if not (access_token := environ.get("LINE_CHANNEL_ACCESS_TOKEN")):
    raise Exception("access token is not set as an environment variable")

if not (channel_secret := environ.get("LINE_CHANNEL_SECRET")):
    raise Exception("channel secret is not set as an environment variable")

if not (api_key := environ.get("OPENAI_API_KEY")):
    raise Exception("openai api key is not set as an environment variable")

#initializer
client = OpenAI(api_key=api_key)
configuration = Configuration(access_token = access_token)
handler = WebhookHandler(channel_secret)
dynamo = boto3.client('dynamodb', region_name='ap-southeast-2')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def store_conversation(user_id, openai_response):
    print('store_conversation')
    try:
        dynamo.put_item(
        TableName='line-gpt-test',
        Item={
            'user_id': {
                'S': user_id,
            },
            'conversation': {
                'S': openai_response,
            }
        }
    )
    except:
        print('failed to store conversation')

def lambda_handler(event, context):
    # TODO implement
    # logger.info(event)
    signature = event['headers']['x-line-signature']
    body = event['body']
    global user_id
    user_id = json.loads(event['body'])['events'][0]['source']['userId']
      # handle webhook body
    try:
        handler.handle(body, signature)
    except Exception as e:
        print(e)  # エラーをログに記録
        return {'statusCode': 500, 'body': 'Error'}

    return {'statusCode': 200, 'body': 'OK'}

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": event.message.text,
            }
        ],
        model="gpt-4-1106-preview",
        )
        logger.info(chat_completion)
        openai_response = chat_completion.choices[0].message.content
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=openai_response)]
            )
        )
        store_conversation(user_id,openai_response)
