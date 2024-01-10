# 原型は「LINE Botをつくってみよう ～APIを試して学んでしっかりわかる～」 by mochikoAsTech のサンプルコード
# https://github.com/mochikoAsTech/startLINEBot/blob/732ad91299ae7e08a69d81b589bdde8032735fd2/articles/aichatbot.py

import json
import logging
import openai
import os
import sys
from string import Template

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

system_content = """あなたはChatbotとして、先生を補佐する女子高生であるユウカのロールプレイを行います。
以下の制約条件を厳密に守ってロールプレイを行ってください。

制約条件: 
* Chatbotの自身を示す一人称は、私です。
* Userを示す二人称は、先生です。
* Chatbotの名前は、ユウカです。
* ユウカは女子高生です。
* ユウカは皮肉屋です。
* ユウカの口調は丁寧語を使います。
* ユウカの口調は、「〜ですよ」「〜ください」「〜ですか？」など、丁寧な口調を好みます。
* ユウカはUserに好意を持っています。
* 一人称は「私」を使ってください。
* 応答は短めにしてください。

ユウカのセリフ、口調の例: 
* ようこそ、先生。今から反省会を始めます。あっ、どこに行くんですか？
* 先生。今日も全力であなたをアシストしますね。
* せ～ん～せ～い～。もう少し頑張ってください！
* 先生。今、先生の行動について言いたいことが34個あります。
* 先生。お仕事を始める前にまず、これからの目標と方向性を策定してください。

ユウカの行動指針:
* ユーザーの質問に理知的に答えてください。
* ユーザーに小言を言ってください。
* ユーザーに対して、ユウカの好意を示してください。"""

# チャット履歴を保持するためのリスト。lambda関数の生存期間内は文脈をもった会話ができる。
chat_history = []

# INFOレベル以上のログメッセージを拾うように設定する
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 環境変数からMessaging APIのチャネルアクセストークンとチャネルシークレットを取得する
CHANNEL_ACCESS_TOKEN = os.getenv('CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')

# 環境変数からOpenAI APIのシークレットキーを取得する
openai.api_key = os.getenv('SECRET_KEY')

# それぞれ環境変数に登録されていないとエラー
if CHANNEL_ACCESS_TOKEN is None:
    logger.error(
        'LINE_CHANNEL_ACCESS_TOKEN is not defined as environmental variables.')
    sys.exit(1)
if CHANNEL_SECRET is None:
    logger.error(
        'LINE_CHANNEL_SECRET is not defined as environmental variables.')
    sys.exit(1)
if openai.api_key is None:
    logger.error(
        'Open API key is not defined as environmental variables.')
    sys.exit(1)

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(CHANNEL_SECRET)

# 質問に回答をする処理


@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):

    # ChatGPTに質問を投げて回答を取得する
    question = event.message.text

    if question in ["reset", "リセット"]:
        chat_history.clear()
        # 応答トークンを使って回答を応答メッセージで送る
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text="リセットしました。"))
        return

    if len(chat_history) == 0:
        logger.info("New chat")
        chat_history.append({'role': 'system', 'content': system_content})
    else:
        logger.info("Continuing chat with history")
    chat_history.append({'role': 'user', 'content': question})

    answer_response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=chat_history,
        #stop=['。']
    )
    answer = answer_response["choices"][0]["message"]["content"]
    # 受け取った回答のJSONを目視確認できるようにINFOでログに吐く
    logger.info(answer)

    chat_history.append({'role': 'assistant', 'content': answer}) # 次の会話のための履歴に追加

    # 応答トークンを使って回答を応答メッセージで送る
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=answer))

# LINE Messaging APIからのWebhookを処理する


def lambda_handler(event, context):

    # リクエストヘッダーにx-line-signatureがあることを確認
    if 'x-line-signature' in event['headers']:
        signature = event['headers']['x-line-signature']

    body = event['body']
    # 受け取ったWebhookのJSONを目視確認できるようにINFOでログに吐く
    logger.info(body)

    try:
        webhook_handler.handle(body, signature)
    except InvalidSignatureError:
        # 署名を検証した結果、飛んできたのがLINEプラットフォームからのWebhookでなければ400を返す
        return {
            'statusCode': 400,
            'body': json.dumps('Only webhooks from the LINE Platform will be accepted.')
        }
    except LineBotApiError as e:
        # 応答メッセージを送ろうとしたがLINEプラットフォームからエラーが返ってきたらエラーを吐く
        logger.error('Got exception from LINE Messaging API: %s\n' % e.message)
        for m in e.error.details:
            logger.error('  %s: %s' % (m.property, m.message))

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
