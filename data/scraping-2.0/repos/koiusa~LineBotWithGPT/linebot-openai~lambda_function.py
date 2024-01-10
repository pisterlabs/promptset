import os
import sys
import logging
import openai
import status
import behavior
import guard

from linebot import (LineBotApi, WebhookHandler)
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, StickerMessage)
from linebot.exceptions import (LineBotApiError, InvalidSignatureError)

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# 環境変数からLINEBotのチャンネルアクセストークンとシークレットを読込
# 環境変数からChatGpt APIの鍵を読込
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
openai.api_key = os.getenv("OPENAI_API_KEY")

# トークンが確認できない場合エラー出力
if channel_secret is None:
    logger.error('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    logger.error('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

# apiとhandlerの生成（チャンネルアクセストークンとシークレットを渡す）
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# Lambdaのメインの動作


def lambda_handler(event, context):
    authorizer = guard.authorizer()
    if not authorizer.request(event):
        return status.const.forbidden_json

    @handler.add(MessageEvent, message=(TextMessage, StickerMessage, ImageMessage))
    def message(line_event):
        eventcontext = behavior.eventcontext(
            event=line_event, linebot=line_bot_api)
        behavior.completion.reply(event_context=eventcontext)

# 例外処理としての動作
    try:
        handler.handle(authorizer.body, authorizer.signature)
    except LineBotApiError as e:
        logger.error("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            logger.error("  %s: %s" % (m.property, m.message))
        return status.const.error_json
    except InvalidSignatureError:
        return status.const.error_json

    return status.const.ok_json
