import json
import logging
from typing import TypedDict

import boto3
import openai
from linebot import LineBotApi, WebhookHandler, WebhookPayload
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    Event,
    JoinEvent,
    LeaveEvent,
    MessageEvent,
    TextMessage,
    TextSendMessage,
)
from tweepy import API, OAuth2BearerHandler

from .env import Env, get_env
from .lambda_types import LambdaContext, LambdaResponse

# loggerの設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)

RequestHeadersFromLineBot = TypedDict(
    "RequestHeadersFromLineBot",
    {"x-line-signature": str, "X-Line-Signature": str},
    total=False,
)


class RequestFromLineBot(TypedDict):
    headers: RequestHeadersFromLineBot
    body: WebhookPayload


def lambda_handler(
    request: RequestFromLineBot, context: LambdaContext
) -> LambdaResponse:
    ok_json = LambdaResponse(
        {
            "isBase64Encoded": False,
            "statusCode": 200,
            "headers": {},
            "body": "",
        }
    )
    error_json = LambdaResponse(
        {
            "isBase64Encoded": False,
            "statusCode": 500,
            "headers": {},
            "body": "Error",
        }
    )

    env = get_env()
    handler = WebhookHandler(env.LINE_CHANNEL_SECRET)

    @handler.add(MessageEvent, message=TextMessage)
    def handler_message_text(event: Event) -> None:
        text = event.message.text
        reply_token = event.reply_token
        message(text, reply_token, env)

    @handler.add(JoinEvent)
    def handler_join(event: Event) -> None:
        sender_id = event.source.sender_id
        store_id(sender_id, env)

    @handler.add(LeaveEvent)
    def handler_leave(event: Event) -> None:
        sender_id = event.source.sender_id
        delete_id(sender_id, env)

    body = request["body"]
    signature = ""
    if "x-line-signature" in request["headers"]:
        signature = request["headers"]["x-line-signature"]
    elif "X-Line-Signature" in request["headers"]:
        signature = request["headers"]["X-Line-Signature"]

    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        logger.error("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            logger.error("  %s: %s" % (m.property, m.message))
        return error_json
    except InvalidSignatureError:
        logger.error("Detected invalid signature")
        return error_json

    return ok_json


def message(text: str, reply_token: str, env: Env) -> None:
    if text == "今週の聖堂":
        auth = OAuth2BearerHandler(env.TWITTER_BEARER_TOKEN)
        twitter_api = API(auth)

        status_list = twitter_api.user_timeline(
            screen_name="DeadbyBHVR_JP",
            count=50,
            tweet_mode="extended",
            exclude_replies=True,
            include_rts=False,
        )
        output_list = list(
            filter(lambda x: "シュライン・オブ・シークレット" in x.full_text, status_list)
        )

        if not output_list:
            return

        reply_line(output_list[0].full_text, reply_token, env)

    if (text_list := text.split())[0] == "/chatgpt":
        if len(text_list) == 1:
            return

        openai.api_key = env.OPENAI_API_KEY

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": " ".join(text_list[1:])},
            ],
        )  # type: ignore

        reply_line(response["choices"][0]["message"]["content"], reply_token, env)


def reply_line(reply: str, reply_token: str, env: Env) -> None:
    messages = []
    messages.append(TextSendMessage(text=reply))

    line_bot_api = LineBotApi(env.LINE_CHANNEL_ACCESS_TOKEN)
    line_bot_api.reply_message(reply_token, messages=messages)


def store_id(sender_id: str, env: Env) -> None:
    logger.info("参加先id: " + sender_id)

    s3 = boto3.resource("s3")
    obj = s3.Object(env.S3_BUCKET_NAME, env.S3_KEY_NAME)
    ids = json.loads(obj.get()["Body"].read())

    if sender_id in ids:
        logger.info("id重複のため書き込まない")
    else:
        ids.append(sender_id)
        response = obj.put(Body=json.dumps(ids))
        logger.info(response)


def delete_id(sender_id: str, env: Env) -> None:
    logger.info("退室先id: " + sender_id)

    s3 = boto3.resource("s3")
    obj = s3.Object(env.S3_BUCKET_NAME, env.S3_KEY_NAME)
    ids = json.loads(obj.get()["Body"].read())

    if sender_id in ids:
        new_ids = list(filter(lambda x: not x == sender_id, ids))
        response = obj.put(Body=json.dumps(new_ids))
        logger.info(response)
    else:
        logger.info("idが無いため削除しない")
