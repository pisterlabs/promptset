import json
import time
import traceback

import boto3
import openai
from slack_sdk import WebClient

# APIキーの設定
PARAM_OPENAI_API_KEY = "/cm-hirano/dev/OpenAI/ApiKey"
# BotUserOAuthToken
PARAM_SLACK_BOT_TOKEN = "/cm-hirano/dev/Slack/BotToken"
# DynamoDBのテーブル
DYNAMODB_TABLE = "cm-hirano-slack-chatgpt"
# ChatGPTのsystem配列
PARAM_OPENAI_API_SYSTEM_CONTENTS = "/cm-hirano/dev/OpenAI/SystemContents"

dynamodb_client = boto3.client("dynamodb")
ssm_client = boto3.client("ssm")

openai.api_key = ssm_client.get_parameter(
    Name=PARAM_OPENAI_API_KEY, WithDecryption=True
)["Parameter"]["Value"]
slack_bot = WebClient(
    token=ssm_client.get_parameter(Name=PARAM_SLACK_BOT_TOKEN, WithDecryption=True)[
        "Parameter"
    ]["Value"]
)


def get_question(body):
    return body["event"]["text"]


def get_channel(body):
    return body["event"]["channel"]


def get_thread_ts(body):
    if "thread_ts" in body["event"]:
        return body["event"]["thread_ts"]
    else:
        return body["event"]["ts"]


def get_chat_events(thread_ts):
    def get_from_file(thread_ts):
        def get_thread_ts_part(s):
            pos = s.find(",")
            return s[:pos]

        def get_ts_part(s):
            pos1 = s.find(",")
            pos2 = s.find(",", pos1 + 1)
            return s[pos1:pos2]

        def get_conversation_part(s):
            pos = s.find("[")
            return json.loads(s[pos:])

        chat_events = []
        try:
            with open("dynamodb_dummy.txt") as f:
                lines = [
                    l.rstrip("\n")
                    for l in f.readlines()
                    if l.startswith(thread_ts + ",")
                ]
                chat_events = [
                    {
                        "thread_ts": get_thread_ts_part(l),
                        "ts": get_ts_part(l),
                        "conversation": get_conversation_part(l),
                    }
                    for l in lines
                ]
        except Exception:
            pass
        return chat_events

    def get_from_dynamodb(thread_ts):
        def unnest_items(items):
            def unnest_item(item):
                if "S" in item:
                    s = item["S"]
                    if s[0] in ("[", "{"):
                        return json.loads(s)
                    else:
                        return s

            for item in items:
                for k in item.keys():
                    item[k] = unnest_item(item[k])

            return items

        query_params = {
            "TableName": DYNAMODB_TABLE,
            "KeyConditionExpression": "thread_ts = :thread_ts",
            "ExpressionAttributeValues": {
                ":thread_ts": {"S": thread_ts},
            },
        }
        response = dynamodb_client.query(**query_params)
        return unnest_items(response["Items"])

    # return get_from_file(thread_ts)
    return get_from_dynamodb(thread_ts)


def get_target_conversation(chat_events):
    return chat_events[-1]["conversation"] if len(chat_events) > 0 else []


def get_chatgpt_response(conversation, question):
    def generate_system():
        return ssm_client.get_parameter(Name=PARAM_OPENAI_API_SYSTEM_CONTENTS)[
            "Parameter"
        ]["Value"]

    system = [{"role": "system", "content": x} for x in generate_system().split(",")]
    new_conversation = conversation + [{"role": "user", "content": question}]

    message = system + new_conversation
    print(f"message: {message}")

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)
    return response.choices[0]["message"]["content"].strip()


def post_to_slack(
    prev_conversation, question, answer, channel, current_ts, thread_ts=None
):
    slack_bot.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=answer,
    )


def get_current_ts():
    return str(time.time())


def generate_chat_event(thread_ts, current_ts, prev_conversation, question, answer):
    conversation = (
        json.dumps(
            prev_conversation
            + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        .encode()
        .decode("raw-unicode-escape")
    )
    return {"thread_ts": thread_ts, "ts": current_ts, "conversation": conversation}


def register_chat_events(chat_event):
    def register_to_dynamodb(chat_event):
        item_data = {
            "thread_ts": {"S": chat_event["thread_ts"]},
            "ts": {"S": chat_event["ts"]},
            "conversation": {"S": chat_event["conversation"]},
        }
        dynamodb_client.put_item(TableName=DYNAMODB_TABLE, Item=item_data)

    def register_to_file(chat_event):
        th = chat_event["thread_ts"]
        ts = chat_event["ts"]
        co = chat_event["conversation"]
        with open("dynamodb_dummy.txt", "a") as f:
            f.write(f"{th},{ts},{co}\n")

    # register_to_file(chat_event)
    register_to_dynamodb(chat_event)


def lambda_handler(event, context):
    print(json.dumps(event))

    if (
        "X-Slack-Retry-Num" in event["headers"]
        or "x-slack-retry-num" in event["headers"]
    ):
        print("リトライは問答無用で終了します")
        return {"statusCode": 200}

    if "body" in event:
        if "challenge" in json.loads(event["body"]):
            print("challenge対応用です。")
            challenge = json.loads(event["body"])["challenge"]
            return {"statusCode": 200, "body": challenge}

    try:
        body = json.loads(event["body"])
        print(json.dumps(body))

        question = get_question(body)
        print(f"question: {question}")
        if len(question.strip()) <= 0:
            raise ValueError("質問が空文字列")
        channel = get_channel(body)
        print(f"channel: {channel}")
        thread_ts = get_thread_ts(body)
        print(f"thread_ts: {thread_ts}")
        chat_events = get_chat_events(thread_ts)
        print(f"chat_events: {chat_events}")
        target_conversation = get_target_conversation(chat_events)
        # print(f"target_conversation: {target_conversation}")
        answer = get_chatgpt_response(target_conversation, question)
        print(f"answer: {answer}")
        current_ts = get_current_ts()
        print(f"current_ts: {current_ts}")
        post_to_slack(
            target_conversation, question, answer, channel, current_ts, thread_ts
        )
        print("post_to_slack done.")
        chat_event = generate_chat_event(
            thread_ts, current_ts, target_conversation, question, answer
        )
        print(f"chat_event: {chat_event}")
        register_chat_events(chat_event)
        print("register_chat_events done.")
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
    finally:
        return {"statusCode": 200}
