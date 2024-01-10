import os
import re
import typing

import openai
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

openai.api_key = os.getenv("OPENAI_API_KEY")


# 愚直にconnector.pyをコピペして試してみる
def send_text(text: str, model="gpt-3.5-turbo") -> openai.openai_object.OpenAIObject:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": text},
        ],
    )

    return response


def send_messages(
    messages: typing.List[dict], *, model="gpt-3.5-turbo", max_tokens=1024
) -> openai.openai_object.OpenAIObject:
    """list型のメッセージをopen ai apiに投げ込む.
    messagesは
    [
    {
        "role": "system" or "user" or "assistant",
        "content": "text"
    },
    {
        ...
    }, ...
    ]
    のような並び.
    "role"は、systemで恒久的なルールを、userでopenaiに聞いたtextを、assistantでoepnaiから返答をいれることで、会話の履歴を再現できる。

    Args:
        messages (typing.List[dict]): _description_
        model (str, optional): _description_. Defaults to "gpt-3.5-turbo".
        max_tokens (int, optional): _description_. Defaults to 1024.

    Returns:
        openai.openai_object.OpenAIObject: _description_
    """

    response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens)

    return response


def response_to_text(response: openai.openai_object.OpenAIObject) -> str:
    usd_to_jpg = 140
    token_cost = response["usage"]["total_tokens"]
    usd_cost = response["usage"]["total_tokens"] / 1000 * 0.002
    jpy_cost = usd_cost * usd_to_jpg

    text = response["choices"][0]["message"]["content"] + "\n"
    # cost_text = f"ちなみに、このテキストを作るのに{token_cost}トークン使用し、{jpy_cost}円かかりました。({usd_to_jpg}yen/usd換算)" + "\n"
    # text += cost_text
    print(f"token:{token_cost} jpy:{jpy_cost}")

    return text


# 愚直にconnector.pyをコピペして試すところ終わり


app = App(
    process_before_response=True,
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)
slack_handler = SlackRequestHandler(app=app)

bot_user_id = app.client.auth_test()["user_id"]
print(f"bot_id:{bot_user_id}")


def remove_mention(message: str) -> str:
    m = re.match("<@.*>", message)
    if m is not None:
        return message[m.span()[1] :]
    return message


# スレッドを作成する関数
def reply_thread(channel_id, thread_ts, text):
    app.client.chat_postMessage(channel=channel_id, text=text, thread_ts=thread_ts)


event_ts_list = []


@app.event("app_mention")
def handle_mention(event, say):
    print(event)
    user_id = event["user"]
    channel_id = event["channel"]

    global event_ts_list
    event_ts = event["event_ts"]
    if event_ts in event_ts_list:
        print("duplicat event")
        return

    # eventの起動リストを最大100個保持しておく
    event_ts_list.append(event_ts)
    event_ts_list = event_ts_list[-100:]

    # スレッドのタイムスタンプを取得する
    thread_ts = event["thread_ts"] if "thread_ts" in event else event["ts"]

    response = app.client.conversations_replies(channel=channel_id, ts=thread_ts)
    messages = response["messages"]

    # 指定されたユーザによるメッセージを取得する
    message_prompt = []
    for message in messages:
        if message.get("user") != bot_user_id:
            message_prompt.append({"role": "user", "content": remove_mention(message["text"])})
        else:
            message_prompt.append({"role": "assistant", "content": remove_mention(message["text"])})

    system_prompt = [
        {
            "role": "system",
            "content": "最新のメッセージ以外は会話の履歴です。roleがuserのものはユーザーが発したもので、roleがassistantのものは、chatgptからの返答になっています。会話履歴を踏まえたうえで、最新のメッセージに回答してください。",
        }
    ]

    message_list = system_prompt + message_prompt
    print(message_list)

    try:
        response = send_messages(message_list)
        response_text = response_to_text(response)
        # スレッドで返信する
        response_text = f"<@{user_id}> {response_text}"
        reply_thread(channel_id, thread_ts, response_text)
    except Exception as e:
        say(f"なんかエラーだって {e}")


def handler(event, context):
    print(event)
    print(context)
    return slack_handler.handle(event, context)
