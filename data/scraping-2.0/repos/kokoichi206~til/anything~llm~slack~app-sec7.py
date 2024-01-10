import os
import re
import time
from typing import Any
from datetime import timedelta

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.memory import MomentoChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

import langchain as lc
print(lc.__version__)


app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    # FaaS では HTTP レスポンスの送信を返した後にスレッドやプロセスを続けられない。
    # 応答を別インスタンスで実行可能にする。
    process_before_response=True,
)

CHAT_UPDATE_INTERVAL_SEC = 1

class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.message += token

        now = time.time()
        if now - self.last_send_time > CHAT_UPDATE_INTERVAL_SEC:
            self.last_send_time = now
            app.client.chat_update(
                channel=self.channel,
                ts=self.ts,
                text=f"{self.message}...",
            )

    def on_llm_end(self, response: LLMResult, **keywargs: Any) -> None:
        # app.client.chat_update(
        #     channel=self.channel,
        #     ts=self.ts,
        #     text=self.message,
        # )
        message_context = "OpenAI API で生成される情報は不正確または不適切な場合があります。"
        message_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{self.message}",
                },
            },
            {
                "type": "divider",
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": message_context,
                    }
                ]
            },
        ]
        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
            blocks=message_blocks,
        )

# @app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    # user = event["user"]
    thread_ts = event["ts"]
    message = re.sub("<@.*>", "", event["text"])

    # 投稿キー (=Momento key)
    # 初回　= event["ts"], 2 回目以降 = event["thread_ts"]
    id_ts = event["ts"]
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"]

    history = MomentoChatMessageHistory.from_client_params(
        id_ts,
        os.environ["MOMENT_CACHE"],
        timedelta(hours=int(os.environ["MOMENTO_TTL"]))
    )

    messages = [SystemMessage(content="You are a good assistant.")]
    messages.extend(history.messages)
    messages.append(HumanMessage(content=message))
    print('----------------- messages ---------------------')
    print(messages)

    history.add_user_message(message)

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)
    llm = ChatOpenAI(
        model_name=os.environ["OPENAI_MODEL_NAME"],
        temperature=os.environ["OPENAI_TEMPERATURE"],
        streaming=True,
        callbacks=[callback]
    )

    ai_message = llm(messages)
    print("------------------- ai_message -------------------")
    print(ai_message)
    print(type(ai_message))
    history.add_ai_message(ai_message.content)

def just_ack(ack):
    ack()

app.event("app_mention")(ack=just_ack, lazy=[handle_mention])


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
