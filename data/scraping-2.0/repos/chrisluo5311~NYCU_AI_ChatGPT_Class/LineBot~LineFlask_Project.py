# 安置CKIP套件並且下載中文Transformer權重檔
from ckiptagger import data_utils
#data_utils.download_data("./")

from flask import Flask, request, abort
from ckiptagger import WS
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

# 建立OpenAI的連線與ChatGPT的問答連線 & 設定OpenAI金鑰
import openai
key = ""
openai.api_key = key

def chatgpt_qa(q):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=q,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )
    # 返回答案
    return response["choices"][0]["text"].strip()

#ws = WS("./data")

# access_token
configuration = Configuration(
    access_token='')
# CHANNEL_SECRET
handler = WebhookHandler('')

app = Flask(__name__)

@app.route("/test",methods=['GET'])
def test():
    return 'test'

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=event.message.text)]
            )
        )



if __name__ == "__main__":
    app.run(port=8080)
