from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent
from linebot.models import TextMessage, TextSendMessage
from linebot.models import TemplateSendMessage, ButtonsTemplate, MessageTemplateAction

import os, sys
import random
import time
import subprocess
import openai

linebot_client = LineBotApi("/diBCY/NIHQdUmXML33amlI/a6j8JQva55yTjB4RjTkBjckI2PxItJRh8vRk1n9Eo6fVJoTFCX+aBDBQvRnJtfrV1KUUbMVTXCvdts0AoRC30Mbe2rDe3GRGQksXVwxjqfK6NdWOHdnQPcjGa8+SFAdB04t89/1O/w1cDnyilFU=")
linebot_handler = WebhookHandler("3a7f225af653746219179110ad663f74")

app = Flask(__name__)

@app.route('/callback', methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        linebot_handler.handle(body, signature)

    except InvalidSignatureError:
        abort(400)

    return 'ok'


@linebot_handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):

    if event.reply_token == '0' * 32:
        return

    time.sleep(1.5)

    if event.message.text == "Basic":
        content = "You choose Basic,\nthe test result will be report by line notify."
        linebot_client.reply_message(
            event.reply_token,
            TextSendMessage(content)
        )
        os.system("cd /home/jim/AutoTestSuite/ExpectScriptAutoTest")
        os.system("sh /home/jim/AutoTestSuite/ExpectScriptAutoTest/feature/Basic_System/basicsystem_auto_test_linedemo.sh")

    elif event.message.text == "MMS":
        content = "You choose MMS,\nthe test result will be report by line notify."
        linebot_client.reply_message(
            event.reply_token,
            TextSendMessage(content)
        )
        os.system("cd /home/jim/AutoTestSuite/ExpectScriptAutoTest/feature/MMS")
        os.system("sh /home/jim/AutoTestSuite/ExpectScriptAutoTest/feature/MMS/mms_auto_test_linedemo.sh")
    elif event.message.text == "Dot1x":
        content = "You choose Dot1x,\nthe test result will be report by line notify."
        linebot_client.reply_message(
            event.reply_token,
            TextSendMessage(content)
        )
        os.system("cd /home/jim/AutoTestSuite/ExpectScriptAutoTest/feature/Dot1x")
        os.system("sh /home/jim/AutoTestSuite/ExpectScriptAutoTest/feature/Dot1x/dot1x_auto_test_linedemo.sh")
    elif event.message.text == 'hello':
        linebot_client.reply_message(
            event.reply_token,
            TemplateSendMessage(
                alt_text='Auto Test template',
                template=ButtonsTemplate(
                    title='Auto Test',
                    text='Please choose test item',
                    thumbnail_image_url='https://www.moxa.com/Moxa/media/Global/moxa-logo-open-graph.png?ext=.png',
                    actions=[
                        MessageTemplateAction(
                            label='Basic',
                            text='Basic'
                        ),
                        MessageTemplateAction(
                            label='MMS',
                            text='MMS'
                        ),
                        MessageTemplateAction(
                            label='Dot1x',
                            text='Dot1x'
                        )
                    ]
                )
            )
        )
    else:
        openai.api_key = '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'
        response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=event.message.text,
                max_tokens=256,
                temperature=0.5
                )
        reply_msg = response["choices"][0]["text"].replace('\n','')
        linebot_client.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_msg)
                )
        pass


if __name__ == '__main__':
    app.run(debug=True)
