import os
import re

import requests
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from utility import SlackCallbackHandler

load_dotenv()


class CoPyBotMd:
    def __init__(self):
        self.slack_app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
        self.register_listeners()

    def create_chain(self, llm):
        system_template = """
        You are an assistant who thinks step by step and includes a thought path in your response.
        Your answers are in Japanese.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = """
        以下の文章を要約してね。
        なお、要約は箇条書きでお願い。
        また、項目立てをして作成してね。
            {md_content}
        """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt.input_variables = ["md_content"]

        chain = LLMChain(llm=llm, prompt=chat_prompt)

        return chain

    def get_llm(self, say):
        def say_function(message):
            say(message)

        callback_manager = CallbackManager([SlackCallbackHandler(say_function)]) if self.is_streaming else None

        return ChatOpenAI(temperature=0,
                          openai_api_key=os.environ.get("OPENAI_API_KEY"),
                          model_name="gpt-3.5-turbo",
                          streaming=self.is_streaming,
                          callback_manager=callback_manager)

    def register_listeners(self):
        @self.slack_app.message(re.compile("(マークダウン要約|よろしく)"))
        def message_streamling_mode_selection(say):
            text = "こんにちは。co-py-bot だよ。\nストリーミングモードで実行する？"
            say(
                blocks=[
                    {
                        "type": "section",
                        "block_id": "section677",
                        "text": {"type": "mrkdwn", "text": text},
                        "accessory": {
                            "action_id": "mode_selection",
                            "type": "static_select",
                            "placeholder": {"type": "plain_text", "text": "選択してください"},
                            "options": [{"text": {"type": "plain_text", "text": "はい"}, "value": "1"}, {"text": {"type": "plain_text", "text": "いいえ"}, "value": "0"}],
                        },
                    }
                ],
                text=text,
            )

        @self.slack_app.action("mode_selection")
        def message_ryokai(body, ack, say):
            ack()
            self.is_streaming = int(body["actions"][0]["selected_option"]["value"])
            if self.is_streaming:
                say("了解。ストリーミングモードで実行するね。\n準備はできたよ。要約して欲しいマークダウンファイル(.md)を投稿してね。")
            else:
                say("了解。ストリーミングモードはオフで実行するね。\n準備はできたよ。要約して欲しいマークダウンファイル(.md)を投稿してね。")

        @self.slack_app.event('message')
        def handle_file_share_events(body, say, client):
            try:
                self.is_streaming
            except AttributeError:
                self.is_streaming = 0
            say("ファイルを受け取ったよ。なかなか良い文章だね。要約しているからちょっと待ってね。")
            event = body['event']
            self.process_file_share(event, say, client)

    def process_file_share(self, event, say, client):
        try:
            file_info = client.files_info(file=event['files'][0]['id']).data['file']
            if file_info['name'][-3:] == '.md':
                file_url = file_info['url_private_download']
                header = {'Authorization': f"Bearer {os.environ.get('SLACK_BOT_TOKEN')}"}
                file_res = requests.get(file_url, headers=header)
                if file_res.status_code != 200:
                    print(f"Failed to download file: status code {file_res.status_code}")
                    print(f"Response body: {file_res.text}")
                    return
                md_content = file_res.text

                self.chain = self.create_chain(self.get_llm(say))
                summary = self.chain.run(md_content=md_content[:3950])
                say(summary)
            else:
                say("指定されたファイルが .md 形式ではありません。.md ファイルをアップロードしてください。")

        except SlackApiError as e:
            print(f"Error getting file info: {e}")

    def start(self):
        SocketModeHandler(self.slack_app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    bot = CoPyBotMd()
    bot.start()
