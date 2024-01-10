import logging
from connectai.lark.websocket import *
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler


class TextMessageBot(Bot):

    def __init__(self, app=None, *args, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def on_message(self, data, *args, **kwargs):
        if 'header' in data:
            if data['header']['event_type'] == 'im.message.receive_v1' and data['event']['message']['message_type'] == 'text':
                content = json.loads(data['event']['message']['content'])
                if self.app:
                    return self.app.process_text_message(text=content['text'], **data['event']['sender']['sender_id'], **data['event']['message'])
        logging.warn("unkonw message %r", data)


class OpenAICallbackHandler(BaseCallbackHandler):
    def __init__(self, bot, message_id):
        self.bot = bot
        self.message_id = message_id
        self.result = ''
        self.send_length = 0
        self.reply_message_id = ''

    def on_llm_start(self, *args, **kwargs):
        response = self.bot.reply_card(
            self.message_id,
            FeishuMessageCard(
                FeishuMessageDiv(''),
                FeishuMessageNote(FeishuMessagePlainText('正在思考，请稍等...'))
            )
        )
        self.reply_message_id = response.json()['data']['message_id']

    def on_llm_new_token(self, token, **kwargs):
        logging.info("on_llm_new_token %r", token)
        self.result += token
        if len(self.result) - self.send_length < 25:
            return
        self.send_length = len(self.result)
        self.bot.update(
            self.reply_message_id,
            FeishuMessageCard(
                FeishuMessageDiv(self.result, tag="lark_md"),
                FeishuMessageNote(FeishuMessagePlainText('正在生成，请稍等...'))
            )
        )

    def on_llm_end(self, response, **kwargs):
        content = response.generations[0][0].text
        logging.info("on_llm_end %r", content)
        self.bot.update(
            self.reply_message_id,
            FeishuMessageCard(
                FeishuMessageDiv(content, tag="lark_md"),
                FeishuMessageNote(FeishuMessagePlainText("reply from openai.")),
            )
        )


class Session(object):
    store = {}
    def __init__(self, app_id, user_id):
        self.key = f"{app_id}:{user_id}"
        self.data = self.store.get(self.key, dict(
            chat_history=[], temperature=0.7,
            system_role='', model='gpt-3.5-turbo'
        ))

    def __getattr__(self, name):return self.data.get(name)
    def __enter__(self): return self
    def __exit__(self, *args):
        self.store[self.key] = self.data


class Application(object):

    def __init__(self, openai_api_base='', openai_api_key='', system_role='', temperature=0.7, streaming=True, **kwargs):
        self.system_role = system_role
        # self.bot.app = self
        self.bot = TextMessageBot(app=self, **kwargs)
        self.temperature = temperature
        self.openai_options = dict(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            streaming=streaming,
        )

    def process_text_message(self, text, message_id, open_id, **kwargs):
        with Session(self.bot.app_id, open_id) as session:
            if text == '/help' or text == '帮助':
                self.bot.reply_card(
                    message_id,
                    FeishuMessageCard(
                        FeishuMessageDiv('👋 你好呀，我是一款基于OpenAI技术的智能聊天机器人'),
                        FeishuMessageHr(),
                        FeishuMessageDiv('👺 **角色扮演模式**\n文本回复*/system*+空格+角色信息', tag='lark_md'),
                        FeishuMessageHr(),
                        FeishuMessageDiv('🎒 **需要更多帮助**\n文本回复 *帮助* 或 */help*', tag='lark_md'),
                        header=FeishuMessageCardHeader('🎒需要帮助吗？'),
                    )
                )
            elif text[:7] == '/system' and text[7:]:
                session.data['system_role'] = text[7:]
                session.data['chat_history'] = []
                self.bot.reply_card(
                    message_id,
                    FeishuMessageCard(
                        FeishuMessageDiv('请注意，这将开始一个全新的对话'),
                        header=FeishuMessageCardHeader('👺 已进入角色扮演模式'),
                    )
                )
            elif text:
                chat = ChatOpenAI(
                    callbacks=[OpenAICallbackHandler(self.bot, message_id)],
                    temperature=session.temperature or self.temperature,
                    model=session.model,
                    **self.openai_options
                )
                system_role = session.system_role or self.system_role
                system_message = [SystemMessage(content=system_role)] if system_role else []
                messages = system_message + session.chat_history + [HumanMessage(content=text)]
                message = chat(messages)
                # save chat_history
                session.chat_history.append(HumanMessage(content=text))
                session.chat_history.append(message)
                logging.info("reply message %r\nchat_history %r", message, session.chat_history)
            else:
                logging.warn("empty text", text)


if __name__ == "__main__":
    import click
    @click.command()
    @click.option('--openai_api_base', prompt="OpenAI API BASE", help='Your openai_api_base')
    @click.option('--openai_api_key', prompt="OpenAI API KEY", help='Your openai_api_key')
    @click.option('--system_role', default='', prompt="SYSTEM ROLE", help='OpenAI system_role')
    @click.option('--temperature', default=0.7, prompt="TEMPERATURE", help='OpenAI temperature')
    @click.option('--app_id', prompt="APP ID", help='Your app_id')
    @click.option('--app_secret', default='', prompt="APP SECRET", help='Your app_secret')
    @click.option('--verification_token', default='',
                  prompt="VERIFICATION TOKEN", help='Your verification_token')
    @click.option('--encrypt_key', prompt="ENCRYPT KEY", help='Your encrypt_key')
    @click.option('--debug', default=False, prompt="DEBUG MODE", help='debug mode')
    def main(debug, **kwargs):
        app = Application(**kwargs)
        client = Client(app.bot)
        client.start(debug)  # debug mode

    main()


