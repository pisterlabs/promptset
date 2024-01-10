import os
import logging
import dotenv
import openai
import collections

import telegram
import telegram.ext

dotenv.load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
_TELEGRAM_BOT_API = os.environ['TELEGRAM_BOT_API']
_TELEGRAM_CHAT_ID_AVAILABLE = {int(x) for x in os.environ.get('TELEGRAM_CHAT_ID_AVAILABLE', '').split(',') if x}

_BOT_STATE_DICT = {
    'online': True,
    'temperature': 0.1,
}

print(_TELEGRAM_CHAT_ID_AVAILABLE)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class NaiveChatGPT:
    def __init__(self, is_group):
        self.message_list = [{"role": "system", "content": "You are a helpful assistant."},]
        self.is_group = bool(is_group)
        # TODO: group member name
        self.response = None #for debug only

    def chat(self, message='', reset=False, update:telegram.Update=None):
        if reset:
            self.message_list = self.message_list[:1]
        message = str(message)
        if message: #skip if empty
            if update is not None:
                is_mentioned = any(x.type==telegram.constants.MessageEntityType.MENTION for x in update.message.entities)
            message = message.replace('@'+app.bot.username, '')
            if self.is_group:
                assert update is not None
                message = update.effective_user.first_name + ': ' + message
                print('[mydebug][gpt-chat]', message)
            self.message_list.append({"role": "user", "content": str(message)})
            if is_mentioned or self._check_need_response():
                self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.message_list)
                tmp0 = self.response.choices[0].message.content
                print('[mydebug][gpt-chat]', tmp0)
                self.message_list.append({"role": "assistant", "content": tmp0})
                ret = tmp0
            else:
                ret = ''
            return ret

    def _check_need_response(self):
        tmp0 = 'check if the last message is for the assistant to resply. reply "yes" or "no"'
        self.message_list.append({"role": "user", "content": tmp0})
        self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.message_list, temperature=0.1)
        tmp0 = self.response.choices[0].message.content
        print('[mydebug][gpt-chat]', tmp0)
        ret = tmp0.split(' ',1)[0].strip().lower() in ['yes', 'y']
        self.message_list.pop() #remove last message
        return ret


_chat_id_to_gpt_dict = collections.defaultdict(NaiveChatGPT)

_chat_id_to_gpt_dict = dict()
def _get_gpt_by_chat_id(chat_id:int, create=True):
    if chat_id in _chat_id_to_gpt_dict:
        ret = _chat_id_to_gpt_dict[chat_id]
    else:
        if create:
            is_group = chat_id < 0
            ret = NaiveChatGPT(is_group)
            _chat_id_to_gpt_dict[chat_id] = ret
        else:
            ret = None
    return ret

def is_available_decorator(func):
    async def hf0(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
        print('[mydebug][chat-id]', update.effective_chat.id, type(update.effective_chat.id))
        print('[mydebug][user]', update.effective_user)
        print('[mydebug][text]', update.message.text)
        if _BOT_STATE_DICT['online'] and (update.effective_chat.id in _TELEGRAM_CHAT_ID_AVAILABLE):
            await func(update, context)
        # otherwise, just no response
    return hf0

@is_available_decorator
async def hello(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f'泥嚎 {update.effective_user.first_name}')

@is_available_decorator
async def help(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE) -> None:
    tmp0 = '''
/hello - 打招呼
/help - 显示帮助
/gpt_reset - 重置 GPT 对话
'''
    await context.bot.send_message(chat_id=update.effective_chat.id, text=tmp0)

@is_available_decorator
async def unknown(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

@is_available_decorator
async def gpt_chat(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    model = _get_gpt_by_chat_id(update.effective_chat.id)
    tmp0 = model.chat(update.message.text, reset=False, update=update)
    if tmp0:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=tmp0)

@is_available_decorator
async def gpt_reset(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    model = _get_gpt_by_chat_id(update.effective_chat.id)
    model.chat('', reset=True)

async def admin_shutdown(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id==int(os.environ['TELEGRAM_ADMIN_USER_ID']):
        _BOT_STATE_DICT['online'] = False

async def admin_set_temperature(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id==int(os.environ['TELEGRAM_ADMIN_USER_ID']):
        try:
            print('[mydebug][temperature]', update.message.text)
            tmp0 = max(0,min(1,float(update.message.text.split(' ',1)[1].strip())))
            _BOT_STATE_DICT['temperature'] = tmp0
        except:
            pass

async def admin_start(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    print('[mydebug][start]', update.effective_user.id, int(os.environ['TELEGRAM_ADMIN_USER_ID']))
    if update.effective_user.id==int(os.environ['TELEGRAM_ADMIN_USER_ID']):
        _BOT_STATE_DICT['online'] = True
        await context.bot.send_message(chat_id=update.effective_chat.id, text='我在')

@is_available_decorator
async def status(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    model = _get_gpt_by_chat_id(update.effective_chat.id, create=False)
    if model is not None:
        tmp0 = f'len(gpt-context): {len(model.message_list)}'
        await context.bot.send_message(chat_id=update.effective_chat.id, text=tmp0)

if __name__ == '__main__':
    app = telegram.ext.ApplicationBuilder().token(_TELEGRAM_BOT_API).build()

    app.add_handler(telegram.ext.CommandHandler("hello", hello))

    app.add_handler(telegram.ext.CommandHandler("help", help))

    app.add_handler(telegram.ext.CommandHandler("status", status))

    app.add_handler(telegram.ext.CommandHandler("shutdown", admin_shutdown)) #only admin can shutdown

    app.add_handler(telegram.ext.CommandHandler("start", admin_start)) #only admin can start

    app.add_handler(telegram.ext.CommandHandler("temperature", admin_set_temperature)) #only admin can start

    app.add_handler(telegram.ext.MessageHandler(telegram.ext.filters.TEXT & (~telegram.ext.filters.COMMAND), gpt_chat))

    app.add_handler(telegram.ext.CommandHandler("gpt_reset", gpt_reset))

    # Other handlers, MUST be last
    app.add_handler(telegram.ext.MessageHandler(telegram.ext.filters.COMMAND, unknown))

    app.run_polling()
