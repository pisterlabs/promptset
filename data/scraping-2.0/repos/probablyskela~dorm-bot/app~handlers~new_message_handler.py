import random
import re
import openai

from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, filters

from app.utils.utils import send_message_wrapper
from app.utils.config import settings
from app.utils import cache


async def new_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.edited_message is not None:
        bot_reply_id = await cache.cache.get(f'{update.effective_chat.id}-{update.effective_message.id}')
        if bot_reply_id is not None:
            await context.bot.edit_message_text(text='редачери гавноєди',
                                                chat_id=update.effective_chat.id,
                                                message_id=int(bot_reply_id))
            await cache.cache.delete(update.edited_message.id)
    elif update.effective_message.text.lower() == 'ні':
        await send_message_wrapper(update=update,
                                   context=context,
                                   text='hello')
    elif re.search(r'хто з \d{3}', update.effective_message.text.lower()) is not None:
        await send_message_wrapper(update=update,
                                   context=context,
                                   text='@kodein0slav, @Gwinbllade, @afekvova і @zemfirque (але останній лох)')
    elif random.randint(1, 200) == 69:
        await send_message_wrapper(update=update,
                                   context=context,
                                   text=settings.copypaste)
    elif '@probablyskela' in update.effective_message.text.lower() is not None:
        await send_message_wrapper(update=update,
                                   context=context,
                                   text='скела крутий')
    elif re.search(r'заберіть [прання|одяг]', update.effective_message.text.lower()):
        await context.bot.send_document(chat_id=update.effective_chat.id,
                                        document='https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3FtdWNoM25wYzFyMGY0eGt3MDYwZm5sNXgzNG40cTdlZTlxcmlvbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7FD6S5KE2KXauBATlK/giphy.gif',
                                        reply_to_message_id=update.effective_message.id)
    elif update.effective_message.text.lower().startswith('бот,') and 10 < len(update.effective_message.text):
        try:
            response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                                    messages=[{'role': 'user',
                                                               'content': update.effective_message.text}])
            await send_message_wrapper(update=update,
                                       context=context,
                                       text=response.choices[-1].message.content)
        except:
            await send_message_wrapper(update=update,
                                       context=context,
                                       text='@probablyskela, брат, гроші закінчилися, сам відповідай.')

new_message_handler = MessageHandler(filters=filters.TEXT & (~filters.COMMAND),
                                     callback=new_message)
