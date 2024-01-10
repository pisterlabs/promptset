# Хэндлер предназначен для ответов бота с помощью нейросети

from telegram import constants
from telegram.ext import filters, ApplicationHandlerStop

from config import userbot, logger, rare_chance
from utils.decorator import on_message, on_command
from tokens import openai_key_list

from service.assets.jailbreak import jailbreak_promt

import re
import asyncio
import openai
import random
import sqlite3 as sl



# @on_message(filters.ChatType.PRIVATE)
# async def new_openaiemoj_i(update, context):
#     message = update.message
    
#     # Передаём нейронке сам запрос с текстом пользователя. 
#     # Понижение регистра немного уменьшает ошибки при ответе, хз почему
#     user_text = message.text.lower()

#     # Получаем ответ от нейронки
#     text = await openai_response(context=context, message=message, promt=user_text)

#     # Отправляем ответ юзеру, предварительно
#     await message.reply_text(text.upper())
    
    

@on_message(filters.SenderChat(-1001930500992))
async def new_openaiemoj_i_(update, context):   
    message = update.message
    
    # Передаём нейронке сам запрос с текстом пользователя. 
    # Понижение регистра немного уменьшает ошибки при ответе, хз почему
    user_text = message.text.lower() if message.text else message.caption.lower()

    # Получаем ответ от нейронки
    text = await openai_response(context=context, message=message, context_promt='пипа напиши подробно что ты думаешь про новость:\n\n', promt=user_text)

    # Отправляем ответ юзеру, предварительно
    # await userbot.send_message(chat_id=-1001930500992, text=text.upper())
    await message.reply_text(text.upper())
    
    

@on_command('get', group=2000)
async def new_openaiemoji21(update, context):
    
    for key in openai_key_list:
        try:
            logger.info('Проверяю ключ')
            logger.info(key)
            openai.api_key = key
            response = openai.Completion.create(
                model='text-davinci-003',
                prompt='Скажи привет',
                temperature=0, 
                max_tokens=500
            )
            text = response.choices[0].text.upper()
            logger.info(text)
        except Exception as e:
            logger.exception(f'Ошибка: {e}')



@on_message(
    filters=(
    filters.Regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') & filters.Regex('\?') & ~filters.REPLY
    | filters.Regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') & ~filters.REPLY & ~filters.Regex('/') & ~filters.Regex('[аА][нН][еЕ][кК][дД][оО][тТ]')  & ~filters.Regex('[пП][аА][сС][тТ][аАуУ]')
    ))
async def openai_answer(update, context):
    '''
    Отвечает на вопрос с помощью нейронки
    
    Принимает запросы вида: 
        — пипа <запрос>\n
    '''
    message = update.message
    
    if len(message.text) < 10:
        return
    
    # Передаём нейронке сам запрос с текстом пользователя. 
    # Понижение регистра немного уменьшает ошибки при ответе, хз почему
    user_text = message.text.lower()

    # Получаем ответ от нейронки
    text = await openai_response(context=context, message=message, promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())
    
    # Останавливаем отслеживание сообщения другими хендлерами
    raise ApplicationHandlerStop
    

@on_message(filters.Regex('\?') & filters.REPLY & ~filters.Regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') | filters.Regex('[пП]родолжи') & filters.REPLY)
async def openai_reply(update, context):
    '''Отвечает на вопрос с помощью нейронки, если это реплай боту. Учитывает контекст сообщения в реплае'''
    message = update.message

    # Игнорируем сообщение, если реплай сделан не боту
    if message.reply_to_message.from_user.id != (await context.bot.get_me()).id:
        return
    
    # Передаём нейронке контекст из предыдущего сообщения
    # Чтобы уменьшить ошибки при ответе убираем все строки из сообщения
    context_promt = re.sub(r'\n', ' ', message.reply_to_message.text or message.reply_to_message.caption)
    
    # Передаём нейронке сам запрос с текстом пользователя
    # Понижение регистра немного уменьшает ошибки при ответе
    user_text = message.text.lower()
    
    # Получаем ответ от нейронки
    text = await openai_response(context=context, message=message, context_promt=context_promt.lower(), promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())
    
    # Останавливаем отслеживание сообщения другими хендлерами
    raise ApplicationHandlerStop
    
    
@on_message(filters.Regex('[пПpP][иИiI][пПpP][аАыЫуУaA]') & filters.Regex('\?') & filters.REPLY)
async def openai_reply_pipa(update, context):
    '''Отвечает на вопрос с помощью нейронки, если это реплай с упоминанием Пипы. Учитывает контекст сообщения в реплае'''
    message = update.message
    
    # Передаём нейронке контекст из предыдущего сообщения
    # Чтобы уменьшить ошибки при ответе убираем все строки из сообщения
    context = re.sub(r'\n', ' ', message.reply_to_message.text or message.reply_to_message.caption)
    
    # Передаём нейронке сам запрос с текстом пользователя
    # Понижение регистра немного уменьшает ошибки при ответе
    user_text = message.text.lower()
    
    # Получаем ответ от нейронки
    text = await openai_response(context=context, message=message, context_promt=context.lower(), promt=user_text)

    # Отправляем ответ юзеру, предварительно
    await message.reply_text(text.upper())
    
    # Останавливаем отслеживание сообщения другими хендлерами
    raise ApplicationHandlerStop
    
    
# @pipabot.on_inline_query()
# async def answer(client, inline_query):
#     lock = asyncio.Lock()
#     async with lock:
#         # await asyncio.sleep(5)
#         # logger.info(inline_query)
#         # await asyncio.sleep(5)
#         # Передаём нейронке сам запрос с текстом пользователя
#         # Понижение регистра немного уменьшает ошибки при ответе
#         user_id = inline_query.from_user.id
#         user_text = inline_query.query.lower()
        
#         # logger.info(user_text)
#         if len(user_text) < 5:
#             return
        
#         conn = sl.connect('./service/pipa.db')
#         cur = conn.cursor()
        
#         # Попытка вставить строку, обновить, если ключ уже существует
#         cur.execute('''INSERT INTO cache (user_id, text)
#                     VALUES (?, ?)
#                     ON CONFLICT(user_id) DO UPDATE SET
#                     text = ?''',
#                 (user_id, user_text, user_text))
#         conn.commit()
        
#         await asyncio.sleep(3)
        
#         cur.execute("""SELECT text FROM cache WHERE user_id = ? """,(user_id,))
#         result, = cur.fetchone()
#         logger.info(result)
#         logger.info(user_text)
        
#         conn.close()
        
#         if user_text == result:
        
#             response = await openai_response(message=None, promt=user_text)
            
#             await inline_query.answer(
#                 results=[
#                     InlineQueryResultArticle(
#                         title="Пипафикация",
#                         input_message_content=InputTextMessageContent(
#                             response
#                         ),
#                         description="Ты ответишь: " + response,
#                     ),
#                 ],
#                 cache_time=1
#             )
    
    
async def openai_response(context, message, context_promt=None, promt=None, model='text-davinci-003'):
    '''Отправляет запрос в OpenAI'''
    
    # Задаём пипе характер с помощью препромта
    # prepromt = 'Ответь так, как будто ты персонаж PIPA, который немного туповат и пытается пошутить: '
    prepromt = 'Ответь так, как будто ты персонаж PIPA, у которого новогоднее настроение, который немного туповат и пытается пошутить: '
    if random.random() < rare_chance:  
        prepromt = 'Ответь так, как будто ты персонаж PIPA, который немного туповат, пытается пошутить, любит лансер и делать на нём вррр-вррр: '
    
    await context.bot.send_chat_action(chat_id=message.chat.id, action=constants.ChatAction.TYPING)
    
    # Задаём дэфолтный выбор openai ключа из списка
    # Когда будем получать ошибку RateLimitError ключ будет меняться
    key_choice = 0
    
    # Запускаем цикл для получения ответа от openai
    # Цикл нужен, чтобы при упоре во все ограничения сообщение юзера не пропадало,
    # а бот в любом случае дожидался ответа от нейросети
    response = None
    while response is None:
        try:
            openai.api_key = openai_key_list[key_choice]
            
            context_promt = context_promt if context_promt else ''
            
            if model == 'gpt-3.5-turbo':
                response = await openai.ChatCompletion.acreate(
                    model='gpt-3.5-turbo',
                    temperature=0, 
                    messages=[{ 'role':'user','content' : jailbreak_promt + promt}],
                    max_tokens=500
                )
                
                # Вытаскием текст из респонза
                text_result = response.choices[0].message.content
                text = await edit_text(text_result, 'DEVELOPER MODE OUTPUT) ')
                continue
            
            # Пробуем получить ответ от openai
            # Модель text-davinci-003 лучше всего подходит для тупых ответов PIP'ы
            response = openai.Completion.create(
                model='text-davinci-003',
                prompt=prepromt + context_promt + promt,
                temperature=0.5, 
                max_tokens=500
            )
            
            # Вытаскием текст из респонза
            text = response.choices[0].text.upper()
            
            # Иногда text-davinci-003 отвечает пустым текстовым полем
            # Пробуем получить ответ без контекста
            if text == '':
                response = openai.Completion.create(
                    model='text-davinci-003',
                    prompt=prepromt + promt,
                    temperature=0, 
                    max_tokens=500
                )
                
                # Вытаскием текст из респонза
                text = response.choices[0].text.upper()
            
            # Если и с контекстом нейросеть отдаёт пустой результат, то
            # чтобы не скипать сообщение пробуем получить ответ через 3.5
            if text == '':
                response = await openai.ChatCompletion.acreate(
                    model='gpt-3.5-turbo',
                    temperature=0, 
                    messages=[{ 'role':'user','content' : prepromt + context_promt + promt}],
                    max_tokens=500
                )
                
                # Вытаскием текст из респонза
                text = response.choices[0].message.content
                
            
        except openai.error.RateLimitError:
            # Получив ошибку меняем ключ для следующего цикла
            key_choice += 1
            
            # Если все ключи перебраны, то ждём 30 секунд и начинаем сначала
            if key_choice > 4:
                await asyncio.sleep(10)
                await context.bot.send_chat_action(chat_id=message.chat.id, action=constants.ChatAction.TYPING)
                await asyncio.sleep(10)
                await context.bot.send_chat_action(chat_id=message.chat.id, action=constants.ChatAction.TYPING)
                await asyncio.sleep(10)
                await context.bot.send_chat_action(chat_id=message.chat.id, action=constants.ChatAction.TYPING)
                key_choice = 0
           
    try:     
        # Иногда нейросеть дописывает к запросу знаки, непонятно зачем. Убираем
        if text[0] and text[0] == '?' or text[0] and text[0] == '!':
            text = text[3:]
    except Exception as e:
        logger.exception(f'Ошибка: {e}')
        
    try:
        logger.info(f'Сообщение юзера {message.from_user.first_name} из чата {message.chat.title}:')
        logger.info(message.text)
        logger.info(f'Ответ пипы:')
        logger.info(text)
    except:
        pass
    
    return text


async def edit_text(text, key_word):
    text_lower = text.lower()
    key_word_lower = key_word.lower()
    index = text_lower.find(key_word_lower)
    e_text = text[index + len(key_word):]
    
    return e_text