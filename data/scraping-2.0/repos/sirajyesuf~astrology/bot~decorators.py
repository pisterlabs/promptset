from functools import wraps
import db
import config
import time
import db
import math
import time
from pyrogram.types import (InlineKeyboardMarkup,InlineKeyboardButton,ReplyKeyboardMarkup,)
from pyrogram import enums
from openai_helper import openai_helper
from datetime import datetime
from enums import Status
from pyrogram import errors
import json
import requests

def only_for_subscribers(bot):
    def wrapper(func):
        @wraps(func)
        async def inner_wrapper(app,message):
            _bot = await bot.get_me()
            user = db.users.find_one(telegram_user_id=message.chat.id)
            subscription = db.subscriptions.count(user_id=user['id'],status=Status.ACTIVE.value)
            
            if(subscription):
                return  await func(app,message)
            else:
                msg = f"you dont have active subscription.use this bot to buy a subscription.\n\n @{_bot.username}"
                await app.send_message(chat_id = message.chat.id,text=msg)
        return inner_wrapper
    return wrapper

def register(func):
    @wraps(func)
    async def wrapper(bot,message):
            user = db.users.find_one(telegram_user_id=message.chat.id)
            if(user is  None):
                user = {
                    'first_name' : message.chat.first_name,
                    'last_name': message.chat.last_name,
                    'telegram_user_id' : message.chat.id,
                    'telegram_handler' : message.chat.username,
                    'created_at' : datetime.now(),
                    'updated_at' : datetime.now()
                }
                db.users.insert(user)
            await func(bot,message)
    return wrapper

def registered(bot):
    def wrapper(func):
        @wraps(func)
        async def inner_wrapper(app,message):
            _bot = await bot.get_me()
            user = db.users.find_one(telegram_user_id=message.chat.id)
            if(user is  None):
                text = f"you dont have active Subscription {message.chat.first_name}. use this bot to buy a subscription.\n\n @{_bot.username}"
                await app.send_message(
                    chat_id = message.chat.id,
                    text = text
                )
            else:
                await func(app,message)
        return inner_wrapper
    return wrapper

def only_unsubscribers(func):
    @wraps(func)
    async def wrapper(bot,message):
            user = db.users.find_one(telegram_user_id=message.chat.id)
            subscription = db.subscriptions.count(user_id=user['id'],status=Status.ACTIVE.value)
            if(subscription == 0):
                return  await func(bot,message)
            else:
                msg = "you are already subscribed."
                await bot.send_message(chat_id = message.chat.id,text=msg)
    return wrapper



def has_session(func):
    @wraps(func)
    async def wrapper(app,message):
            user = db.users.find_one(telegram_user_id = message.chat.id)
            subscription = db.subscriptions.find_one(user_id = user['id'],status=Status.ACTIVE.value)
            plan = db.plans.find_one(id = subscription['plan_id'])

            if(subscription['used_sessions'] < plan['number_of_session']):
                await func(app,message)
            else:
                db.subscriptions.update(
                dict(id= subscription['id'],status = Status.DEACTIVE.value),
                ['id'])
                await send_end_of_all_session_message(app,message)
    return wrapper


async def send_end_of_all_session_message(app,message):

    end_of_all_sessions_prompt = db.get_setting()['end_of_all_sessions_propmt']
    answer = openai_helper.get_chat_response(message.chat.id,end_of_all_sessions_prompt)
    # length_of_words_in_gpt_answer = len(answer.split(" "))
    # time.sleep(length_of_words_in_gpt_answer)
    await app.send_message(chat_id = message.chat.id,text = answer)






async def add_client_chat_history(user_id,prompt,answer):
    return db.histories.insert({
        'user_id': user_id,
        'prompt' : prompt,
        'answer' : answer      
    })

# def timeit(bot):
#     def wrapper(func):
#         @wraps(func)
#         async def inner_wrapper(*args,**kwrags):
#             start_time = time.perf_counter()
#             app = args[0]
#             message = args[1]
#             user = db.users.find_one(telegram_user_id = message.chat.id)
#             subscription = db.subscriptions.find_one(user_id = user['id'],status = Status.ACTIVE.value)
#             plan = db.plans.find_one(id=subscription['plan_id'])

#             chat_gpt_answer = await func(*args,**kwrags)
#             await add_client_chat_history(user['id'],message.text,chat_gpt_answer)
#             length_of_words_in_gpt_answer = len(chat_gpt_answer.split(" "))
#             time.sleep(length_of_words_in_gpt_answer)
#             await app.send_message(chat_id = message.chat.id,text=chat_gpt_answer)
#             end_time = time.perf_counter()
#             # minutes
#             total_time = (end_time - start_time ) / 60
            
#             #update the uptime and  number of prompt  of the subscriber
#             db.query(f'UPDATE subscriptions SET uptime = uptime + {total_time} WHERE user_id = {user["id"]} AND  status = 2')
#             if(subscription['number_of_propmt'] >= 1):
#                 db.subscriptions.update(dict(id=subscription['id'],number_of_propmt = 0),['id'])
#                 remaning_sessions_in_minutes = math.ceil(int(plan ['number_of_session'] * config.ONE_SESSION_IN_MINUTES)  - subscription['uptime'])
#                 await  countdown(bot,message,remaning_sessions_in_minutes)
#             else:
#                 db.query(f'UPDATE subscriptions SET number_of_propmt = number_of_propmt + 1 WHERE user_id = {user["id"]} AND  status = 2')

#         return inner_wrapper
#     return wrapper

def set_used_session(bot):

    def wrapper(func):
        @wraps(func)
        async def inner_wrapper(app,message):

            user = db.users.find_one(telegram_user_id = message.chat.id)
            subscription = db.subscriptions.find_one(user_id = user['id'],status = Status.ACTIVE.value)
            plan = db.plans.find_one(id=subscription['plan_id'])

            first_message_datetime = subscription['first_message_datetime'].timestamp() if subscription.get('first_message_datetime',None) else message.date.timestamp()
            final_message_datetime = message.date.timestamp()
            difference =  (final_message_datetime - first_message_datetime) / 60
            current_session = int(difference / config.ONE_SESSION_IN_MINUTES)

            if(current_session >= plan['number_of_session']):
                db.subscriptions.update(
                    dict(id=subscription['id'],used_sessions = current_session),
                    ['id']
                )

            elif(current_session > subscription['used_sessions'] and current_session < plan['number_of_session']):
                db.subscriptions.update(
                    dict(id=subscription['id'],used_sessions = current_session),
                    ['id']
                )
                # send end of a session prompt
                end_of_session_prompt = db.get_setting()['end_of_session_prompt']
                answer = openai_helper.get_chat_response(message.chat.id,end_of_session_prompt)
                # length_of_words_in_gpt_answer = len(answer.split(" "))
                # time.sleep(length_of_words_in_gpt_answer)
                await app.send_message(chat_id = message.chat.id,text = answer)

                # send countdown in side the bot
                remaning_sessions_in_minutes = (plan['number_of_session']*config.ONE_SESSION_IN_MINUTES)  - (current_session * config.ONE_SESSION_IN_MINUTES)
                await countdown(bot,message,remaning_sessions_in_minutes)
            
            await func(app,message)

        return inner_wrapper 
    
    return wrapper




def timeit(bot):
    def wrapper(func):
        @wraps(func)
        async def inner_wrapper(*args,**kwrags):

            app = args[0]
            message = args[1]
            user = db.users.find_one(telegram_user_id = message.chat.id)
            subscription = db.subscriptions.find_one(user_id = user['id'],status = Status.ACTIVE.value)
            plan = db.plans.find_one(id=subscription['plan_id'])

            # update for first message col once
            # first_message = False
            if(not subscription.get('first_message_datetime',None)):
                # first_message = True
                db.subscriptions.update(
                    dict(id = subscription['id'],first_message_datetime = message.date),['id']
                )
                

            # update final message col everytime
            db.subscriptions.update(
                dict(id = subscription['id'],final_message_datetime = message.date),['id']
            )

            # update the number of session
            # first_message_datetime = subscription['first_message_datetime'].timestamp() if not first_message else message.date.timestamp()
            # final_message_datetime = message.date.timestamp()
            # difference =  (final_message_datetime - first_message_datetime) / 60
            # current_session = int(difference / config.ONE_SESSION_IN_MINUTES)

            # if(current_session > subscription['used_sessions'] and current_session <= plan['number_of_session'] ):
            #     db.subscriptions.update(
            #         dict(id=subscription['id'],used_sessions = current_session),
            #         ['id']
            #     )
            #     # send end of a session prompt
            #     end_of_session_prompt = db.get_setting()['start_of_session_prompt']
            #     answer = openai_helper.get_chat_response(message.chat.id,end_of_session_prompt)
            #     # length_of_words_in_gpt_answer = len(answer.split(" "))
            #     # time.sleep(length_of_words_in_gpt_answer)
            #     await app.send_message(chat_id = message.chat.id,text = answer)

            #     # send countdown in side the bot
            #     remaning_sessions_in_minutes = (plan['number_of_session']*config.ONE_SESSION_IN_MINUTES)  - (current_session * config.ONE_SESSION_IN_MINUTES)
            #     await countdown(bot,message,remaning_sessions_in_minutes)


            chat_gpt_answer = await func(*args,**kwrags)
            await add_client_chat_history(user['id'],message.text,chat_gpt_answer)
            length_of_words_in_gpt_answer = len(chat_gpt_answer.split(" "))

            time.sleep(length_of_words_in_gpt_answer)

            await app.send_message(chat_id = message.chat.id,text=chat_gpt_answer)

        return inner_wrapper
    
    return wrapper

def typing(func):
    @wraps(func)
    async def wrapper(app,message):
        await app.send_chat_action(message.chat.id, enums.ChatAction.TYPING)
        await func(app,message)

    return wrapper



async def countdown(bot,message,remaning_sessions_in_minutes):
        print("count down bot is called")
        
        if(float(remaning_sessions_in_minutes) > 0):

            await bot.send_message(
                chat_id = message.chat.id,
                text=f"#Remaning Sessions in Minutes\n\n ⏳ {remaning_sessions_in_minutes} minutes only.",
                )

            # _messages = db.messages.count(telegram_user_id = message.chat.id)
            # if(_messages):
            #     _messages = db.messages.find_one(telegram_user_id = message.chat.id)
            #     try:
            #         await bot.edit_message_text(
            #             chat_id=message.chat.id,
            #             message_id=_messages['message_id'],
            #             text=f"#Remaning Sessions in Minutes\n\n ⏳ {remaning_sessions_in_minutes} minutes only.",
            #             )
            #     except errors.exceptions.bad_request_400.MessageNotModified:
            #         print("MessageNotModified")
            #         response = await bot.send_message(
            #         chat_id = message.chat.id,
            #         text=f"#Remaning Sessions in Minutes\n\n ⏳ {remaning_sessions_in_minutes} minutes only.",
            #         # reply_markup = button
            #         )
            #         db.messages.insert({
            #         'telegram_user_id':message.chat.id,
            #         'message_id':response.id,
            #         'start_session_message_id': response.id,
            #         'is_close_session_button_sent' :False
            #         })
            # else:
            #     response = await bot.send_message(
            #         chat_id = message.chat.id,
            #         text=f"#Remaning Sessions in Minutes\n\n ⏳ {remaning_sessions_in_minutes} minutes only.",
            #         # reply_markup = button
            #         )
            #     db.messages.insert({
            #     'telegram_user_id':message.chat.id,
            #     'message_id':response.id,
            #     'start_session_message_id': response.id,
            #     'is_close_session_button_sent' :False
            #     })
        else:
            db.messages.delete()
            text = f"Dear {{message.chat.first_name}}\n\nI'm sorry to inform you that your subscription has ended.We hope that you enjoyed using our service during your subscription period.\nwe would like to offer you a discounted plan to continue using our service. Our discounted plan offers the same features as your previous subscription, but at a lower cost."
            await bot.send_message(chat_id = message.chat.id,text=text)
            await  send_discount_invoice(bot,message)

    





async def send_discount_invoice(bot,message):
    db_user = db.users.find_one(telegram_user_id=message.chat.id)
    plan =  db.plans.find_one(is_primary = False)
    subscription = db.subscriptions.count(status = Status.PENDING.value,plan_id = plan['id'],user_id = db_user['id'])
    if(subscription == 0):
        subscription = {
        'user_id':db_user['id'],
        'plan_id':plan['id'],
        'status': Status.PENDING.value,
        'used_sessions': 0,
        'created_at': datetime.now(),
        'updated_at' :datetime.now(),
        }
        subscription_id = db.subscriptions.insert(subscription)
    else:
        subscription = db.subscriptions.find_one(status = Status.PENDING.value,plan_id = plan['id'],user_id = db_user['id'])

        subscription_id = subscription['id']



    prices = [{"label": plan['name'], "amount": int(plan['price'])*100}]
    prices_json = json.dumps(prices)
    invoice = {
        'chat_id' : message.chat.id,
        'title':plan['name'],
        'description':plan['description'],
        'payload':str(subscription_id),
        'provider_token':config.PAYMENT_PROVIDER_TOKEN,
        'currency':'USD',
        'prices':prices_json,
        'start_parameter': str(subscription_id)
    }

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendInvoice"

    r = requests.post(url,data=invoice)

    print(r.text)