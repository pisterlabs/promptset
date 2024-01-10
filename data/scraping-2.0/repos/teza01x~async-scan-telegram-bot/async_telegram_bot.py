import os
import asyncio
from telebot.async_telebot import AsyncTeleBot
from telebot import types
from tesseract_vision import *
from async_openai import *
from async_funcs import *
from sql import *
from config import *



bot = AsyncTeleBot(telebot_token)


# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
async def start(message):
    ### Check user in db and if not = add to db
    user_id = message.chat.id
    db = DataBase(db_name)

    if not db.user_exists(user_id):
        try:
            db.add_user(user_id)
            db.my_balance_add_user(user_id)
            db.close()
        except Exception as e:
            with open("error_log.txt", "w") as txt:
                txt.write("ERROR:\n" + "{}".format(e))
    ### Welcome message
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    await bot.send_message(message.chat.id, dct['hello_word'])



### /commands, show actual commands and their describe
@bot.message_handler(commands=['commands'])
async def info(message):
    ### Available project commands
    user_id = message.chat.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    if message.chat.type == 'private':
        await bot.send_message(message.chat.id, dct['commands'])
    else:
        await bot.send_message(message.chat.id, dct['sry_private'])


### /menu, show actual menu buttons
@bot.message_handler(commands=['menu'])
async def menu(message):
    ### Accessible project menu, with buttons
    user_id = message.chat.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    if message.chat.type == 'private':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        send_photo = types.KeyboardButton(dct['send_p'])
        send_text = types.KeyboardButton(dct['send_t'])
        get_text = types.KeyboardButton(dct['get_t'])
        my_balance = types.KeyboardButton(dct['mbal'])

        markup.add(send_photo, send_text, get_text, my_balance)
        await bot.send_message(message.chat.id, dct['menu_active'], reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, dct['sry_private'])


### /id command, describe supported language ids
@bot.message_handler(commands=['id'])
async def lang_list(message):
    user_id = message.chat.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    if message.chat.type == 'private':
        await bot.send_message(message.chat.id, dct['sup_lang_list'])
    else:
        await bot.send_message(message.chat.id, dct['sry_private'])


### /lang command, choose language of interface
@bot.message_handler(commands=['lang'])
async def lang_list(message):
    user_id = message.chat.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    if message.chat.type == 'private':
        button_list = [
            types.InlineKeyboardButton(dct['lang1'], callback_data='button1'),
            types.InlineKeyboardButton(dct['lang2'], callback_data='button2'),
            types.InlineKeyboardButton(dct['lang3'], callback_data='button3'),
            types.InlineKeyboardButton(dct['lang4'], callback_data='button4'),
            types.InlineKeyboardButton(dct['lang5'], callback_data='button5'),
        ]
        reply_markup = types.InlineKeyboardMarkup([button_list])
        await bot.send_message(message.chat.id, dct['c_lang'], reply_markup=reply_markup)
    else:
        await bot.send_message(message.chat.id, dct['sry_private'])


### Обработка нажатия кнопки, определение типажа и изменение в БД
@bot.callback_query_handler(func=lambda call: True)
async def callback_query(call):
    user_id = call.from_user.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    if call.data == "button1":
        db = DataBase(db_name)
        db.update_user_language("English", user_id)
        db.close()
        await bot.answer_callback_query(call.id, dct['lang_switch1'])
    elif call.data == "button2":
        db = DataBase(db_name)
        db.update_user_language("Russian", user_id)
        db.close()
        await bot.answer_callback_query(call.id, dct['lang_switch2'])
    elif call.data == "button3":
        db = DataBase(db_name)
        db.update_user_language("Ukrainian", user_id)
        db.close()
        await bot.answer_callback_query(call.id, dct['lang_switch3'])
    elif call.data == "button4":
        db = DataBase(db_name)
        db.update_user_language("German", user_id)
        db.close()
        await bot.answer_callback_query(call.id, dct['lang_switch4'])
    elif call.data == "button5":
        db = DataBase(db_name)
        db.update_user_language("French", user_id)
        db.close()
        await bot.answer_callback_query(call.id, dct['lang_switch5'])


    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    await bot.send_message(user_id, dct['reactivate_menu'])


### /a command get simple answer on only text request, with out photo handling
@bot.message_handler(commands=['a'])
async def get_answer(message):
    user_id = message.chat.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    if message.chat.type == 'private':
        await bot.send_message(message.chat.id, dct['pls_wait'])
        text = await clean_text(message.text, '/a')

        ### Check for len /a (answer) text True/False
        if bool(len(text)) == True:

            ### Check for decimal len of text
            if len(text) <= token_limit:

                ### DataBase connection + payment proc
                db = DataBase(db_name)
                if db.my_balance_payment(user_id) == True:
                    db.close()
                    await bot.send_message(message.chat.id, dct['pay_token_suc'])

                    ### Send request to openAI
                    correct_text = await text_answer_request(text)

                    ### Send to user answer on his question
                    text_answer = dct['post_text'] + correct_text
                    await bot.send_message(message.chat.id, text_answer)

                    ### Decline payment if small balance
                elif db.my_balance_payment(user_id) == False:
                    db.close()
                    await bot.send_message(message.chat.id, dct['pay_token_decl'])
            else:
                await bot.send_message(message.chat.id, dct['error_tlim'])
        else:
            await bot.send_message(message.chat.id, dct['error_stext'])
    else:
        await bot.send_message(message.chat.id, dct['sry_private'])


### Handling Menu Requests
@bot.message_handler(content_types=['text'])
async def bot_send_photo(message):
    user_id = message.chat.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    if message.chat.type == 'private':
        if message.text == dct['send_p']:
            await bot.send_message(message.chat.id, dct['sp_desc'])
        elif message.text == dct['send_t']:
            await bot.send_message(message.chat.id, dct['st_desc'])
        elif message.text == dct['get_t']:
            await bot.send_message(message.chat.id, dct['gt_desc'])
        elif message.text == dct['mbal']:
            db = DataBase(db_name)
            token = db.my_balance_check(user_id)
            await bot.send_message(message.chat.id, dct['gb_check'].format(token))


### Get text from photo using lang id + get request from openai with answer
@bot.message_handler(content_types=["photo"])
async def text_detection(message):
    user_id = message.chat.id
    db = DataBase(db_name)
    dct = await lang_dict(db.check_user_laguage(user_id))
    db.close()
    ### Process message
    if message.chat.type == 'private':
        await bot.send_message(message.chat.id, dct['pls_wait'])

        if message.caption != None:
            caption_list = message.caption.split(" ")
            lid = await lang_choose(message.caption, sup_lang)
        else:
            caption_list = ['eng']
            lid = caption_list[0]

        ### Check for /t command
        if caption_list[0] != '/t':

            ### Check for img size if bigger than limit >>> decline
            if message.photo[-1].file_size <= allow_img_size:

                ### Download photo by file_id
                fileID = message.photo[-1].file_id
                file_info = await bot.get_file(fileID)
                downloaded_file = await bot.download_file(file_info.file_path)

                ### Image path + random name
                random_file_name = await random_img_name()
                image = image_path + random_file_name

                with open(image, 'wb') as new_file:
                    new_file.write(downloaded_file)

                ### Tesseract text detection in var text
                text = await text_detect(image, lid)

                ### Checking for text detection
                if bool(len(text)) == True:
                    try:
                        ### DataBase connection + payment
                        db = DataBase(db_name)
                        if db.my_balance_payment(user_id) == True:
                            db.close()
                            await bot.send_message(message.chat.id, dct['pay_token_suc'])

                            ### Send request with text to openAI API
                            correct_text = await text_answer_request(text)
                            text_answer = dct['post_text'] + correct_text

                            ### Send answer text to user
                            await bot.send_message(message.chat.id, text_answer)

                            ### Payment decline if small balance
                        elif db.my_balance_payment(user_id) == False:
                            db.close()
                            await bot.send_message(message.chat.id, dct['pay_token_decl'])
                    except Exception as e:
                        with open("error_log.txt", "w") as txt:
                            txt.write("ERROR:\n" + "{}".format(e))
                elif bool(len(text)) == False:
                    await bot.send_message(message.chat.id, dct['error_tlim'])

                ### Delete used photo
                asyncio.get_running_loop().run_in_executor(None, os.remove, image)
            else:
                await bot.send_message(message.chat.id, dct['error_too_big_img'])

        ### Check for command /t
        elif caption_list[0] == '/t':
            lid = await lang_choose(await clean_text(message.caption, '/t'), sup_lang)

            ### Check img size
            if message.photo[-1].file_size <= allow_img_size:
                ### Download photo by file_id
                fileID = message.photo[-1].file_id
                file_info = await bot.get_file(fileID)
                downloaded_file = await bot.download_file(file_info.file_path)

                ### Image path + random name
                random_file_name = await random_img_name()
                image = image_path + random_file_name

                with open(image, 'wb') as new_file:
                    new_file.write(downloaded_file)

                ### Tesseract text detection in var text
                text = await text_detect(image, lid)

                ### Checking for text detection
                if bool(len(text)) == True:
                    try:
                        ### DataBase connection + payment
                        db = DataBase(db_name)
                        if db.my_balance_payment(user_id) == True:
                            db.close()
                            await bot.send_message(message.chat.id, dct['pay_token_suc'])

                            ### Send request with text to openAI API
                            text_answer = dct['post_text'] + text

                            ### Send answer text to user
                            await bot.send_message(message.chat.id, text_answer)

                            ### Payment decline if small balance
                        elif db.my_balance_payment(user_id) == False:
                            db.close()
                            await bot.send_message(message.chat.id, dct['pay_token_decl'])
                    except Exception as e:
                        with open("error_log.txt", "w") as txt:
                            txt.write("ERROR:\n" + "{}".format(e))
                elif bool(len(text)) == False:
                    await bot.send_message(message.chat.id, dct['error_tlim'])

                ### Delete used photo
                asyncio.get_running_loop().run_in_executor(None, os.remove, image)
            else:
                await bot.send_message(message.chat.id, dct['error_too_big_img'])
    else:
        await bot.send_message(message.chat.id, dct['sry_private'])


async def main():
    while True:
        try:
            await bot.send_message(chat_id=admin_chat_id, text="🤖 Bot started working")
            await bot.infinity_polling()
        except Exception as e:
            await bot.send_message(chat_id=admin_chat_id, text=f"⚠️ Bot has been crashed. Error: {str(e)}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
