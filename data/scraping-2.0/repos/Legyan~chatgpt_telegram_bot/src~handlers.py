import logging
from typing import Union

from aiogram import F, Router, flags
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from prettytable import PrettyTable

import kb
import text
from config import dp
from db import (add_user, del_user, get_all_users_tokens,
                get_user_tokens_and_images, is_admin_user,
                is_user_in_whitelist, reset_all_users_tokens,
                reset_user_tokens, set_admin, update_image_count,
                update_tokens)
from openai_requests import generate_image_dalle, generate_text_chatgpt4
from states import Gen

admins_router = Router()
whitelist_users_router = Router()


@dp.message(Command('start'))
async def start_handler(msg: Message):
    await msg.answer(
        text.HELLO_TEXT.format(name=msg.from_user.username),
        reply_markup=kb.exit_kb
    )


@dp.callback_query(F.data == 'help')
@dp.message(Command('help'))
async def help_callback(input_obj: Union[CallbackQuery, Message]):
    if not is_admin_user(input_obj.from_user.id):
        answer = text.HELP_TEXT
    else:
        answer = text.ADMIN_HELP_TEXT
    if isinstance(input_obj, CallbackQuery):
        await input_obj.message.answer(answer)
    else:
        await input_obj.answer(answer)


@dp.message(F.text == 'Меню')
@dp.message(Command('menu'))
async def menu(msg: Message):
    await msg.answer('Главное меню', reply_markup=kb.menu)
    await msg.delete()


@whitelist_users_router.callback_query(F.data == 'my_tokens')
@whitelist_users_router.message(Command('my_tokens'))
async def get_tokens_callback(input_obj: Union[CallbackQuery, Message]):
    try:
        tokens, images = get_user_tokens_and_images(input_obj.from_user.id)
        answer = (text.SPENT_TOKENS.format(tokens=tokens, images=images))
    except Exception as error:
        logging.error(error)
        answer = text.COMMAND_ERROR
    if isinstance(input_obj, CallbackQuery):
        await input_obj.message.answer(answer, reply_markup=kb.exit_kb)
    else:
        await input_obj.answer(answer, reply_markup=kb.exit_kb)


@whitelist_users_router.callback_query(F.data == 'generate_text')
async def input_text_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.text_prompt)
    await clbck.message.edit_text(text.ENTER_PROMPT.format(model='ChatGPT-4'))


@whitelist_users_router.message(Gen.text_prompt)
@flags.chat_action('typing')
async def generate_text(msg: Message):
    mesg = await msg.reply(text.WAITING_ANSWER.format(model='ChatGPT-4'))
    name = msg.from_user.username
    message = msg.text
    try:
        gpt_response, tokens = await generate_text_chatgpt4(message, name)
        update_tokens(msg.from_user.id, tokens)
        await mesg.edit_text(gpt_response, disable_web_page_preview=True)
    except Exception as error:
        logging.error(error)
        await msg.edit_text(text.COMMAND_ERROR, reply_markup=kb.iexit_kb)


@whitelist_users_router.callback_query(F.data == 'generate_image')
async def input_image_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.img_prompt)
    await clbck.message.edit_text(text.ENTER_PROMPT.format(model='DALL·E'))


@whitelist_users_router.message(Gen.img_prompt)
@flags.chat_action('upload_photo')
async def generate_image(msg: Message):
    mesg = await msg.reply(text.WAITING_ANSWER.format(model='DALL·E'))
    try:
        img = await generate_image_dalle(msg.text)
        update_image_count(msg.from_user.id)
        await mesg.delete()
        await msg.answer_photo(
            photo=img[0], caption=f'Изображение по запросу "{msg.text}"'
        )
    except Exception as error:
        logging.error(error)
        await mesg.answer(text.COMMAND_ERROR)


@admins_router.message(Command('add_user'))
async def add_user_handler(msg: Message):
    name = msg.from_user.username
    message = msg.text
    try:
        _, new_user_name, new_user_id = message.split()
    except ValueError:
        logging.warning(text.INVALID_COMMAND_FROM.format(name=name))
        await msg.answer(text.INVALID_ADD_USER)
        return
    if is_user_in_whitelist(new_user_id):
        logging.info(
            text.ADMIN_COMMAND_NOT_DONE.format(name=name) + text.ALREADY_IN_WL
        )
        await msg.answer(text.ALREADY_IN_WL)
        return
    try:
        add_user(new_user_id, new_user_name)
        await msg.answer(
            text.SUCCESS_ADD_USER.format(name=new_user_name, id=new_user_id)
        )
    except Exception as error:
        logging.error(error)
        await msg.answer(text.COMMAND_ERROR)


@admins_router.message(Command('del_user'))
async def del_user_handler(msg: Message):
    name = msg.from_user.username
    message = msg.text
    try:
        _, del_user_id = message.split()
    except ValueError:
        logging.warning(text.INVALID_COMMAND_FROM.format(name=name))
        await msg.answer(text.INVALID_DEL_USER)
        return
    if not is_user_in_whitelist(del_user_id):
        logging.info(text.NOT_IN_WHITELIST)
        await msg.answer(text.NOT_IN_WHITELIST)
        return
    try:
        del_user(del_user_id)
        await msg.answer(text.SUCCESS_DEL_USER.format(id=del_user_id))
    except Exception as error:
        logging.error(error)
        await msg.answer(text.COMMAND_ERROR)


@admins_router.message(Command('reset_tokens'))
async def reset_user_tokens_handler(msg: Message):
    name = msg.from_user.username
    message = msg.text
    try:
        _, user_id = message.split()
    except ValueError:
        logging.warning(text.INVALID_COMMAND_FROM.format(name=name))
        await msg.answer(text.INVALID_RESET_TOKENS)
        return
    if not is_user_in_whitelist(user_id):
        logging.warning(text.INVALID_COMMAND_FROM.format(name=name))
        await msg.answer(text.NOT_IN_WHITELIST)
        return
    try:
        reset_user_tokens(msg.from_user.id)
        await msg.answer(text.SUCCESS_RESET_TOKENS)
    except Exception as error:
        logging.error(error)
        await msg.answer(text.COMMAND_ERROR)


@admins_router.message(Command('reset_all_tokens'))
async def reset_all_tokens_handler(msg: Message):
    try:
        reset_all_users_tokens()
        await msg.answer(text.SUCCESS_RESET_ALL_TOKENS)
    except Exception as error:
        logging.error(error)
        await msg.answer(text.COMMAND_ERROR)


@admins_router.message(Command('add_admin'))
async def add_admin_handler(msg: Message):
    name = msg.from_user.username
    message = msg.text
    try:
        _, user_id = message.split()
    except ValueError:
        logging.warning(text.INVALID_COMMAND_FROM.format(name=name))
        await msg.answer(text.INVALID_ADD_ADMIN)
        return
    if not is_user_in_whitelist(user_id):
        logging.warning(text.NOT_IN_WHITELIST)
        await msg.answer(text.NOT_IN_WHITELIST)
        return
    try:
        set_admin(user_id, True)
        await msg.answer(text.SUCCESS_ADD_ADMIN.format(id=user_id))
    except Exception as error:
        logging.error(error)
        await msg.answer(text.COMMAND_ERROR)


@admins_router.message(Command('del_admin'))
async def del_admin_handler(msg: Message):
    name = msg.from_user.username
    message = msg.text
    try:
        _, user_id = message.split()
    except ValueError:
        logging.warning(text.INVALID_COMMAND_FROM.format(name=name))
        await msg.answer(text.INVALID_DEL_ADMIN)
        return
    if not is_user_in_whitelist(user_id):
        logging.warning(text.NOT_IN_WHITELIST)
        await msg.answer(text.NOT_IN_WHITELIST)
        return
    try:
        set_admin(user_id, False)
        await msg.answer(text.SUCCESS_DEL_ADMIN.format(id=user_id))
    except Exception as error:
        logging.error(error)
        await msg.answer(text.COMMAND_ERROR)


@admins_router.message(Command('users'))
async def all_users_handler(msg: Message):
    try:
        users_list = get_all_users_tokens()
        x = PrettyTable()
        x.field_names = ['id', 'name', 'tokens', 'images', 'admin']
        for user in users_list:
            x.add_row(user)
        await msg.answer(
            '```\n' +
            x.get_string(fields=['name', 'tokens', 'images']) +
            '\n```',
            parse_mode='MarkdownV2'
        )
    except Exception as error:
        logging.error(error)
        await msg.answer(text.COMMAND_ERROR)


@whitelist_users_router.message()
@flags.chat_action('typing')
async def message_handler(msg: Message):
    mesg = await msg.reply(text.WAITING_ANSWER.format(model='ChatGPT-4'))
    name = msg.from_user.username
    message = msg.text
    try:
        gpt_response, tokens = await generate_text_chatgpt4(message, name)
        update_tokens(msg.from_user.id, tokens)
        await mesg.edit_text(gpt_response)
    except Exception as error:
        logging.error(error)
        await mesg.edit_text(text.COMMAND_ERROR)
