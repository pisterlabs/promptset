import sys
from datetime import datetime, timedelta

import config
import kb
import text

# from generators.image_gen import get_image
# from generators.text_gen import get_text
from admin import users
from advertisement import advert
from advertisement.apshed import parse_time_make_job  # , send_message_time_middleware
from aiogram import F, Router, flags, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from openAI import openai_utils
from states import Bot_advert, Gen
from utils import check_advert, check_flood, split_text

router = Router()


@router.message(Command("start"))
async def start_handler(msg: Message):
    """
    Command "/start"
    Answers with greetings and main menu
    """
    #                                                  -1001600846754             -1001873468435

    print(msg.from_user.id)
    user_channel_status = await config.bot.get_chat_member(
        chat_id="-1001873468435", user_id=msg.from_user.id
    )
    if msg.from_user.id in list(users.users_cls.keys()):
        await msg.answer(text.greet_again)
    else:
        await msg.answer(text.greet)
        users.all_users.update([msg.from_user.id])
        users.new_user(msg.from_user)

    # print(user_channel_status)
    if user_channel_status.status in ("member", "creator"):
        await msg.answer(text.menu, reply_markup=kb.menu)
    else:
        await config.bot.send_message(
            msg.from_user.id, text.not_subscribed, reply_markup=kb.not_subscribed_kb
        )


@router.message(Command("help"))
async def help_msg(msg: Message):
    """
    Command "/help"
    Answers with info
    """
    return await msg.answer(text.help_, reply_markup=kb.exit_kb)

@router.message(Command("advert"))
async def advert_msg(msg: Message):
    """
    Command "/help"
    Answers with info
    """
    return await msg.answer(text.advert_wait, reply_markup=kb.avdert_menu)


# =============================================MENU=======================================================


# @router.message(F.text == "Выйти в меню")
# @router.message(F.text == "◀️ Выйти в меню")
# async def menu_msg(msg: Message):
#     """
#     "Back to main menu" handler using message
#     """
#     await msg.answer(text.menu, reply_markup=kb.menu)


@router.callback_query(F.data == "done")
async def done(clbck: types.CallbackQuery, state: FSMContext):
    user_channel_status = await config.bot.get_chat_member(
        chat_id="-1001873468435", user_id=clbck.from_user.id
    )
    if user_channel_status.status in ("member", "creator"):
        await config.bot.send_message(
            clbck.from_user.id, text.menu, reply_markup=kb.menu
        )
    else:
        await config.bot.send_message(
            clbck.from_user.id, text.not_subscribed, reply_markup=kb.not_subscribed_kb
        )


@router.callback_query(F.data == "menu")
async def menu_callback(clbck: types.CallbackQuery, state: FSMContext):
    """
    Open main menu
    """
    # Переключение состояние на отсутсвие промптов на что-либо
    await state.set_state(Gen.nothing)
    return await clbck.message.answer(text.menu, reply_markup=kb.menu)


@router.callback_query(F.data == "help")
async def help_callback(clbck: types.CallbackQuery, state: FSMContext):
    """
    "Back to main menu" handler using callback from inline keybord
    """
    await clbck.message.answer(text.help_, reply_markup=kb.exit_kb)


@router.callback_query(F.data == "advert")
async def advert_callback(clbck: types.CallbackQuery, state: FSMContext):
    if users.isAdmin(clbck.from_user.id):
        await state.set_state(Bot_advert.start)
        return await clbck.message.answer(text.advert_start, reply_markup=kb.ad_kb)
    else:
        return await clbck.message.answer(text.notAdmin, reply_markup=kb.exit_kb)


@router.callback_query(F.data == "off")
async def shutdown_callback(clbck: types.CallbackQuery, state: FSMContext):
    sys.exit(0)


# ========================================================================================================


# =============================================Advertisement=======================================================


@router.callback_query(F.data == "ad_new")
async def ad_new(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_new)
    await clbck.message.answer(text.advert)


@router.callback_query(F.data == "ad_edit")
async def ad_edit(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_choose)
    print(advert.ads)
    await clbck.message.answer(text.advert_choose, reply_markup=kb.ad_choose_kb())


@router.callback_query(Bot_advert.ad_choose)
async def ad_chosen(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_edit)
    await state.update_data(editing_ad=await advert.getAdById(int(clbck.data)))
    await clbck.message.answer(text.advert_edit, reply_markup=kb.ad_edit_kb)


@router.callback_query(F.data == "ad_edit_info")
async def ad_edit_info(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_edit_info)
    await clbck.message.answer(text.advert_edit_info)


@router.callback_query(F.data == "ad_edit_timer")
async def ad_edit_timer(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_edit_timer)
    await clbck.message.answer(text.advert_edit_timer, reply_markup=kb.ad_timer_kb)


@router.callback_query(F.data == "ad_once")
async def ad_edit_timer_once(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_once)
    await clbck.message.answer(text.advert_edit_timer_once)


@router.callback_query(F.data == "ad_every_day")
async def ad_edit_timer_every_day(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_every_day)
    await clbck.message.answer(text.advert_edit_timer_many_datetime)


@router.callback_query(F.data == "ad_intervals")
async def ad_edit_timer_intervals(clbck: types.CallbackQuery, state: FSMContext):
    await state.set_state(Bot_advert.ad_intervals)
    await clbck.message.answer(text.advert_edit_timer_many_interval)


@router.callback_query(F.data == "ad_delete")
async def ad_delete(clbck: types.CallbackQuery, state: FSMContext):
    a = (await state.get_data()).get("editing_ad")
    await a.delete()


# @router.callback_query(F.data == "ad_send")
# async def ad_send(clbck: types.CallbackQuery, state: FSMContext):
#     a = (await state.get_data()).get("editing_ad")
#     await a.timed_send()


@router.callback_query(F.data == "ad_send")
async def ad_send(
    clbck: types.CallbackQuery, state: FSMContext, apscheduler: AsyncIOScheduler
):
    a = (await state.get_data()).get("editing_ad")
    apscheduler.add_job(
        a.send,
        trigger="date",
        run_date=datetime.now() + timedelta(seconds=5),
        kwargs={"user": clbck.from_user.id},
    )


# =================================================================================================================


# =============================================Состояния=======================================================
@router.callback_query(F.data == "generate_text")
async def input_text_prompt(clbck: types.CallbackQuery, state: FSMContext):
    if users.users_cls[clbck.from_user.id].limit <= 0:
        await clbck.message.answer(text.exceeded, reply_markup=kb.exit_kb)
        return 
    await state.set_state(Gen.text_prompt)
    await clbck.message.edit_text(text.gen_text)
    # await clbck.message.answer(text.gen_exit, reply_markup=kb.exit_kb)


@router.callback_query(F.data == "generate_image")
async def input_image_prompt(clbck: types.CallbackQuery, state: FSMContext):
    if users.users_cls[clbck.from_user.id].limit <= 0:
        await clbck.message.answer(text.exceeded, reply_markup=kb.exit_kb)
        return
    await state.set_state(Gen.image_prompt)
    await clbck.message.edit_text(text.gen_image)
    # await clbck.message.answer(text.gen_exit, reply_markup=kb.exit_kb)


# =============================================Состояния END====================================================


@router.message(Gen.text_prompt)
@flags.chat_action("typing")
async def generate_text(msg: types.Message, state: FSMContext):
    # Антиспам
    if await check_flood(message=msg, state=state):
        return False

    # testers_gonna_test = (await state.get_data()).get("context")
    # print(testers_gonna_test)
    prompt = msg.text
    _id = msg.from_user.id

    await users.users_cls[_id].input_prompt_text(prompt)

    prompt = users.users_cls[_id].previous_prompt + "\n" + prompt

    # if (prev_context := (await state.get_data()).get("context")) != None:
    #     prompt = prev_context + "\n" + prompt
    #     print(prompt)

    # Обновление контекста ПРОВЕРИТЬ НА РАЗНЫХ ПОЛЬЗОВАТЕЛЯХ!!!!!
    # await state.update_data(context=prompt)

    msg_tmp = await msg.answer(text.gen_wait)
    res = [prompt]
    # ===================================================================================================

    # Переделать под GPT!!!!!!!!!!!!!!!!
    res = await openai_utils.generate_text(prompt)

    if not res:
        await msg_tmp.delete()
        return await msg_tmp.edit_text(text.gen_error, reply_markup=kb.exit_kb)

    # Переделать под GPT!!!!!!!!!!!!!!!!
    await msg_tmp.edit_text(res[0], disable_web_page_preview=True)

    # ===================================================================================================
    await msg_tmp.delete()
    async for answer in split_text(text=res[0], max_length=2000):
        await config.bot.send_message(
            chat_id=_id, text=answer, disable_web_page_preview=True
        )
        # await msg.answer()
    # await msg_tmp.edit_text(res[0], disable_web_page_preview=True)

    # Надо ли добавлять return?


@router.message(Gen.image_prompt)
@flags.chat_action("upload_photo")
async def generate_image(msg: types.CallbackQuery, state: FSMContext):
    """
    Generate image
    """
    # Антиспам
    if await check_flood(message=msg, state=state):
        return False

    prompt = msg.text
    msg_tmp = await msg.answer(text.gen_wait)

    # ===================================================================================================
    res = await openai_utils.generate_image(prompt)

    if len(res) == 0:
        return msg_tmp.edit_text(text.gen_error, reply_markup=kb.exit_kb)

    await msg_tmp.delete()
    await msg_tmp.answer_photo(photo=res[0], caption=text.img_watermark)


# ===================================================================================================


# await clbck.message.answer(text.gen_image, reply_markup=kb.exit_kb)


@router.message(Gen.nothing)
async def echo(message: types.Message):
    """
    echo bot
    """
    await message.answer(
        "Это всего-лишь эхо далёких звёзд. Чтобы что-то сгенерировать, выбери пункт из списка.",
        reply_markup=kb.menu,
    )


# @router.message(Bot_advert.start)
# async def advert_get(message: types.Message, state: FSMContext):
#     return await check_advert(msg=message, state=state)


@router.message(Bot_advert.ad_new)
async def advert_get(message: types.Message, state: FSMContext):
    ad_type = await check_advert(msg=message, state=state)

    # получение фото и текста под фото
    print(message.photo, message.text)
    # print(message.caption)

    if ad_type == 1:
        new_ad = advert.Ad(
            "placeholder", message.caption, message.photo[0].file_id, ad_type
        )
    elif ad_type == 0:
        new_ad = advert.Ad("placeholder", message.text, "", ad_type)
    else:
        new_ad = advert.Ad("placeholder", "", message.photo[0].file_id, ad_type)

    await new_ad.send(message.from_user.id)
    await state.set_state(Bot_advert.ad_name)
    await state.update_data(ad=new_ad)
    await message.answer(text.advert_name)


@router.message(Bot_advert.ad_once)
async def timer_ad_once(
    message: types.Message, state: FSMContext, apscheduler: AsyncIOScheduler
):
    # print("I'm HERE BROOOO")
    date_format = "%Y-%m-%d %H:%M"

    await parse_time_make_job(
        msg=message.text,
        state=state,
        apscheduler=apscheduler,
        date_format=date_format,
        _type="once",
    )


@router.message(Bot_advert.ad_every_day)
async def timer_ad_every_day(
    message: types.Message, state: FSMContext, apscheduler: AsyncIOScheduler
):
    # print("I'm HERE BROOOO (2)")
    date_format = "%H:%M"
    txt = message.text

    await parse_time_make_job(
        msg=txt,
        state=state,
        apscheduler=apscheduler,
        date_format=date_format,
        _type="every_day",
    )

    # date_string = message.text
    # date_format = "%Y-%m-%d %H:%M"
    # new_datetime = datetime.strptime(date_string, date_format)
    # editing_ad = (await state.get_data()).get("editing_ad")

    # if editing_ad.id in apscheduler.get_jobs():
    #     apscheduler.modify_job(id=editing_ad.id, run_date=new_datetime)
    # else:
    #     apscheduler.add_job(
    #         editing_ad.timed_send,
    #         trigger="date",
    #         run_date=new_datetime,
    #     )


@router.message(Bot_advert.ad_intervals)
async def timer_ad_intervals(
    message: types.Message, state: FSMContext, apscheduler: AsyncIOScheduler
):
    # print("I'm HERE BROOOO (3)")
    date_format = "%H:%M"
    txt = message.text

    await parse_time_make_job(
        msg=txt,
        state=state,
        apscheduler=apscheduler,
        date_format=date_format,
        _type="intervals",
    )

    # date_string = message.text
    # date_format = "%Y-%m-%d %H:%M"
    # new_datetime = datetime.strptime(date_string, date_format)
    # editing_ad = (await state.get_data()).get("editing_ad")

    # if editing_ad.id in apscheduler.get_jobs():
    #     apscheduler.modify_job(id=editing_ad.id, run_date=new_datetime)
    # else:
    #     apscheduler.add_job(
    #         editing_ad.timed_send,
    #         trigger="date",
    #         run_date=new_datetime,
    #     )


@router.message(Bot_advert.ad_edit_info)
async def edit_info(message: types.Message, state: FSMContext):
    ad_type = await check_advert(msg=message, state=state)

    # получение фото и текста под фото
    print(message.photo, message.text)
    # print(message.caption)

    a = (await state.get_data()).get("editing_ad")

    if ad_type == 1:
        await a.edit(message.caption, message.photo, ad_type)
    elif ad_type == 0:
        await a.edit(message.text, [], ad_type)
    else:
        await a.edit("", message.photo, ad_type)

    await a.send(message.from_user)


@router.message(Bot_advert.ad_name)
async def ad_name(message: types.Message, state: FSMContext):
    a = (await state.get_data()).get("ad")
    a.name = message.text
    await state.set_state(Bot_advert.ad_edit)
    await state.update_data(editing_ad=a)
    await message.answer(text.advert_edit, reply_markup=kb.ad_edit_kb)


# for any text before first gen try
@router.message()
async def echo_new(message: types.Message):
    """
    echo bot
    """
    # async for answer in split_text(message.text, max_length=20):
    #     await message.answer(answer)
    print(config.bot)
    # print(message.text)
    # print(message.photo)
    # print(message.caption)
    await message.answer("Это всего-лишь эхо далёких звёзд", reply_markup=kb.menu)
