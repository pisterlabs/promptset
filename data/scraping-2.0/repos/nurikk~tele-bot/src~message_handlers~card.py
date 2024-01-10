from src.prompt_generator import get_depiction_ideas
from src.img import Proxy
import datetime
import logging
import re
from enum import Enum

import i18n
from aiogram import types, Router, Bot, Dispatcher, F
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, InlineKeyboardButton, \
    InlineKeyboardMarkup, SwitchInlineQueryChosenChat, URLInputFile, InputMediaPhoto
from aiogram.utils.chat_action import ChatActionSender
from aiogram.utils.deep_linking import create_start_link
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.markdown import hbold, hpre
from openai import BadRequestError, AsyncOpenAI
from tortoise.functions import Max

from src import db, card_gen
from src.commands import card_command
from src.db import user_from_message, TelebotUsers, CardRequests, CardRequestQuestions, CardRequestsAnswers, CardResult
from src.fsm.card import CardForm
from src.image_generator import ImageGenerator
from src.s3 import S3Uploader
from src.settings import Settings


async def debug_log(request_id: int, bot: Bot,
                    user: TelebotUsers, debug_chat_id: int, s3_uploader: S3Uploader, image_proxy: Proxy):
    card_request = await CardRequests.get(id=request_id)
    answers = await db.CardRequestsAnswers.filter(request_id=request_id).all()
    prompt_data = ''
    for item in answers:
        prompt_data += f"{item.question.value}: {item.answer}\n"
    messages = [
        f"New card for {hbold(user.full_name)} @{user.username}!",
        f"User response: \n {hpre(prompt_data)}",
        f"Generated prompt:\n {hpre(card_request.generated_prompt)}"
    ]
    await bot.send_message(chat_id=debug_chat_id, text="\n".join(messages))

    await send_photos(chat_id=debug_chat_id, request_id=request_id, image_proxy=image_proxy, s3_uploader=s3_uploader, bot=bot)


class Action(str, Enum):
    ACTION_REGENERATE = "regenerate"


class CardGenerationCallback(CallbackData, prefix="my"):
    action: Action
    request_id: int


def generate_image_keyboad(locale: str, request_id: int) -> InlineKeyboardBuilder:
    button_label = i18n.t('regenerate', locale=locale)
    callback_data = CardGenerationCallback(action=Action.ACTION_REGENERATE, request_id=request_id).pack()
    buttons = [
        # [InlineKeyboardButton(text=button_label, callback_data=callback_data)],
        [InlineKeyboardButton(
            text=i18n.t("share_with_friend", locale=locale),
            switch_inline_query_chosen_chat=SwitchInlineQueryChosenChat(allow_user_chats=True,
                                                                        allow_group_chats=True,
                                                                        allow_channel_chats=True,
                                                                        query=str(request_id))
        )]
    ]

    return InlineKeyboardBuilder(markup=buttons)


async def send_photos(chat_id: int, request_id: int, image_proxy: Proxy, s3_uploader: S3Uploader, bot: Bot):
    image_results = await CardResult.filter(request_id=request_id).all()

    photos = [
        InputMediaPhoto(
            media=URLInputFile(url=image_proxy.get_full_image(s3_uploader.get_website_url(image_result.result_image)), filename="card.png")
        )
        for image_result in image_results
    ]

    await bot.send_media_group(chat_id=chat_id, media=photos, protect_content=True)  # reply_markup=keyboard.as_markup()


async def deliver_generated_samples_to_user(request_id: int, bot: Bot, user: TelebotUsers, locale: str,
                                            image_generator: ImageGenerator, debug_chat_id: int,
                                            s3_uploader: S3Uploader, image_proxy: Proxy, async_openai_client: AsyncOpenAI) -> None:
    try:
        async with ChatActionSender.upload_photo(bot=bot, chat_id=user.telegram_id):
            await card_gen.render_card(request_id=request_id, user=user, locale=locale, image_generator=image_generator,
                                       s3_uploader=s3_uploader, async_openai_client=async_openai_client)
            request = await CardRequests.get(id=request_id)
            await bot.send_message(chat_id=user.telegram_id, text=request.greeting_text)
            await send_photos(chat_id=user.telegram_id, request_id=request_id, image_proxy=image_proxy, s3_uploader=s3_uploader, bot=bot)
            keyboard = generate_image_keyboad(locale=locale, request_id=request_id)
            await bot.send_message(chat_id=user.telegram_id, text=i18n.t('share_description', locale=locale), reply_markup=keyboard.as_markup())
            await bot.send_message(chat_id=user.telegram_id, text=i18n.t('commands.card', locale=locale))

            await debug_log(request_id=request_id, bot=bot,
                            user=user,
                            debug_chat_id=debug_chat_id, image_proxy=image_proxy, s3_uploader=s3_uploader)
    except BadRequestError as e:
        if isinstance(e.body, dict) and 'message' in e.body:
            await bot.send_message(chat_id=user.telegram_id, text=e.body['message'])


async def get_samples(question: CardRequestQuestions, locale: str) -> list[str]:
    return i18n.t(f"card_form.{question.value}.samples", locale=locale).split(",")


def generate_samples_keyboard(samples: list[str], columns: int = 2) -> ReplyKeyboardMarkup:
    keyboard = []
    for pair in zip(*[iter(samples)] * columns):
        keyboard.append([KeyboardButton(text=sample) for sample in pair])
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


async def generate_answer_samples_keyboard(locale: str, question: CardRequestQuestions, columns: int = 2) -> ReplyKeyboardMarkup:
    samples = await get_samples(question=question, locale=locale)
    return generate_samples_keyboard(samples=samples, columns=columns)


async def generate_depictions_samples_keyboard(client: AsyncOpenAI, locale: str, request_id: int) -> ReplyKeyboardMarkup:
    samples = await get_depiction_ideas(client=client, locale=locale, request_id=request_id)

    return generate_samples_keyboard(samples=samples, columns=1)


async def generate_descriptions_samples_keyboard(user: TelebotUsers, locale: str, samples_count: int = 5):
    # Refactor this to make DISTINCT ON query
    answers = await CardRequests.filter(user=user,
                                        answers__language_code=locale,
                                        answers__question=CardRequestQuestions.DESCRIPTION
                                        ).annotate(min_created_at=Max('created_at')).order_by("-min_created_at").group_by('answers__answer').limit(
        samples_count).values("answers__answer")

    descriptions = [answer['answers__answer'] for answer in answers]
    if descriptions:
        return generate_samples_keyboard(samples=descriptions, columns=1)
    return ReplyKeyboardRemove()


async def handle_no_more_cards(message: types.Message, user: types.User):
    locale = user.language_code

    kb = [[
        InlineKeyboardButton(
            text=i18n.t("invite_friend", locale=locale),
            switch_inline_query_chosen_chat=SwitchInlineQueryChosenChat(allow_user_chats=True,
                                                                        allow_group_chats=True,
                                                                        allow_channel_chats=True)
        )
    ]]
    await message.answer(
        i18n.t("no_cards_left", locale=locale),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=kb)
    )


async def ensure_user_has_cards(message: types.Message, user: types.User = None) -> bool:
    telebot_user = await user_from_message(telegram_user=user)
    if telebot_user.remaining_cards <= 0:
        await handle_no_more_cards(message=message, user=user)
        return False
    return True


async def generate_reason_samples_keyboard(locale: str):
    reasons = await db.get_near_holidays(country_code=locale, days=7)
    samples = await get_samples(question=CardRequestQuestions.REASON, locale=locale)
    for r in reasons:
        month_name = i18n.t(f"month_names.month_{r.month}", locale=locale)
        samples.append(f"{r.title} ({r.day} {month_name})")
    return generate_samples_keyboard(samples=samples, columns=1)


async def command_start(message: types.Message, state: FSMContext) -> None:
    locale = message.from_user.language_code
    user = await user_from_message(telegram_user=message.from_user)
    if await ensure_user_has_cards(message=message, user=message.from_user):
        request: CardRequests = await CardRequests.create(user=user, language_code=locale)
        await state.update_data(request_id=request.id)
        await state.set_state(CardForm.reason)
        reason_samples_keyboard = await generate_reason_samples_keyboard(locale=locale)
        await message.answer(i18n.t("card_form.reason.response", locale=locale), reply_markup=reason_samples_keyboard)


async def process_reason(message: types.Message, state: FSMContext) -> None:
    locale = message.from_user.language_code
    request_id = (await state.get_data())['request_id']
    await CardRequestsAnswers.create(request_id=request_id, question=CardRequestQuestions.REASON, answer=message.text, language_code=locale)
    await state.set_state(CardForm.description)
    answer_samples_keyboard = await generate_answer_samples_keyboard(
        locale=locale, question=CardRequestQuestions.DESCRIPTION, columns=4)
    await message.answer(i18n.t(f"card_form.{CardRequestQuestions.DESCRIPTION.value}.response", locale=locale), reply_markup=answer_samples_keyboard)


async def process_description(message: types.Message, state: FSMContext, async_openai_client: AsyncOpenAI, bot: Bot) -> None:
    locale = message.from_user.language_code
    request_id = (await state.get_data())['request_id']
    await CardRequestsAnswers.create(request_id=request_id, question=CardRequestQuestions.DESCRIPTION, answer=message.text, language_code=locale)
    await state.set_state(CardForm.depiction)

    await message.answer(i18n.t("card_form.depiction.coming_up_with_ideas", locale=locale), reply_markup=ReplyKeyboardRemove())
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        depiction_ideas = await generate_depictions_samples_keyboard(locale=locale, request_id=request_id, client=async_openai_client)
        await message.answer(i18n.t(f"card_form.{CardRequestQuestions.DEPICTION.value}.response", locale=locale), reply_markup=depiction_ideas)


async def process_depiction(message: types.Message, state: FSMContext, bot: Bot, settings: Settings,
                            s3_uploader: S3Uploader, image_proxy: Proxy,
                            image_generator: ImageGenerator, async_openai_client: AsyncOpenAI) -> None:
    user = await user_from_message(telegram_user=message.from_user)
    locale = message.from_user.language_code
    request_id = (await state.get_data())['request_id']
    await CardRequestsAnswers.create(request_id=request_id, question=CardRequestQuestions.DEPICTION, answer=message.text, language_code=locale)
    await state.set_state(CardForm.style)

    await message.answer(i18n.t("card_form.wait.response", locale=locale), reply_markup=ReplyKeyboardRemove())
    await state.clear()
    await deliver_generated_samples_to_user(request_id=request_id, bot=bot, user=user, locale=locale,
                                            image_generator=image_generator, debug_chat_id=settings.debug_chat_id, s3_uploader=s3_uploader,
                                            image_proxy=image_proxy,
                                            async_openai_client=async_openai_client)


async def regenerate(query: CallbackQuery, callback_data: CardGenerationCallback, bot: Bot,
                     settings: Settings,
                     s3_uploader: S3Uploader, image_proxy: Proxy, image_generator: ImageGenerator, async_openai_client: AsyncOpenAI):
    if await ensure_user_has_cards(message=query.message, user=query.from_user):
        user = await user_from_message(telegram_user=query.from_user)
        locale = query.from_user.language_code
        await query.answer(text=i18n.t("card_form.wait.response", locale=locale))
        await deliver_generated_samples_to_user(request_id=callback_data.request_id, bot=bot, user=user, locale=locale,
                                                image_generator=image_generator, debug_chat_id=settings.debug_chat_id, s3_uploader=s3_uploader,
                                                image_proxy=image_proxy, async_openai_client=async_openai_client)


def escape_markdown(text: str) -> str:
    return re.sub(r'([_*[\]()~`>#+-=|{}.!])', r'\\\1', text)


def get_message_content(locale: str, reason: CardRequestsAnswers, full_name: str, photo_url: str, greeting_text: str) -> str:
    return i18n.t('share_message_content_markdown',
                  locale=locale,
                  reason=escape_markdown(reason.answer),
                  name=escape_markdown(full_name),
                  photo_url=photo_url,
                  greeting_message=escape_markdown(greeting_text) if greeting_text else '')


async def inline_query(query: types.InlineQuery, bot: Bot,
                       s3_uploader: S3Uploader,
                       image_proxy: Proxy) -> None:
    user = await user_from_message(telegram_user=query.from_user)
    link = await create_start_link(bot, str(user.id))
    request_id = query.query
    results = []
    request_qs = CardRequests.filter(user=user).prefetch_related('results').order_by("-created_at")
    if request_id:
        request_qs = request_qs.filter(id=request_id)
    requests = await request_qs.limit(10)
    reply_markup = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text=i18n.t("generate_your_own", locale=query.from_user.language_code), url=link)
        ]]
    )

    thumbnail_width = 256
    thumbnail_height = 256
    for request in requests:
        reason = await CardRequestsAnswers.filter(request_id=request.id, question=CardRequestQuestions.REASON).first()

        for result in request.results:
            photo_url = image_proxy.get_full_image(s3_uploader.get_website_url(result.result_image))
            thumbnail_url = image_proxy.get_thumbnail(s3_uploader.get_website_url(result.result_image), width=thumbnail_width,
                                                      height=thumbnail_height)

            logging.info(f"{photo_url=} {thumbnail_url=}")
            results.append(types.InlineQueryResultArticle(
                id=str(datetime.datetime.now()),
                title=i18n.t('shared_title', locale=query.from_user.language_code, name=query.from_user.full_name),
                description=i18n.t('shared_description', locale=query.from_user.language_code, name=query.from_user.full_name, reason=reason.answer),
                thumbnail_width=thumbnail_width,
                thumbnail_height=thumbnail_height,
                thumbnail_url=thumbnail_url,
                input_message_content=types.InputTextMessageContent(
                    message_text=get_message_content(locale=query.from_user.language_code, reason=reason,
                                                     full_name=query.from_user.full_name,
                                                     photo_url=photo_url,
                                                     greeting_text=request.greeting_text),
                    parse_mode="MarkdownV2",
                ),
                caption=i18n.t('shared_from', locale=query.from_user.language_code, name=query.from_user.full_name),
                reply_markup=reply_markup,
            ))

    await query.answer(results=results, cache_time=0)


async def chosen_inline_result_handler(chosen_inline_result: types.ChosenInlineResult):
    request_id = chosen_inline_result.query
    if request_id:
        from tortoise.expressions import F
        await db.CardRequests.filter(id=request_id).update(shares_count=F("shares_count") + 1)


async def edited_message_handler(edited_message: types.Message):
    pass


def register(dp: Dispatcher):
    form_router = Router()
    form_router.message(card_command)(command_start)
    form_router.message(CardForm.reason)(process_reason)
    form_router.message(CardForm.description)(process_description)
    form_router.message(CardForm.depiction)(process_depiction)
    form_router.callback_query(CardGenerationCallback.filter(F.action == Action.ACTION_REGENERATE))(regenerate)

    form_router.edited_message()(edited_message_handler)

    form_router.inline_query()(inline_query)
    form_router.chosen_inline_result()(chosen_inline_result_handler)

    dp.include_router(form_router)
