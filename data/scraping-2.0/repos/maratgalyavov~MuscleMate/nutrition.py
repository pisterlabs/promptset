from telegram import ReplyKeyboardRemove, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
)
import openai
import httpx
import translators as trans
import re

import config
from menu_handler import show_main_menu
from db import add_record


async def nutrition_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handles nutrition-related callbacks, offering meal planning or description input for calorie counting.
    """
    from main import logger
    try:
        user = update.effective_user
        dest_lang = context.user_data['lang']
        query = update.callback_query
        choice = query.data
        logger.info(f"User {user.id} selected {choice}")

        if choice == 'meal_planning':
            duration_mes = trans.translate_text(
                query_text='select the preffered duration of the meal plan',
                translator='google', to_language=dest_lang)

            one = trans.translate_text(query_text='1', translator='google', to_language=dest_lang)
            three = trans.translate_text(query_text='3', translator='google', to_language=dest_lang)
            five = trans.translate_text(query_text='5', translator='google', to_language=dest_lang)
            seven = trans.translate_text(query_text='7', translator='google', to_language=dest_lang)

            start_keyboard = [
                [InlineKeyboardButton(one, callback_data='1'),
                 InlineKeyboardButton(three, callback_data='3'),
                 InlineKeyboardButton(five, callback_data='5'),
                 InlineKeyboardButton(seven, callback_data='7')]
            ]
            duration_markup = InlineKeyboardMarkup(start_keyboard)
            await query.message.reply_text(duration_mes, reply_markup=duration_markup)
            return config.PLAN
        else:
            meal_description = trans.translate_text(
                query_text='Describe your meal in as much detail as possible',
                translator='google', to_language=dest_lang)
            await query.message.reply_text(meal_description)
            return config.COOKING
    except Exception as e:
        logger.error(f"Error in nutrition_callback: {e}")


async def counting_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Processes user's meal description for calorie counting and provides feedback options.
    """
    from main import logger
    try:
        user = update.message.from_user
        dest_lang = context.user_data['lang']
        meal_description = update.message.text
        query = update.callback_query
        meal_description = trans.translate_text(query_text=meal_description, translator='google', to_language=dest_lang)
        logger.info(f"User {user.id} queried calorie counting")

        proc_mes = trans.translate_text(query_text='Processing your request. This may take a couple of minutes...',
                                        translator='google', to_language=dest_lang)
        feedback_mes = trans.translate_text(query_text='How did you like the response?', translator='google',
                                            to_language=dest_lang)

        await context.bot.send_chat_action(chat_id=update.message.chat_id, action='typing')

        processing_message = await update.message.reply_text(proc_mes)

        async with httpx.AsyncClient(
                timeout=120000000.0) as client:  # Use httpx.AsyncClient to make asynchronous requests
            request_content = f"aproximate calorie count of {meal_description}, your response should contain only an " \
                              f"integerr representig calorie estimate and absolutely, i repeat, absolutely no letters "
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are an expert nutritionist."},
                        {"role": "user", "content": request_content}
                    ],
                },
            )
        response_data = response.json()
        workout_suggestion = response_data['choices'][0]['message']['content'].strip()
        workout_suggestion = trans.translate_text(query_text=workout_suggestion, translator='google',
                                                  to_language=dest_lang)
        kcal = re.findall('[0-9]+', workout_suggestion)[0]
        mes = trans.translate_text(query_text=f'This meal estimation is {kcal} kilocalories', translator='google',
                                   to_language=dest_lang)
        await processing_message.edit_text(mes)
        await add_record(user.id, 'kcal', int(kcal))

        feedback_keyboard = [
            [InlineKeyboardButton(trans.translate_text(query_text='Good', translator='google',
                                                       to_language=dest_lang), callback_data='good'),
             InlineKeyboardButton(trans.translate_text(query_text='Bad', translator='google',
                                                       to_language=dest_lang), callback_data='bad')]
        ]
        feedback_markup = InlineKeyboardMarkup(feedback_keyboard)
        await update.message.reply_text(feedback_mes, reply_markup=feedback_markup)
        return config.FEEDBACK
    except Exception as e:
        logger.error(f"Error in counting_callback: {e}")


async def plan_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Generates a meal plan based on selected duration and provides feedback options.
    """
    from main import logger
    try:
        user = update.effective_user
        dest_lang = context.user_data['lang']
        query = update.callback_query
        duration = query.data
        logger.info(f"User {user.id} selected meal plan of {duration} days")

        proc_mes = trans.translate_text(query_text='Processing your request. This may take a couple of minutes...',
                                        translator='google', to_language=dest_lang)
        feedback_mes = trans.translate_text(query_text='How did you like the response?', translator='google',
                                            to_language=dest_lang)

        await context.bot.send_chat_action(chat_id=query.message.chat_id, action='typing')

        processing_message = await query.message.reply_text(proc_mes)

        async with httpx.AsyncClient(
                timeout=12000000.0) as client:  # Use httpx.AsyncClient to make asynchronous requests
            request_content = f"Provide a detailed easy to follow healthy meal plan for {duration} days."
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are an expert nutritionist."},
                        {"role": "user", "content": request_content}
                    ],
                },
            )
        response_data = response.json()
        workout_suggestion = response_data['choices'][0]['message']['content'].strip()
        workout_suggestion = trans.translate_text(query_text=workout_suggestion, translator='google',
                                                  to_language=dest_lang)
        await processing_message.edit_text(workout_suggestion)

        feedback_keyboard = [
            [InlineKeyboardButton(trans.translate_text(query_text='Good', translator='google',
                                                       to_language=dest_lang), callback_data='good'),
             InlineKeyboardButton(trans.translate_text(query_text='Bad', translator='google',
                                                       to_language=dest_lang), callback_data='bad')]
        ]
        feedback_markup = InlineKeyboardMarkup(feedback_keyboard)
        await query.message.reply_text(feedback_mes, reply_markup=feedback_markup)
        return config.FEEDBACK
    except Exception as e:
        logger.error(f"Error in plan_callback: {e}")


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handles feedback for the nutrition or meal plan response and navigates back to the main menu.
    """
    from main import logger
    try:
        user = update.effective_user
        dest_lang = context.user_data['lang']
        query = update.callback_query
        feedback = query.data  # This will be 'good' or 'bad' based on the button pressed
        logger.info(f"User {user.id} gave feedback: {feedback}")
        await query.message.reply_text(
            trans.translate_text(query_text="Thank you for your feedback! Redirecting to the main menu.",
                                 translator='google', to_language=dest_lang))
        return await show_main_menu(update, context, query.message)
    except Exception as e:
        logger.error(f"Error in feedback_callback: {e}")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Manages cancellation of the current conversation and sends a farewell message to the user.
    """
    from main import logger
    user = update.message.from_user
    dest_lang = context.user_data['lang']
    logger.info(f"User {user.id} canceled the conversation.")
    cancel_mes = trans.translate_text(query_text="Bye! I hope we can talk again some day.", translator='google',
                                      to_language=dest_lang)
    await update.message.reply_text(cancel_mes, reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END
