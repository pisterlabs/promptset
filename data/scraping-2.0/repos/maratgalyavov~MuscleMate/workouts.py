from telegram import ReplyKeyboardRemove, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
)
import openai
import httpx
import translators as trans

import config
from menu_handler import show_main_menu


async def muscle_group_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Callback for selecting a muscle group, displaying intensity options to the user.
    """
    from main import logger
    user = update.effective_user
    dest_lang = context.user_data['lang']
    query = update.callback_query
    muscle_group = query.data

    logger.info(f"User {user.id} selected muscle group: {muscle_group}")
    context.user_data['muscle_group'] = muscle_group

    low_int = trans.translate_text(query_text="Low intensity", translator='google', to_language=dest_lang)
    med_int = trans.translate_text(query_text="Medium intensity", translator='google', to_language=dest_lang)
    high_int = trans.translate_text(query_text="High intensity", translator='google', to_language=dest_lang)
    int_mes = trans.translate_text(query_text='Choose an intensity level:', translator='google',
                                   to_language=dest_lang)
    keyboard = [
        [InlineKeyboardButton(low_int, callback_data='low'),
         InlineKeyboardButton(med_int, callback_data='medium'),
         InlineKeyboardButton(high_int, callback_data='high')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text(
        int_mes,
        reply_markup=reply_markup)
    return config.INTENSITY


async def workout_area_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Callback for selecting workout area, offering single or compound muscle group choices.
    """
    from main import logger
    user = update.effective_user
    dest_lang = context.user_data['lang']
    query = update.callback_query
    workout_type = query.data
    logger.info(f"User {user.id} selected workout type: {workout_type}")

    chest = trans.translate_text(query_text="Chest", translator='google', to_language=dest_lang)
    back = trans.translate_text(query_text="Back", translator='google', to_language=dest_lang)
    legs = trans.translate_text(query_text="Legs", translator='google', to_language=dest_lang)
    arms = trans.translate_text(query_text="Arms", translator='google', to_language=dest_lang)
    butt = trans.translate_text(query_text="Butt", translator='google', to_language=dest_lang)
    abs_ = trans.translate_text(query_text="Abs", translator='google', to_language=dest_lang)
    muscle_mes = trans.translate_text(query_text='Choose a muscle group:', translator='google', to_language=dest_lang)
    full = trans.translate_text(query_text="Full Body", translator='google', to_language=dest_lang)
    lower = trans.translate_text(query_text="Lower Body", translator='google', to_language=dest_lang)
    upper = trans.translate_text(query_text="Upper Body", translator='google', to_language=dest_lang)
    workout_mes = trans.translate_text(query_text="Choose a workout:", translator='google', to_language=dest_lang)

    if workout_type == 'single_muscle_group':
        keyboard = [
            [InlineKeyboardButton(chest, callback_data='chest'),
             InlineKeyboardButton(back, callback_data='back')],
            [InlineKeyboardButton(legs, callback_data='legs'),
             InlineKeyboardButton(arms, callback_data='arms')],
            [InlineKeyboardButton(butt, callback_data='butt'),
             InlineKeyboardButton(abs_, callback_data='abs')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text(
            muscle_mes,
            reply_markup=reply_markup)
        return config.MUSCLE_GROUP
    else:  # compound workouts
        keyboard = [
            [InlineKeyboardButton(full, callback_data='full_body'),
             InlineKeyboardButton(upper, callback_data='upper_body'),
             InlineKeyboardButton(lower, callback_data='lower_body')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text(
            workout_mes,
            reply_markup=reply_markup)
        return config.MUSCLE_GROUP


async def intensity_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Callback for selecting workout intensity, generates and translates workout plan.
    """
    from main import logger
    try:
        user = update.effective_user
        dest_lang = context.user_data['lang']
        query = update.callback_query
        intensity = query.data
        logger.info(f"User {user.id} selected intensity: {intensity}")
        muscle_group = context.user_data['muscle_group']

        proc_mes = trans.translate_text(query_text='Processing your request. This may take a couple of minutes...',
                                        translator='google', to_language=dest_lang)
        feedback_mes = trans.translate_text(query_text='How did you like the workout plan?', translator='google',
                                            to_language=dest_lang)

        await context.bot.send_chat_action(chat_id=query.message.chat_id, action='typing')

        processing_message = await query.message.reply_text(proc_mes)

        async with httpx.AsyncClient(timeout=12000000.0) as client:  # Use httpx.AsyncClient to make asynchronous requests
            if muscle_group in ['chest', 'back', 'legs', 'arms', 'butt', 'abs']:
                # Single muscle group workout request
                request_content = f"Provide a workout plan for {muscle_group} muscles with {intensity} intensity."
            else:
                # Compound workout request
                request_content = f"Provide a {muscle_group} workout plan with {intensity} intensity."

            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a fitness coach."},
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
            [InlineKeyboardButton("Good", callback_data='good'),
             InlineKeyboardButton("Bad", callback_data='bad')]
        ]
        feedback_markup = InlineKeyboardMarkup(feedback_keyboard)
        await query.message.reply_text(feedback_mes, reply_markup=feedback_markup)
        return config.FEEDBACK
    except Exception as e:
        logger.error(f"Error in muscle_group_callback: {e}")


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Callback for user feedback on workout plan, logs feedback and returns to main menu.
    """
    from main import logger
    user = update.effective_user
    dest_lang = context.user_data['lang']
    query = update.callback_query
    feedback = query.data  # This will be 'good' or 'bad' based on the button pressed
    logger.info(f"User {user.id} gave feedback: {feedback}")
    await query.message.reply_text(
        trans.translate_text(query_text="Thank you for your feedback! Redirecting to the main menu.",
                             translator='google', to_language=dest_lang))
    return await show_main_menu(update, context, query.message)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handles conversation cancellation, logs event and sends farewell message to user.
    """
    from main import logger
    user = update.message.from_user
    dest_lang = context.user_data['lang']
    logger.info(f"User {user.id} canceled the conversation.")
    cancel_mes = trans.translate_text(query_text="Bye! I hope we can talk again some day.", translator='google',
                                      to_language=dest_lang)
    await update.message.reply_text(cancel_mes, reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END
