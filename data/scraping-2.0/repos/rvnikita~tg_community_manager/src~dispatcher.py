import src.logging_helper as logging
import src.openai_helper as openai_helper
import src.chat_helper as chat_helper
import src.db_helper as db_helper
import src.config_helper as config_helper
import src.user_helper as user_helper
import src.rating_helper as rating_helper

import os
from telegram import Bot
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ChatJoinRequestHandler
from telegram.request import HTTPXRequest
from telegram.error import TelegramError
import openai
import traceback
import re
import asyncio
import signal
import sys

from langdetect import detect
import langdetect


from datetime import datetime

config = config_helper.get_config()

logger = logging.get_logger()

logger.info(f"Starting {__file__} in {config['BOT']['MODE']} mode at {os.uname()}")

bot = Bot(config['BOT']['KEY'],
          request=HTTPXRequest(http_version="1.1"), #we need this to fix bug https://github.com/python-telegram-bot/python-telegram-bot/issues/3556
          get_updates_request=HTTPXRequest(http_version="1.1")) #we need this to fix bug https://github.com/python-telegram-bot/python-telegram-bot/issues/3556)

########################

async def send_message_to_admin(bot, chat_id, text: str, disable_web_page_preview: bool = True):
    chat_administrators = await bot.get_chat_administrators(chat_id)

    for admin in chat_administrators:
        if admin.user.is_bot == True: #don't send to bots
            continue
        try:
            await chat_helper.send_message(bot, admin.user.id, text, disable_web_page_preview = True)
        except TelegramError as error:
            if error.message == "Forbidden: bot was blocked by the user":
                logger.info(f"Bot was blocked by the user {admin.user.id}.")
            elif error.message == "Forbidden: user is deactivated":
                logger.info(f"User {admin.user.id} is deactivated.")
            elif error.message == "Forbidden: bot can't initiate conversation with a user":
                logger.info(f"Bot can't initiate conversation with a user {admin.user.id}.")
            else:
                logger.error(f"Telegram error: {error.message}. Traceback: {traceback.format_exc()}")
        except Exception as error:
            logger.error(f"Error: {traceback.format_exc()}")

async def tg_report_reset(update, context):
    try:
        with db_helper.session_scope() as db_session:
            #TODO:HIGH: this command reset all reports for this user. It could work when it is replied to message or followed by nickname. It could be done only by chat admin.
            chat_id = update.effective_chat.id
            message = update.message

            chat_administrators = await bot.get_chat_administrators(chat_id)
            is_admin = False

            for admin in chat_administrators:
                if admin.user.id == message.from_user.id:
                    is_admin = True
                    break

            if not is_admin:
                await chat_helper.send_message(bot, chat_id, "You are not an admin of this chat.")
                return

            if message.reply_to_message:
                reported_user_id = message.reply_to_message.from_user.id
            else:
                #we need to take user nickname and then find user_id
                #TODO:HIGH: implement this
                return

            reports = db_session.query(db_helper.Report).filter(db_helper.Report.reported_user_id == reported_user_id).all()
            for report in reports:
                db_session.delete(report)

            db_session.commit()

            await chat_helper.send_message(bot, chat_id, "Reports for this user were reset.")
    except Exception as error:
        logger.error(f"Error: {traceback.format_exc()}")


async def tg_report(update, context):
    try:
        with db_helper.session_scope() as db_session:
            chat_id = update.effective_chat.id

            message = update.message

            asyncio.create_task(chat_helper.delete_message(bot, chat_id, message.message_id, delay_seconds = 120)) #delete report in 2 minutes

            if not message:
                logger.info(f"Could not get message from update {update}. Traceback: {traceback.format_exc()}. Update: {update}")
                return


            if message and message.reply_to_message:
                number_of_reports_to_warn = int(chat_helper.get_chat_config(chat_id, 'number_of_reports_to_warn'))
                number_of_reports_to_ban = int(chat_helper.get_chat_config(chat_id, 'number_of_reports_to_ban'))

                reported_message_id = message.reply_to_message.message_id
                reported_user_id = message.reply_to_message.from_user.id
                reporting_user_id = message.from_user.id

                # Check if the reported user is an admin of the chat
                chat_administrators = await bot.get_chat_administrators(chat_id)
                for admin in chat_administrators:
                    if admin.user.id == reported_user_id:
                        await chat_helper.send_message(bot, chat_id, "You cannot report an admin.")
                        return

                # Check if the user has already been warned by the same reporter
                existing_report = db_session.query(db_helper.Report).filter(
                    db_helper.Report.chat_id == chat_id,
                    db_helper.Report.reported_user_id == reported_user_id,
                    db_helper.Report.reporting_user_id == reporting_user_id
                ).first()

                if not existing_report or reporting_user_id in [admin.user.id for admin in chat_administrators]:
                    # Add report to the database
                    report = db_helper.Report(
                        reported_user_id=reported_user_id,
                        reporting_user_id=reporting_user_id,
                        reported_message_id=reported_message_id,
                        chat_id=chat_id
                    )
                    db_session.add(report)
                    db_session.commit()
                else:
                    await chat_helper.send_message(bot, chat_id, "You have already reported this user.")
                    return

                # Count unique reports for the reported user in the chat
                report_count = db_session.query(db_helper.Report).filter(
                    db_helper.Report.chat_id == chat_id,
                    db_helper.Report.reported_user_id == reported_user_id
                ).count()

                reported_user_mention = user_helper.get_user_mention(reported_user_id)
                reporting_user_mention = user_helper.get_user_mention(reporting_user_id)

                await send_message_to_admin(bot, chat_id, f"User {reporting_user_mention} reported {reported_user_mention} in chat {await chat_helper.get_chat_mention(bot, chat_id)}. Total reports: {report_count}. \nReported message: {message.reply_to_message.text}")

                await chat_helper.send_message(bot, chat_id, f"User {reported_user_mention} has been reported {report_count}/{number_of_reports_to_ban} times.", reply_to_message_id=reported_message_id, delete_after=120)

                if report_count >= number_of_reports_to_ban:
                    await chat_helper.delete_message(bot, chat_id, reported_message_id)
                    await chat_helper.ban_user(bot, chat_id, reported_user_id)

                    await chat_helper.send_message(bot, chat_id, f"User {reported_user_mention} has been banned due to {report_count} reports.", delete_after=120)

                    await send_message_to_admin(bot, chat_id, f"User {reported_user_mention} has been banned in chat {await chat_helper.get_chat_mention(bot, chat_id)} due to {report_count}/{number_of_reports_to_ban} reports.")

                    # let's now increase rating for all users who reported this user
                    # first let's get all users who reported this user
                    reporting_user_ids = db_session.query(db_helper.Report.reporting_user_id).filter(db_helper.Report.reported_user_id == reported_user_id, db_helper.Report.chat_id == chat_id).distinct().all()
                    reporting_user_ids = [item[0] for item in reporting_user_ids]

                    bot_info = await bot.get_me()

                    # Instead of looping here, just call the function once with the entire list
                    await rating_helper.change_rating(reporting_user_ids, bot_info.id, chat_id, 1)

                    return # we don't need to warn and mute user if he is banned

                if report_count >= number_of_reports_to_warn:
                    await chat_helper.warn_user(bot, chat_id, reported_user_id)
                    await chat_helper.mute_user(bot, chat_id, reported_user_id)

                    await chat_helper.send_message(bot, chat_id, f"User {reported_user_mention} has been warned and muted due to {report_count} reports.", reply_to_message_id=reported_message_id, delete_after=120)
                    await send_message_to_admin(bot, chat_id, f"User {reported_user_mention} has been warned and muted in chat {await chat_helper.get_chat_mention(bot, chat_id)} due to {report_count} reports.")
    except Exception as error:
        logger.error(f"Error: {traceback.format_exc()}")

async def tg_warn(update, context):
    try:
        chat_id = update.effective_chat.id
        message = update.message
        admin_ids = [admin.user.id for admin in await bot.get_chat_administrators(chat_id)]

        chat_helper.delete_message(bot, chat_id, message.message_id, delay_seconds=120)  # clean up the command message

        if message.from_user.id not in admin_ids:
            await chat_helper.send_message(bot, chat_id, "You must be an admin to use this command.", reply_to_message_id=message.message_id, delete_after = 120)
            return

        if not message.reply_to_message:
            await chat_helper.send_message(bot, chat_id, "Reply to a message to warn the user.", reply_to_message_id=message.message_id, delete_after = 120)
            return

        reason = ' '.join(message.text.split()[1:]) or "You've been warned by an admin."
        warned_user_id = message.reply_to_message.from_user.id
        warned_message_id = message.reply_to_message.message_id

        with db_helper.session_scope() as db_session:
            report = db_helper.Report(
                reported_user_id=warned_user_id,
                reporting_user_id=message.from_user.id,
                reported_message_id=warned_message_id,
                chat_id=chat_id,
                reason=reason
            )
            db_session.add(report)
            db_session.commit()

            warn_count = db_session.query(db_helper.Report).filter(
                db_helper.Report.chat_id == chat_id,
                db_helper.Report.reported_user_id == warned_user_id,
                db_helper.Report.reason != None
            ).count()

            number_of_reports_to_ban = int(chat_helper.get_chat_config(chat_id, 'number_of_reports_to_ban'))

            if warn_count >= number_of_reports_to_ban:
                await chat_helper.delete_message(bot, chat_id, warned_message_id)
                await chat_helper.ban_user(bot, chat_id, warned_user_id)
                warned_user_mention = user_helper.get_user_mention(warned_user_id)
                await chat_helper.send_message(bot, chat_id, f"User {warned_user_mention} has been banned due to {warn_count} warnings.", delete_after=120)
                await send_message_to_admin(bot, chat_id, f"User {warned_user_mention} has been banned in chat {await chat_helper.get_chat_mention(bot, chat_id)} due to {warn_count}/{number_of_reports_to_ban} warnings.")

                reporting_user_ids = db_session.query(db_helper.Report.reporting_user_id).filter(
                    db_helper.Report.reported_user_id == warned_user_id,
                    db_helper.Report.chat_id == chat_id
                ).distinct().all()
                reporting_user_ids = [item[0] for item in reporting_user_ids]

                bot_info = await bot.get_me()
                for user_id in reporting_user_ids:
                    await rating_helper.change_rating(user_id, bot_info.id, chat_id, 1)

                return

            warned_user_mention = user_helper.get_user_mention(warned_user_id)
            warning_admin_mention = user_helper.get_user_mention(message.from_user.id)

            await chat_helper.send_message(bot, chat_id, f"{warned_user_mention}, you've been warned {warn_count}/{number_of_reports_to_ban} times. Reason: {reason}", reply_to_message_id=warned_message_id)
            await chat_helper.delete_message(bot, chat_id, warned_message_id)
            await send_message_to_admin(bot, chat_id, f"{warning_admin_mention} warned {warned_user_mention} in chat {await chat_helper.get_chat_mention(bot, chat_id)}. Reason: {reason}. Total Warnings: {warn_count}/{number_of_reports_to_ban}")

    except Exception as error:
        logger.error(f"Error in warn: {traceback.format_exc()}")


async def tg_ban(update, context):
    try:
        with db_helper.session_scope() as db_session:
            message = update.message
            chat_id = update.effective_chat.id
            ban_user_id = None

            await chat_helper.delete_message(bot, chat_id, message.message_id)  # clean up the command message

            # Check if the command was sent by an admin of the chat
            chat_administrators = await bot.get_chat_administrators(chat_id)
            if message.from_user.id not in [admin.user.id for admin in chat_administrators]:
                await chat_helper.send_message(bot, chat_id, "You must be an admin to use this command.", reply_to_message_id=message.message_id, delete_after = 120)
                return

            command_parts = message.text.split()  # Split the message into parts
            if len(command_parts) > 1:  # if the command has more than one part (means it has a user ID or username parameter)
                if '@' in command_parts[1]:  # if the second part is a username
                    user = db_session.query(db_helper.User).filter(db_helper.User.username == command_parts[1][1:]).first()  # Remove @ and query
                    if user is None:
                        await message.reply_text(f"No user found with username {command_parts[1]}.")
                        return
                    ban_user_id = user.id
                elif command_parts[1].isdigit():  # if the second part is a user ID
                    ban_user_id = int(command_parts[1])
                else:
                    await message.reply_text("Invalid format. Use /ban @username or /ban user_id.")
                    return
            else: # Check if a user is mentioned in the command message as a reply to message
                if not message.reply_to_message:
                    await message.reply_text("Please reply to a user's message to ban them.")
                    return
                ban_user_id = message.reply_to_message.from_user.id

                await chat_helper.delete_message(bot, chat_id, message.reply_to_message.message_id)

            # Check if the user to ban is an admin of the chat
            for admin in chat_administrators:
                if admin.user.id == ban_user_id:
                    await message.reply_text("You cannot ban an admin.")
                    return

            # Ban the user
            await chat_helper.ban_user(bot, chat_id, ban_user_id)

            await chat_helper.send_message(bot, chat_id, f"User {user_helper.get_user_mention(ban_user_id)} has been banned.", delete_after=120)

    except Exception as error:
        logger.error(f"Error: {traceback.format_exc()}")

async def tg_gban(update, context):
    try:
        with db_helper.session_scope() as db_session:
            message = update.message
            chat_id = update.effective_chat.id
            ban_user_id = None
            ban_chat_id = None
            ban_reason = None

            # Check if the command was sent by a global admin of the bot
            if message.from_user.id != int(config['BOT']['ADMIN_ID']):
                await message.reply_text("You must be a global bot admin to use this command.")
                return

            command_parts = message.text.split()  # Split the message into parts
            if len(command_parts) > 1:  # if the command has more than one part (means it has a user ID or username parameter)
                ban_reason = f"User was globally banned by {message.text} command."
                if '@' in command_parts[1]:  # if the second part is a username
                    user = db_session.query(db_helper.User).filter(db_helper.User.username == command_parts[1][1:]).first()  # Remove @ and query
                    if user is None:
                        await message.reply_text(f"No user found with username {command_parts[1]}.")
                        return
                    ban_user_id = user.id
                elif command_parts[1].isdigit():  # if the second part is a user ID
                    ban_user_id = int(command_parts[1])
                else:
                    await message.reply_text("Invalid format. Use gban @username or gban user_id.")
                    return
            elif chat_id == int(config['LOGGING']['INFO_CHAT_ID']) or chat_id == int(
                    config['LOGGING']['ERROR_CHAT_ID']):
                ban_reason = f"User was globally banned by {message.text} command in info chat. Message: {message.reply_to_message.text}"
                if not message.reply_to_message:
                    await message.reply_text("Please reply to a message containing usernames to ban.")
                    return
                username_list = re.findall('@(\w+)',
                                           message.reply_to_message.text)  # extract usernames from the reply_to_message
                if len(username_list) > 2:  # Check if there are more than 2 usernames in the message
                    await message.reply_text("More than two usernames found. Please specify which user to ban.")
                    return
                elif len(username_list) == 0:  # Check if there are no usernames in the message
                    await message.reply_text("No usernames found. Please specify which user to ban.")
                    return
                else:  # There is exactly one username
                    # Fetch user_id based on username from database
                    user = db_session.query(db_helper.User).filter(db_helper.User.username == username_list[0]).first()
                    if user is None:
                        await message.reply_text(f"No user found with username {username_list[0]}.")
                        return
                    ban_user_id = user.id
            else: # Check if a user is mentioned in the command message as a reply to message
                ban_reason = f"User was globally banned by {message.text} command in {chat_helper.get_chat_mention(bot, chat_id)}. Message: {message.reply_to_message.text}"
                ban_chat_id = chat_id # We need to ban in the same chat as the command was sent

                if not message.reply_to_message:
                    await message.reply_text("Please reply to a user's message to ban them.")
                    return
                ban_user_id = message.reply_to_message.from_user.id

                await chat_helper.delete_message(bot, chat_id, message.reply_to_message.message_id)

            #await chat_helper.delete_message(bot, chat_id, message.message_id) # delete the ban command message

            # Ban the user and add them to the banned_users table
            await chat_helper.ban_user(bot, ban_chat_id, ban_user_id, True, reason=ban_reason)

            # await bot.send_message(chat_id, f"User {user_helper.get_user_mention(ban_user_id)} has been globally banned for spam.")
            await bot.delete_message(chat_id, message.message_id) # delete the ban command message
    except Exception as error:
        logger.error(f"Error: {traceback.format_exc()}")




async def tg_spam_check(update, context):
    try:
        if update.message is not None:
            agressive_antispam = chat_helper.get_chat_config(update.message.chat.id, "agressive_antispam")
        else:
            logger.warning("Update does not contain a message")
            return

        if agressive_antispam == True:
            # TODO:HIGH: This is a very temporary antispam check. We need to implement a better solution (e.g. with a machine learning model or OpenAI's GPT-4)
            if update.message and update.message.text:
                text = update.message.text.strip()
                if text:
                    try:
                        lang = detect(text)
                        if lang in ['ar', 'fa', 'ur', 'he', 'ps', 'sd', 'ku', 'ug', 'fa', 'zh']:
                            # Ban the user for using Arabic or Persian language
                            await chat_helper.delete_message(bot, update.message.chat.id, update.message.message_id)
                            await chat_helper.ban_user(bot, update.message.chat.id, update.message.from_user.id, reason=f"Filtered language used. Message {update.message.text}. Chat: {await chat_helper.get_chat_mention(bot, update.message.chat.id)}", global_ban=True)
                            await chat_helper.send_message(bot, update.message.chat.id, f"User {user_helper.get_user_mention(update.message.from_user.id)} has been banned based on language filter. - {lang}", delete_after=120)
                            return  # exit the function as the user has already been banned
                    except langdetect.lang_detect_exception.LangDetectException as e:
                        if "No features in text." in str(e):
                            # No features in text
                            pass

            # Check for APK files
            if update.message and update.message.document:
                if update.message.document.file_name.endswith('.apk'):
                    # Ban the user for sending an APK file
                    await chat_helper.delete_message(bot, update.message.chat.id, update.message.message_id)
                    await chat_helper.ban_user(bot, update.message.chat.id, update.message.from_user.id, reason=f"APK file uploaded. Chat: {await chat_helper.get_chat_mention(bot, update.message.chat.id)}", global_ban=True)
                    await chat_helper.send_message(bot, update.message.chat.id, f"User {user_helper.get_user_mention(update.message.from_user.id)} has been banned for uploading an APK file.", delete_after=120)
                    return  # exit the function as the user has already been banned


    except Exception as error:
        logger.error(f"Error: {traceback.format_exc()}")


async def tg_thankyou(update, context):
    try:
        with db_helper.session_scope() as db_session:

            if update.message is not None \
                    and update.message.reply_to_message is not None \
                    and update.message.reply_to_message.from_user.id != update.message.from_user.id:

                # there is a strange behaviour when user send message in topic Telegram show it as a reply to forum_topic_created invisible message. We don't need to process it
                if update.message.reply_to_message.forum_topic_created is not None:
                    return

                like_words = chat_helper.get_chat_config(update.message.chat.id, "like_words")
                dislike_words = chat_helper.get_chat_config(update.message.chat.id, "dislike_words")

                for category, word_list in {'like_words': like_words, 'dislike_words': dislike_words}.items():
                    if word_list is not None:
                        for word in word_list:
                             #check without case if word in update message
                            if word.lower() in update.message.text.lower():

                                user = db_session.query(db_helper.User).filter(
                                    db_helper.User.id == update.message.reply_to_message.from_user.id).first()
                                if user is None:
                                    user = db_helper.User(id=update.message.reply_to_message.from_user.id,
                                                          first_name=update.message.reply_to_message.from_user.first_name,
                                                          last_name=update.message.reply_to_message.from_user.last_name,
                                                          username=update.message.reply_to_message.from_user.username)
                                    db_session.add(user)
                                    db_session.commit()

                                judge = db_session.query(db_helper.User).filter(
                                    db_helper.User.id == update.message.from_user.id).first()
                                if judge is None:
                                    judge = db_helper.User(id=update.message.from_user.id,
                                                           name=update.message.from_user.first_name)
                                    db_session.add(judge)
                                    db_session.commit()

                                if category == "like_words":
                                    await rating_helper.change_rating(update.message.reply_to_message.from_user.id, update.message.from_user.id, update.message.chat.id, 1, update.message.message_id)
                                elif category == "dislike_words":
                                    await rating_helper.change_rating(update.message.reply_to_message.from_user.id, update.message.from_user.id, update.message.chat.id, -1, update.message.message_id)

                                db_session.close()

                                return
            else:
                pass
    except Exception as error:
        logger.error(f"Error: {traceback.format_exc()}")

async def tg_join_request(update, context):
    try:
        welcome_dm_message = chat_helper.get_chat_config(update.effective_chat.id, "welcome_dm_message")
        auto_approve_join_request = chat_helper.get_chat_config(update.effective_chat.id, "auto_approve_join_request")

        if welcome_dm_message is not None and welcome_dm_message != "":
            try:
                await chat_helper.send_message(bot, update.effective_user.id, welcome_dm_message, disable_web_page_preview=True)
                logger.info(f"Welcome message sent to user {update.effective_user.id} in chat {update.effective_chat.id} ({update.effective_chat.title})")
            except TelegramError as e:
                if "bot can't initiate conversation with a user" in e.message:
                    logger.info(f"Bot can't initiate conversation with user {update.effective_user.id} in chat {update.effective_chat.id} ({update.effective_chat.title})")
                else:
                    logger.error(f"Telegram error: {e.message}. Traceback: {traceback.format_exc()}")
            except Exception as e:
                logger.error(f"General error: {traceback.format_exc()}")

    except TelegramError as e:
        logger.error(f"Telegram error: {e.message}. Traceback: {traceback.format_exc()}")

    except Exception as e:
        logger.error(f"General error: {traceback.format_exc()}")

    finally:
        if auto_approve_join_request:
            try:
                await update.chat_join_request.approve()
            except Exception as e:
                logger.error(f"Error while trying to approve chat join request: {traceback.format_exc()}")



async def tg_new_member(update, context):
    try:
        new_user_id = update.message.api_kwargs['new_chat_participant']['id']

        delete_new_chat_members_message = chat_helper.get_chat_config(update.effective_chat.id, "delete_new_chat_members_message")

        if delete_new_chat_members_message == True:
            await bot.delete_message(update.message.chat.id,update.message.id)

            logger.info(f"Joining message deleted from chat {await chat_helper.get_chat_mention(bot, update.message.chat.id)} for user @{update.message.from_user.username} ({update.message.from_user.id})")

        with db_helper.session_scope() as db_session:
            #check user in global ban list User_Global_Ban
            user_global_ban = db_session.query(db_helper.User_Global_Ban).filter(db_helper.User_Global_Ban.user_id == new_user_id).first()
            if user_global_ban is not None:
                logger.info(f"User {new_user_id} is in global ban list. Kicking from chat {update.message.chat.title} ({update.message.chat.id})")
                await chat_helper.ban_user(bot, update.message.chat.id, new_user_id, reason="User is in global ban list")
                await chat_helper.send_message(bot, update.message.chat.id, f"User {new_user_id} is in global ban list. Kicking from chat {update.message.chat.title} ({update.message.chat.id})", delete_after=120)
                return

        welcome_message = chat_helper.get_chat_config(update.effective_chat.id, "welcome_message")

        if welcome_message is not None and welcome_message != "":
            #TODO:MED: Add user mention (with smart approachthrough function get_user_mention. But we need to put it inside message, so use template vars)
            await chat_helper.send_message(bot, update.effective_chat.id, welcome_message, disable_web_page_preview=True)

    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")

async def tg_update_user_status(update, context):
    try:
        #TODO: we need to rewrite all this to support multiple chats. May be we should add chat_id to user table
        if update.message is not None:
            config_update_user_status = chat_helper.get_chat_config(update.message.chat.id, "update_user_status")
            if config_update_user_status == None:
                logger.info(f"Skip: no config for chat {update.message.chat.id} ({update.message.chat.title})")
                return

            if config_update_user_status == True:
                if len(update.message.new_chat_members) > 0: #user added
                    #TODO:HIGH: We need to rewrite this so we can also add full name
                    db_update_user(update.message.new_chat_members[0].id, update.message.chat.id,  update.message.new_chat_members[0].username, datetime.now(), update.message.new_chat_members[0].first_name, update.message.new_chat_members[0].last_name)
                else:
                    # TODO:HIGH: We need to rewrite this so we can also add full name
                    db_update_user(update.message.from_user.id, update.message.chat.id, update.message.from_user.username, datetime.now(), update.message.from_user.first_name, update.message.from_user.last_name)

                #logger.info(f"User status updated for user {update.message.from_user.id} in chat {update.message.chat.id} ({update.message.chat.title})")

            delete_channel_bot_message = chat_helper.get_chat_config(update.message.chat.id, "delete_channel_bot_message") #delete messages that posted by channels, not users

            if delete_channel_bot_message == True:
                if update.message.from_user.is_bot == True and update.message.from_user.name == "@Channel_Bot":
                    #get all admins for this chat

                    delete_channel_bot_message_allowed_ids = chat_helper.get_chat_config(update.message.chat.id, "delete_channel_bot_message_allowed_ids")

                    if delete_channel_bot_message_allowed_ids is None or update.message.sender_chat.id not in delete_channel_bot_message_allowed_ids:
                        await bot.delete_message(update.message.chat.id, update.message.id)
                        await chat_helper.send_message(bot, update.message.chat.id, update.message.text)
                        logger.info(
                            f"Channel message deleted from chat {update.message.chat.title} ({update.message.chat.id}) for user @{update.message.from_user.username} ({update.message.from_user.id})")

            #TODO: we need to separate this part of the code to separate funciton tg_openai_autorespond
            if update.message.chat.id == -1001588101140: #O1
            # if update.message.chat.id == -1001688952630:  # debug
                #TODO: we need to support multiple chats, settings in db etc

                #Let's here check if we know an answer for a question and send it to user
                openai.api_key = config['OPENAI']['KEY']

                messages = [
                    {"role": "system",
                     "content": f"Answer only yes or no"},
                        {"role": "user", "content": f"Is this a question: \"{update.message.text}\""}
                ]

                response = openai.ChatCompletion.create(
                    model=config['OPENAI']['COMPLETION_MODEL'],
                    messages=messages,
                    temperature=float(config['OPENAI']['TEMPERATURE']),
                    max_tokens=int(config['OPENAI']['MAX_TOKENS']),
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                #check if response.choices[0].message.content contains "yes" without case sensitivity
                if "yes" in response.choices[0].message.content.lower():
                    rows = openai_helper.get_nearest_vectors(update.message.text, 0)

                    logger.info("Question detected " + update.message.text)

                    if len(rows) > 0:
                        logger.info("Vectors detected " + str(rows) + str(rows[0]['similarity']))

                        #TODO this is a debug solution to skip questions with high similarity
                        if rows[0]['similarity'] < float(config['OPENAI']['SIMILARITY_THRESHOLD']):
                            logger.info("Skip, similarity=" + str(rows[0]['similarity']) + f" while threshold={config['OPENAI']['SIMILARITY_THRESHOLD']}")
                            return #skip this message

                        messages = [
                            {"role": "system",
                             "content": f"Answer in one Russian message based on user question and embedding vectors. Do not mention embedding. Be applicable and short."},
                            {"role": "user", "content": f"\"{update.message.text}\""}
                        ]

                        for i in range(len(rows)):
                            messages.append({"role": "system", "content": f"Embedding Title {i}: {rows[i]['title']}\n Embedding Body {i}: {rows[i]['body']}"})

                        response = openai.ChatCompletion.create(
                            model=config['OPENAI']['COMPLETION_MODEL'],
                            messages=messages,
                            temperature=float(config['OPENAI']['TEMPERATURE']),
                            max_tokens=int(config['OPENAI']['MAX_TOKENS']),
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
                        await chat_helper.send_message(bot, update.message.chat.id, response.choices[0].message.content + f" ({rows[0]['similarity']:.2f})", reply_to_message_id=update.message.message_id)

                        #resend update.message to admin
                        await bot.forward_message(config['BOT']['ADMIN_ID'], update.message.chat.id, update.message.message_id)
                        await chat_helper.send_message(bot, config['BOT']['ADMIN_ID'], response.choices[0].message.content + f" ({rows[0]['similarity']:.2f})", disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")


def db_update_user(user_id, chat_id, username, last_message_datetime, first_name=None, last_name=None):
    try:
        #TODO: we need to relocate this function to another location

        with db_helper.session_scope() as db_session:
            if chat_id is None:
                logger.info(f"Debug: no chat_id for user {user_id} ({username}) last_message_datetime")

            # Update or insert user
            user = db_session.query(db_helper.User).filter_by(id=user_id).first()
            if user:
                user.username = username
                user.first_name = first_name
                user.last_name = last_name
            else:
                user = db_helper.User(id=user_id, username=username, first_name=first_name, last_name=last_name)
                db_session.add(user)

            # Update or insert user status
            user_status = db_session.query(db_helper.User_Status).filter_by(user_id=user_id, chat_id=chat_id).first()
            if user_status:
                user_status.last_message_datetime = last_message_datetime
            else:
                user_status = db_helper.User_Status(user_id=user_id, chat_id=chat_id, last_message_datetime=last_message_datetime)
                db_session.add(user_status)

            db_session.commit()
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")


class BotManager:
    def __init__(self):
        self.application = None

    def signal_handler(self, signum, frame):
        logger.error(f"Signal {signum} received, exiting...") #TODO:MED: log as an error for now, we will change it to info later

        # If your library supports stopping the polling:
        if self.application:
            self.application.stop()

        sys.exit(0)

    def run(self):
        try:
            self.application = Application.builder().token(config['BOT']['KEY']).build()

            # delete new member message
            self.application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, tg_new_member), group=1)

            # wiretapping
            self.application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.SUPERGROUP, tg_update_user_status), group=2)  # filters.ChatType.SUPERGROUP to get only chat messages
            self.application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, tg_update_user_status), group=2)

            # checking if user says thank you.
            self.application.add_handler(MessageHandler(filters.TEXT, tg_thankyou), group=3)

            # reporting
            self.application.add_handler(CommandHandler(['report', 'r'], tg_report, filters.ChatType.SUPERGROUP), group=4)
            self.application.add_handler(CommandHandler(['warn', 'w'], tg_warn, filters.ChatType.SUPERGROUP), group=4)

            # Add a handler for chat join requests
            self.application.add_handler(ChatJoinRequestHandler(tg_join_request), group=5)

            self.application.add_handler(CommandHandler(['ban', 'b'], tg_ban, filters.ChatType.SUPERGROUP), group=6)
            self.application.add_handler(CommandHandler(['gban', 'g'], tg_gban), group=6)

            self.application.add_handler(MessageHandler(filters.TEXT | filters.Document.ALL, tg_spam_check), group=7)

            # Set up the graceful shutdown mechanism
            signal.signal(signal.SIGTERM, self.signal_handler)

            # Start the Bot
            self.application.run_polling()
        except Exception as e:
            logger.error(f"Error: {traceback.format_exc()}")

if __name__ == '__main__':
    manager = BotManager()
    manager.run()
