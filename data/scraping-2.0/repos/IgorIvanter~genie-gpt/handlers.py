import openai
import telegram
import config

from UserDataProviderV2 import UserDataProvider
from moviepy.editor import AudioFileClip

from messages import (
    DEFAULT_SYSTEM_MESSAGE,
    TEXT_RECEIVED_MESSAGE,
    VOICE_RECEIVED_MESSAGE,
    FREE_TRIAL_EXPIRED_MESSAGE
)

from config import (
    VOICE_MESSAGE_MP3,
    VOICE_MESSAGE_OGG,
    USER_DATA_FILE_PATH,
    MAX_FREE_REQUESTS
)


logging = config.logging

OPENAI_REQUEST_TIMEOUT = 60  # openai request timeout in seconds


def handle_message_text(update, context):
    logging.debug("Entering handle_message_text")

    chat_id = update.message.chat_id
    users = UserDataProvider(USER_DATA_FILE_PATH)
    user_plan_data = users.get_user_data(user_id=chat_id)

    if len(user_plan_data) == 0:
        users.update_user_data(
            user_id=chat_id,
            has_paid_plan=False,
            num_requests=0,
            username=update.message.chat.username
        )
        user_plan_data = users.get_user_data(user_id=chat_id)
    # Check if the user has free requests left
    if user_plan_data["has_paid_plan"] == False and user_plan_data["num_requests"] >= MAX_FREE_REQUESTS:
        update.message.reply_text(
            text=FREE_TRIAL_EXPIRED_MESSAGE
        )
        return
    else:
        # Increment num requests
        users.update_user_data(
            user_id=chat_id,
            has_paid_plan=user_plan_data["has_paid_plan"],
            num_requests=user_plan_data["num_requests"] + 1,
            username=update.message.chat.username
        )

    # Store last update and last message for the case of error
    context.user_data['last_request'] = update.message.text
    context.user_data['last_update'] = update

    # Check if the chat history hasn't been recorded yet or if recorded improperly
    if context.user_data.get('messages') == None:
        context.user_data["messages"] = [{
            "role": "system",
            "content": DEFAULT_SYSTEM_MESSAGE
        }]
    elif context.user_data["messages"][0]["role"] != "system":
        raise NameError(
            "First message role is not system, but '{}'".format(
                context.user_data["messages"][0]["role"]
            )
        )

    # Add the latest user message to history
    context.user_data["messages"].append({
        "role": "user",
        "content": update.message.text
    })
    # Send a 'text received' message
    chat_message = update.message.reply_text(
        text=TEXT_RECEIVED_MESSAGE
    )
    # Send typing action
    context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=telegram.ChatAction.TYPING
    )
    # Get a response from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=context.user_data["messages"],
        request_timeout=OPENAI_REQUEST_TIMEOUT
    )
    # Extract text from response
    response_text = response["choices"][0]["message"]["content"]
    context.bot.delete_message(
        chat_id=update.message.chat_id,
        message_id=chat_message.message_id
    )
    # Send the response text into the chat
    update.message.reply_text(
        text=f"*[Bot]:* {response_text}",
        parse_mode=telegram.ParseMode.MARKDOWN
    )
    # Add response text to chat history
    context.user_data["messages"].append(
        {"role": "assistant", "content": response_text})
    # Shorten chat history to no more then 10 last messages (avoid overflow)
    context.user_data["messages"] = context.user_data["messages"][-10:]
    context.user_data["messages"][0] = {
        "role": "system",
        "content": DEFAULT_SYSTEM_MESSAGE
    }
    logging.debug("Exiting handle_message_text")


def handle_message_voice(update, context):
    logging.debug("Entering handle_message_voice")
    # Send a 'voice received' message
    voice_received_message = update.message.reply_text(
        VOICE_RECEIVED_MESSAGE
    )
    # Download the voice note and convert to mp3
    voice_file_ogg = context.bot.getFile(update.message.voice.file_id)
    voice_file_ogg.download(VOICE_MESSAGE_OGG)
    temp_audio_clip = AudioFileClip(VOICE_MESSAGE_OGG)
    temp_audio_clip.write_audiofile(VOICE_MESSAGE_MP3)
    voice_file_mp3 = open(VOICE_MESSAGE_MP3, "rb")
    # Get the transcription from Whisper API
    transcript = openai.Audio.transcribe("whisper-1", voice_file_mp3).text
    update.message.text = transcript
    # Delete the 'voice received' message
    context.bot.delete_message(
        chat_id=update.message.chat_id,
        message_id=voice_received_message.message_id
    )
    # Send transcript to user
    update.message.reply_text(
        text=f"*[You]:* _{transcript}_",
        parse_mode=telegram.ParseMode.MARKDOWN
    )
    handle_message_text(update, context)
    logging.debug("Exiting handle_message_voice")
