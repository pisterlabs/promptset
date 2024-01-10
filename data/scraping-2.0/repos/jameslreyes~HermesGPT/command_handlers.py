import logging
import asyncio
import requests
from io import BytesIO
from urllib.parse import urlparse, parse_qs
import openai
from deepgram import Deepgram
from elevenlabs import set_api_key
from telegram.ext import ContextTypes, CallbackContext
from scripts.logging import *
from classes.chat_gpt import *
from classes.handlers.voice_handler import *
from classes.handlers.image_handler import *
from classes.handlers.search_handler import *
from classes.telegram_bot import *
from classes.handlers.youtube_handler import *
from classes.handlers.feedback_handler import *
from classes.dropbox import *
from classes.handlers.feedback_handler import *
from classes.handlers.chat_handler import ChatHandler
from auth import *
from config import Config

# Create instances of your classes
chat_gpt = ChatGPT()
voice_handler = VoiceHandler()
image_handler = ImageHandler()
search_handler = SearchHandler()
telegram_bot = TelegramBot()
dropbox_client = DropboxClient()
feedback_handler = FeedbackHandler()

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


openai.api_key = Config.OPENAI_API_KEY
set_api_key(Config.ELEVEN_API_KEY)

# Fetch the list of voices
response = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": Config.ELEVEN_API_KEY})
voice_data = response.json()

# Initialize Deepgram client on startup
deepgram = Deepgram(Config.DEEPGRAM_API_KEY)

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_id = update.effective_user.id

        if user_id not in Config.AUTHORIZED_USER_IDS:
            await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
            return

        await help_command(update, context)
        await send_chat_action_async(update, 'typing')
        await asyncio.sleep(1)
        await update.message.reply_text("Disclaimer: This bot is not affiliated with Telegram.")
    except Exception as e:
        print(f"An error occurred in the start function: {e}")
        # Handle the exception here or log it for further investigation

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_name = update.effective_user.full_name
        user_id = update.effective_user.id
        chat_type = update.effective_chat.type

        if user_id not in Config.AUTHORIZED_USER_IDS:
            await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
            return

        print(f"{user_name} (ID: {user_id}): /help")

        help_text_private = (
            "Hi\\! I am HermesGPT, your personal assistant\\. Here are a couple of ways to interact with me:\n\n"
            "Just send a message to start a conversation with me\\.\n\n"
            "*Voice Settings*\n"
            "/voices \\- Shows a list of available voices\n"
            "/select \\- Selects a voice, for example: /select Josh\n"
            "/v \\- Start a message with '/v' to generate a spoken message\n\n"
            "*Response Settings*\n"
            "/stable \\- Enable stable mode \\(Default\\)\n"
            "/unstable \\- Enable unstable mode\\. Warning: Responses will be almost completely incoherent\\.\n\n"
            "*Other Commands*\n"
            "/search \\- Search the internet for something, for example: /search recent AI news\n"
            "/summarize \\- Summarize a YouTube video, for example: /summarize https://www\\.youtube\\.com/watch?v\\=dQw4w9WgXcQ\n"
            "/image \\- Generates an image based on a prompt, for example: /image a black cat sitting on a throne\n"
            "/clear \\- Clears individual message history\n"
            "/help \\- Shows a list of commands\n\n"
        )
        
        help_text_group = (
            "Hi\\! I'm HermesGPT, your personal assistant\\. Here are a couple of ways to interact with me:\n\n"
            "/ \\- Start a message with '/' to talk to me\n\n"
            "*Voice Settings*\n"
            "/voices \\- Shows a list of available voices\n"
            "/select \\- Selects a voice, for example: /select Josh\n"
            "/v \\- Start a message with '/v' to generate a spoken message\n\n"
            "*Response Settings*\n"
            "/stable \\- Enable stable mode \\(Default\\)\n"
            "/unstable \\- Enable unstable mode\\. Warning: Responses will be almost completely incoherent\\.\n\n"
            "*Other Commands*\n"
            "/search \\- Search the internet for something, for example: /search recent AI news\n"
            "/summarize \\- Summarize a YouTube video, for example: /summarize https://www\\.youtube\\.com/watch?v\\=dQw4w9WgXcQ\n"
            "/image \\- Generates an image based on a prompt, for example: /image a black cat sitting on a throne\n"
            "/clear \\- Clears individual message history\n"
            "/help \\- Shows a list of commands\n\n"
        )

        if chat_type == "private":
            await send_chat_action_async(update, 'typing')
            await asyncio.sleep(0.5)
            await update.message.reply_text(help_text_private, parse_mode='MarkdownV2')
        else:
            await send_chat_action_async(update, 'typing')
            await asyncio.sleep(0.5)
            await update.message.reply_text(help_text_group, parse_mode='MarkdownV2')

        print("ChatGPT: Help message sent.")
    except Exception as e:
        print(f"An error occurred in the help_command function: {e}")
        # Handle the exception here or log it for further investigation

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.full_name
    user_id = update.effective_user.id

    if user_id not in Config.AUTHORIZED_USER_IDS:
        await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code, like so: /passcode 1234.")
        return

    print(f"{user_name} (ID: {user_id}): /clear")

    try:
        if 'messages' not in context.user_data:
            context.user_data['messages'] = {}
        context.user_data['messages'][user_id] = []

        await send_chat_action_async(update, 'typing')
        await asyncio.sleep(0.5)
        await update.message.reply_text("Conversation history cleared.")
        print("HermesGPT: Conversation history cleared.")
    except Exception as e:
        print(f"Error occurred while clearing conversation history: {str(e)}")
        await update.message.reply_text("An error occurred while clearing conversation history. Please try again later.")

async def speak_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text
    user_name = update.effective_user.full_name
    user_id = update.effective_user.id

    if user_id not in Config.AUTHORIZED_USER_IDS:
        await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
        return

    # Remove the '/v' prefix
    user_input = user_input[2:].strip()

    print(f"{user_name} (ID: {user_id}): {user_input}")

    try:
        # Add the following lines to get the ChatGPT response first
        await send_chat_action_async(update, 'typing')
        response_data = await chat_gpt.get_chat_gpt_response(user_input, user_id, context)
        chat_gpt_response = response_data['choices'][0]['message']['content']
        
        if voice_handler.modes.get(user_id, "stable") == "unstable":
            chat_gpt_response = await ChatHandler.unstable_text_transform(chat_gpt_response)

        print(f"ChatGPT: {chat_gpt_response}")

        # Send the ChatGPT response as a text message
        await send_chat_action_async(update, 'typing')
        await asyncio.sleep(0.5)
        await update.message.reply_text(chat_gpt_response)

        # # Then generate the voice message based on the ChatGPT response
        # await send_chat_action_async(update, 'record_audio')

        # Retrieve the user's mode
        mode = voice_handler.modes.get(user_id)

        # Remove the hardcoded voice_id, and use the user's selected voice (with a default fallback)
        voice_id = voice_handler.selected_voices.get(user_id, "7kRUX4UzUC1zcoeqNF4s")
        print(f"[Debug] speak_command: Voice ID selected for user {user_name} ({user_id}) is {voice_id}")
        voice_message, error_message = await voice_handler.generate_voice_message(chat_gpt_response, voice_id, Config.ELEVEN_API_KEY, mode)
        
        if voice_message:
            with BytesIO(voice_message) as voice_stream:
                await send_chat_action_async(update, 'record_audio')
                await update.message.reply_voice(voice=voice_stream)
        else:
            await send_chat_action_async(update, 'typing')
            await asyncio.sleep(0.5)
            await update.message.reply_text(error_message)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        await update.message.reply_text("An error occurred. Please try again later.")

async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_name = update.effective_user.full_name

    try:
        if user_id not in Config.AUTHORIZED_USER_IDS:
            await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
            return

        # Use the stored voice data
        voice_list = voice_data["voices"]

        voices_text = "Available voices:\n\n"
        for voice in voice_list:
            voices_text += f"{voice['name']}\n"
        await send_chat_action_async(update, 'typing')
        await asyncio.sleep(0.5)
        await update.message.reply_text(voices_text)
        await update.message.reply_text("Type a name after the /select command to choose a voice. For example: /select Adam")

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        await update.message.reply_text("Oops! Something went wrong. Please try again later.")

async def select_voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_name = update.effective_user.full_name

    if user_id not in Config.AUTHORIZED_USER_IDS:
        await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
        return

    try:
        voice_name = " ".join(update.message.text.split()[1:]).capitalize()
        print(f"{user_name} (ID: {user_id}): /select {voice_name}")

        # Use the stored voice data
        voice_list = voice_data["voices"]

        # Check if provided voice_name is valid
        voice_id = next((voice["voice_id"] for voice in voice_list if voice["name"].lower() == voice_name.lower()), None)
        if voice_id is not None:
            voice_handler.selected_voices[user_id] = voice_id
            print(f"Updated selected_voices: {voice_handler.selected_voices}, {voice_name}")
            await send_chat_action_async(update, 'typing')
            await asyncio.sleep(0.5)
            await update.message.reply_text(f"Voice successfully set to {voice_name}.")
        else:
            await send_chat_action_async(update, 'typing')
            await asyncio.sleep(0.5)
            await update.message.reply_text("Sorry, the provided voice name is not valid. Use the /voices command to view the available voice options.")

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        await update.message.reply_text("Oops! Something went wrong. Please try again later.")
        
async def stable_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_name = update.effective_user.full_name

    if user_id not in Config.AUTHORIZED_USER_IDS:
        await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
        return

    try:
        print(f"{user_name} (ID: {user_id}): /stable")

        voice_handler.modes[user_id] = "stable"
        await send_chat_action_async(update, 'typing')
        await asyncio.sleep(0.5)
        await update.message.reply_text("Mode set to stable.")
    except Exception as e:
        error_message = f"An error occurred in the `stable_command` function of `command_handlers.py`: {str(e)}"
        print(error_message)
        await update.message.reply_text("Oops! Something went wrong. Please try again later.")

async def unstable_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_name = update.effective_user.full_name

    if user_id not in Config.AUTHORIZED_USER_IDS:
        await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
        return

    try:
        voice_handler.modes[user_id] = "unstable"
        print(f"{user_name} (ID: {user_id}): /unstable")
        
        await send_chat_action_async(update, 'typing')
        await asyncio.sleep(0.5)
        await update.message.reply_text("Mode set to unstable.")
    except Exception as e:
        error_message = f"An error occurred in the `unstable_command` function of `command_handlers.py`: {str(e)}"
        print(error_message)
        await update.message.reply_text("Oops! Something went wrong. Please try again later.")

async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    callback_data = query.data

    image_number = int(callback_data.split('_')[1])

    with open(f'./out/v1_txt2img_{image_number}.png', 'rb') as f:
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=f
        )

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    image_handler = ImageHandler()
    await image_handler.generate_image(update, context)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_input = update.message.text
        user_id = update.effective_user.id

        if user_id not in Config.AUTHORIZED_USER_IDS:
            await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
            return
        
        query = user_input[len("/search"):].strip()

        # Use the handle_search_command from SearchHandler
        await SearchHandler().handle_search_command(update, context, query)

    except Exception as e:
        error_message = f"ERROR: command_handles.py/`search_command()`: {str(e)}"
        await update.message.reply_text(error_message)
        print(error_message)

async def passcode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_id = update.effective_user.id
        user_name = update.effective_user.full_name

        provided_passcode = " ".join(update.message.text.split()[1:])

        if provided_passcode == "4309":
            if user_id not in Config.AUTHORIZED_USER_IDS:
                Config.AUTHORIZED_USER_IDS.append(user_id)

                # Save the updated list of authorized user IDs to Dropbox
                await save_allowed_user_ids_to_dropbox(Config.AUTHORIZED_USER_IDS, Config.dbx)

                await update.message.reply_text(f"Access granted! Welcome, {user_name}.")
                print(f"{user_name} (ID: {user_id}) has been granted access.")
            else:
                await update.message.reply_text("You already have access.")
        else:
            await update.message.reply_text("Incorrect passcode. Please try again.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await update.message.reply_text(error_message)
        print(error_message)

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_id = update.effective_user.id
        user_name = update.effective_user.full_name
        youtube_url = update.message.text[len("/summarize"):].strip()
        youtube_url = YouTubeHandler.convert_to_desktop_link(youtube_url)

        if user_id not in Config.AUTHORIZED_USER_IDS:
            await update.message.reply_text("You do not have permission to use this bot. If you have a passcode, simply type /passcode followed by your code.")
            return

        print(f"{user_name} (ID: {user_id}): /summarize {youtube_url}")

        # Parse YouTube URL
        parsed_url = urlparse(youtube_url)
        if parsed_url.netloc == "youtu.be":
            video_id = parsed_url.path[1:]
        else:
            video_id = parse_qs(parsed_url.query).get("v")

        if not video_id:
            await update.message.reply_text("Invalid YouTube URL.")
            return

        video_id = video_id[0]

        # Get video captions
        caption_text = await YouTubeHandler.get_caption_text(video_id)

        if not caption_text:
            await update.message.reply_text("Unable to retrieve video captions.")
            return

        # Initialize messages in user_data if not present
        if 'messages' not in context.user_data:
            context.user_data['messages'] = {}
        if user_id not in context.user_data['messages']:
            context.user_data['messages'][user_id] = []

        # Call GPT and generate summary
        generating_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Generating video summary...")
        generating_message_id = generating_message.message_id
        summary = await chat_gpt.call_gpt(caption_text, user_id, context.user_data)

        if summary:
            await send_chat_action_async(update, 'typing')
            await asyncio.sleep(1)
            await send_chat_action_async(update, 'cancel')
            await context.bot.edit_message_text(chat_id=update.effective_chat.id,
                                                message_id=generating_message_id,
                                                text="Here's the video summary:\n\n" + summary)
            # Add the generated summary to the conversation history
            if 'messages' not in context.user_data:
                context.user_data['messages'] = {}
            if user_id not in context.user_data['messages']:
                context.user_data['messages'][user_id] = []
            context.user_data['messages'][user_id].append({"role": "assistant", "content": summary})

        else:
            await send_chat_action_async(update, 'typing')
            await asyncio.sleep(1)
            await send_chat_action_async(update, 'cancel')
            await context.bot.edit_message_text(chat_id=update.effective_chat.id,
                                                message_id=generating_message_id,
                                                text="Sorry, I couldn't generate a summary for the video. Please try a different video or make sure the link is valid.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await update.message.reply_text(error_message)
        print(error_message)

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        feedback_text = update.message.text[len("/feedback"):].strip()
        user_name = update.effective_user.full_name

        await feedback_handler.store_feedback(user_name, feedback_text)

        await update.message.reply_text("Thank you for your feedback!")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Print error message for clear understanding in the terminal
        await update.message.reply_text("Oops! Something went wrong. Please try again later.")  # Send error message to the user