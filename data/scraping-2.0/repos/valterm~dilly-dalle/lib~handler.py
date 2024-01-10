from telegram import Update, User, Message
from telegram.ext import Updater,CallbackContext
from .dalle import *
from .gpt import *
from .globals import *
import threading
import requests
from io import BytesIO
from openai.error import InvalidRequestError

# Create RequestHandler class
class RequestHandler:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def __strip_input(self, input: str, experssions: list):
        logging.debug('Entering: __strip_input')
        # Strip input
        for e in experssions:
            input = input.replace(e, "")
        return input
    
    def __get_username(self, update: Update):
        # Get user
        user = update.effective_user
        if not user.username:
            return(f"{user.last_name} {user.first_name}")
        else:
            return (user.username)

    def __download_image_into_memory(self, *args, url=None):
        logging.debug('Entering: __download_image_into_memory')
        if not url and args:
            url = args[0]
        headers = {
            "User-Agent": "Chrome/51.0.2704.103",
        }
        response = requests.get(url,headers=headers)

        if response.status_code == 200:
            byte_stream = BytesIO(response.content)
            byte_array = byte_stream.getvalue()
            return byte_array
        else:
            return 1
    
    def __send_text_message(self, update: Update, context: CallbackContext, message: str):
        logging.debug('Entering: __send_text_message')
        # Send message
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=message,
            # parse_mode="MarkdownV2"
            # parse_mode=telegram.constants.ParseMode.MARKDOWN_V2
        )
        logging.debug('Exiting: __send_text_message')
    
    def __send_text_reply(self, update: Update, context: CallbackContext, message: str):
        logging.debug('Entering: __send_reply_text')
        # Send message
        update.message.reply_text(
            text=message,
            reply_to_message_id=update.message.message_id
            # parse_mode="MarkdownV2"
            # parse_mode=telegram.constants.ParseMode.MARKDOWN_V2
        )
        logging.debug('Exiting: __send_reply_text')
    
    def __send_image_message(self, update: Update, context: CallbackContext, image: bytes, caption=None):
        logging.debug('Entering: __send_image_message')
        # Send image
        context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=image,
            caption=caption
        )
        logging.debug('Exiting: __send_image_message')

    def __send_image_reply(self, update: Update, context: CallbackContext, image: bytes, caption=None):
        logging.debug('Entering: __send_image_message')
        # Send image
        context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=image,
            caption=caption,
            reply_to_message_id=update.message.message_id,
        )
        logging.debug('Exiting: __send_image_message')

        
    def __get_image_from_message(self, update: Update):
        logging.debug('Entering: __get_image_from_message')
        # Get image from message
        message = update.effective_message
        photo = message.photo[-1]
        image = photo.get_file()
        return image
    
    def __get_image_from_reply(self, message: Message):
        logging.debug('Entering: __get_image_from_reply')
        # Get image from message
        photo = message.photo[-1]
        image = photo.get_file()
        return image

    def __generate_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: __generate_handler')

        prompt = update.message.text
        prompt = self.__strip_input(prompt, ['/picgen', f"@{BOT_USERNAME}"])

        # Generate image with Dalle class
        dalle = Dalle(self.openai_api_key)

        try:
            image_url = dalle.generate_image(prompt)
        except Exception as e:
            logging.error(f"Error generating image: ")
            self.__send_text_reply(update, context, f"{e}")
            return

        # Download image into memory
        image = self.__download_image_into_memory(image_url)

        # Get username
        username = self.__get_username(update)
        # Create caption
        caption = f"@{username}:{prompt}"

        # Send image
        self.__send_image_reply(update, context, image)
        logging.debug('Exiting: __generate_handler')

    def __variation_handler(self, update: Update, context: CallbackContext, request_type: str):
        logging.debug('Entering: __variation_handler')

        dalle = Dalle(self.openai_api_key)

        # Get image from message
        try:
            if request_type == 'photo':
                image_file = self.__get_image_from_message(update)
            elif request_type == 'reply':
                image_file = self.__get_image_from_reply(update.message.reply_to_message)

            image_jpg = self.__download_image_into_memory(image_file.file_path)
            image_png = dalle.convert_to_png(image_jpg)
        
        except Exception as e:
            logging.error(f"Error getting image: ")
            self.__send_text_reply(update, context, f"There was an error processing the image:\n{e}")
            return

        # Generate image variation with Dalle class
        try:
            image_url = dalle.generate_image_variation(image_png)
        except Exception as e:
            logging.error(f"Error generating image: ")
            self.__send_text_reply(update, context, f"{e}")
            return

        # Download image into memory
        image = self.__download_image_into_memory(image_url)

        # Get username
        username = self.__get_username(update)
        # Create caption
        caption = f"Photo variation for @{username}"
        
        self.__send_image_reply(update, context, image_url)
        logging.debug('Exiting: __variation_handler')
    
    def __description_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: __description_handler')

        prompt = update.message.text
        prompt = self.__strip_input(prompt, ['describe', f"@{BOT_USERNAME}"])

        # Generate description with GPT class
        gpt = GPT(self.openai_api_key)
        description = gpt.generate_description(prompt)

        # Send description
        self.__send_text_reply(update, context, description)

        logging.debug('Exiting: __description_handler')
    
    def __rephrase_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: __rephrase_handler')

        prompt = update.message.text
        prompt = self.__strip_input(prompt, ['rephrase', f"@{BOT_USERNAME}"])

        # Rephrase prompt with GPT class
        gpt = GPT(self.openai_api_key)
        rephrased_prompt = gpt.rephrase_prompt(prompt)

        # Send rephrased prompt
        self.__send_text_reply(update, context, rephrased_prompt)

        logging.debug('Exiting: __rephrase_handler')
    
    def __help_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: __help_handler')
        # Create help message describing the commands
        help_message = f"Hi @{self.__get_username(update)}! I'm {BOT_USERNAME} and I can generate images from text prompts. Here are the commands I understand:\n\n"
        help_message += f"1. /picgen <text prompt> - Generates an image from a text prompt.\n"
        help_message += f"2. /variation <image> - Generates an image variation from an image.\n"
        help_message += f"DISABLED 3. /describe <text prompt> - Generates a description from a text prompt.\n"
        help_message += f"4. /rephrase <text prompt> - Rephrases a text prompt.\n"
        help_message += f"5. /help - Displays this help message.\n\n"
        help_message += f"Please note that I'm still in beta and I may not work as expected. If you encounter any issues, please report them to github @ {GITHUB_REPO}.\n"
        
        # Send help message
        self.__send_text_message(update, context, help_message)

        logging.debug('Exiting: __help_handler')
    
    def __start_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: __start_handler')
        # Create start message
        start_message = f"Hi @{self.__get_username(update)}! I'm {BOT_USERNAME} and I can generate images from text prompts. Send /help to see the commands I understand.\n\n"
        start_message += f"Please note that I'm still in beta and I may not work as expected. If you encounter any issues, please report them to github @ {GITHUB_REPO}.\n"

        # Send start message
        self.__send_text_message(update, context, start_message)

    def __unknown_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: __unknown_handler')
        # Create unknown message
        unknown_message = f"Sorry {self.__get_username(update)}, I didn't understand that command. Send /help to see the commands I understand.\n\n"

        # Send unknown message
        self.__send_text_message(update, context, unknown_message)

    def __prototype_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: __prototype_handler')
        # Create prototype message
        prototype_message = f"Sorry, but that command is still in unavailable. Please send /help to see the list of available commands."
        # Send prototype message
        self.__send_text_message(update, context, prototype_message)

    # Create command handlers
    def start_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: start_command_handler')
        threading.Thread(target=self.__start_handler, args=(update, context)).start()
        logging.debug('Exiting: start_command_handler')

    def help_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: help_command_handler')
        threading.Thread(target=self.__help_handler, args=(update, context)).start()
        logging.debug('Exiting: help_command_handler')

    def picgen_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: picgen_command_handler')
        threading.Thread(target=self.__generate_handler, args=(update, context)).start()
        logging.debug('Exiting: picgen_command_handler')
    
    def photo_filter_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: photo_handler')
        if update.message.caption and '/variation' in update.message.caption:
            threading.Thread(target=self.__variation_handler, args=(update, context, 'photo')).start()
        logging.debug('Exiting: photo_handler')

    def variation_reply_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: variation_reply_command_handler')
        if update.message.reply_to_message:
            if update.message.reply_to_message.photo:
                threading.Thread(target=self.__variation_handler, args=(update, context, 'reply')).start()
            else:
                self.__send_text_reply(update, context, "Please either reply to an image with /variation or send a photo with /variation in the caption.")
        else:
            self.__send_text_reply(update, context, "Please either reply to an image with /variation or send a photo with /variation in the caption.")
        logging.debug('Exiting: variation_reply_command_handler')

    def description_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: description_command_handler')
        threading.Thread(target=self.__description_handler, args=(update, context)).start()
        logging.debug('Exiting: description_command_handler')

    def rephrase_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: rephrase_command_handler')
        threading.Thread(target=self.__rephrase_handler, args=(update, context)).start()
        logging.debug('Exiting: rephrase_command_handler')
    
    def unknown_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: unknown_command_handler')
        threading.Thread(target=self.__unknown_handler, args=(update, context)).start()
        logging.debug('Exiting: unknown_command_handler')

    def prototype_command_handler(self, update: Update, context: CallbackContext):
        logging.debug('Entering: prototype_command_handler')
        threading.Thread(target=self.__prototype_handler, args=(update, context)).start()
        logging.debug('Exiting: prototype_command_handler')