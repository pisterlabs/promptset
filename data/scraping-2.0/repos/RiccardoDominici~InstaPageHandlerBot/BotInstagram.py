import time
import instabot
import openai
import sys
import os
import json
import io
from telegram import Update
from telegram.ext import Application, CommandHandler, filters, ContextTypes
from typing import Final
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from functions import imageHandler



#/start command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = await update.message.reply_text('Wait a moment...')
    global bot
    # Authenticate to Instagram
    bot = instabot.Bot()
    bot.login(username = USERNAME_INSTA , password = PASSWORD_INSTA)
    
    await message.delete()
    await update.message.reply_text('Login done')

#/delete_config command
async def logout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Deleting config file...")
    os.popen("rm -rf config")
    await update.message.reply_text('Done')
    print("Done")

#/watch_stories_from_user command
async def watch_stories_from_user_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #await update.message.reply_text("Which user's followers do you want to get? how many?\nwrite it with nick-number format")
    #text = update.message.text
    #await update.message.reply_text(text)
    await update.message.reply_text("Coming soon") 

#/post_now command
async def post_now_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot
    global IMAGES_FOLDER_PATH
    message = await update.message.reply_text("Generating...")

    image_path, caption = imageHandler.generateImage(os.path.join(DIRPATH, IMAGES_FOLDER_PATH))
    
    await update.message.reply_photo(open(image_path, 'rb'))
    await update.message.reply_text("Caption: "+caption)

    message = await update.message.reply_text("Posting...")

    await imageHandler.post_image(bot, image_path, caption)
    await update.message.reply_text("Image posted")
    await message.delete()
       
#help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('List of comands:\nstart - Start the bot and instagram login\npost - Start the recurrent publication\npost_now - Post on instagram now\nview_error - View error files\nlogout - Delete config files\nhelp - View list of comands\nstop_post - Stop recurrent pubblication')

#/post command
async def post_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    await update.message.reply_text('Starting recurrent posting')
    global loop
    global bot
    global IMAGES_FOLDER_PATH
    loop = True
    
    while loop:
        # Calculation of waiting time until publication time
        now = datetime.now()
        scheduled_time = datetime(now.year, now.month, now.day, POST_HOUR, POST_MINUTE)
        time_to_wait = scheduled_time - now
        seconds_to_wait = time_to_wait.total_seconds()
        
        if seconds_to_wait < 0:
            seconds_to_wait = (86400 + seconds_to_wait)
        
        if loop:
            message = await update.message.reply_text("Waiting for {} hours...".format(int(seconds_to_wait/(60*60))))
            
        time.sleep(seconds_to_wait)
        
        if loop:
            image_path, caption = imageHandler.generateImage(os.path.join(DIRPATH, IMAGES_FOLDER_PATH))
            await update.message.reply_photo(open(image_path, 'rb'))
            await update.message.reply_text("Caption: "+caption)
            await imageHandler.post_image(bot, image_path, caption)
            await update.message.reply_text("Image posted")
            
        # Date increment for next post
        scheduled_time += timedelta(days=1)
        
    await update.message.reply_text('Stop recurrent posting')
    """
    await update.message.reply_text('Coming soon')
    
#/stop_post command
async def stop_post_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global loop
    global message
    loop = False
    await update.message.reply_text('...')
    
# Lets us use the /view_error command
async def view_error_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Coming soon')

# Log errors
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

#------------------------------------------------------------------

DIRPATH = os.path.dirname(__file__)

# Load .env file
dotenv_path = os.path.join(DIRPATH, 'secrets.env')
load_dotenv(dotenv_path)


# Token Telegram
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
BOT_USERNAME = os.getenv("BOT_USERNAME")


# Instagram credentials
USERNAME_INSTA = os.getenv("USERNAME_INSTA")
PASSWORD_INSTA = os.getenv("PASSWORD_INSTA")


# Set the time of publication of the post
POST_HOUR = 10
POST_MINUTE = 49

loop = False

IMAGES_FOLDER_PATH = "image"

#------------------------------------------------------------------


# Run the program
if __name__ == '__main__':
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('logout', logout_command))
    app.add_handler(CommandHandler('view_error', view_error_command))
    app.add_handler(CommandHandler('post', post_command))
    app.add_handler(CommandHandler('post_now', post_now_command))
    app.add_handler(CommandHandler('watch_stories_from_user', watch_stories_from_user_command))
    app.add_handler(CommandHandler('stop_post', stop_post_command))
    app.add_handler(CommandHandler('help', help_command))

    # Log all errors
    app.add_error_handler(error)

    # Run the bot
    app.run_polling(poll_interval=1)

#------------------------------------------------------------------