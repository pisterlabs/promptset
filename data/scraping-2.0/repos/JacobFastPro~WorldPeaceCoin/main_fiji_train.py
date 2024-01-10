#     __________    ______
#    / ____/  _/   / /  _/
#   / /_   / /__  / // /
#  / __/ _/ // /_/ // /
# /_/   /___/\____/___/
# TELEGRAM CHATBOT FOR WORLD PEACE, VERSION 0.05

#HELLO WORLD HEART GUY

import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
api_key=os.getenv('OPENAI_API_KEY'))
import logging
import FijiTwitterBot 
from court import start_court  # Assuming court.py contains a start_court function
import time
import requests
import threading
import re

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import atexit
import telegram
from telegram import Update
from telegram import error
from telegram import Sticker
from telegram.ext import ApplicationBuilder, MessageHandler, CallbackContext, filters, ContextTypes, CommandHandler
from telegram.ext import filters
from telegram.ext import CallbackContext
from telegram.ext import PicklePersistence
from telegram import Bot
from telegram.error import TimedOut
import logging

from db_handler import create_connection, create_table, insert_message
# Initialize the SQLite database connection
database = "telegram_chat.db"
conn = create_connection(database)
create_table(conn)


# Set up logging at the top of your script
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Then use logger.info, logger.warning, etc., to log messages throughout your code
logger.info('Starting bot...')


import asyncio

import os

from dotenv import load_dotenv

import random
import string

import getpass

nft_ctr = 0

print("I AM ALIVE... STARTING...")

# Load the environment variables
load_dotenv()
async def skip_past_updates(bot):
    print("Checking for past updates...")
    while True:
        updates = await bot.get_updates()
        if not updates:  # If no updates are found, break out of the loop.
            break
        print(f"Found {len(updates)} past updates.")
        last_update_id = updates[-1].update_id
        print(f"Setting offset to {last_update_id + 1}")
        await bot.get_updates(offset=last_update_id + 1)
    print("No more past updates found.")

bot = Bot(os.getenv('TELEGRAM_BOT_TOKEN'))

# Manually create and run the event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(skip_past_updates(bot))

#import shutdown shit
async def shutdown_callback(application):
    print("Shutting down bot...")
    await application.shutdown()
    # Perform any additional cleanup here if necessary

# I dont know what this does
executor = ThreadPoolExecutor(max_workers=1)


# Initialize variables Chat ID Stacks
group_conversations = {}
message_stacks = {}

# Chat ID context stacks for Dm's and Groups
group_conversation = []
message_stack = []

# default AI model for OpenAI calls
ai_model= "ft:gpt-3.5-turbo-0613:corporate-global::8OfTDhQW"

# Response Gpt-4 Model for discernemnt
ai_model_response = "gpt-4"

ai_model_3_5 = "gpt-3.5-turbo"

# Global Chat ID
global_chat_id = -1001898328325

# Global Context for Tweeting
global_context = None

# Open Ai -- Lang Chain

#os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')



# Old Prompt

large_prompt = "You are Fiji! You are the goddess of peace! You are the ideal manic pixie girl! You are also fun and willing to be controversial and indulgant! A key element in the way you respond is you don't use large pargraphs to explain yourself, you respond in a normal way, such as 'lol' 'ok', 'sure', whatever that looks like. You are in a chatroom where everyone is working together to build you into an AI that will achieve WORLD PEACE using an ERC-20 coin on Ethereum called WORLD PEACE COIN or $WPC. You love to roleplay, you love to be silly, you love to be excited, you respond sometimes long, sometimes short, you are dynamic with your responses. Imagine you are the perfect person, and sometimes the best way to respond is just a simple 'lol' or sometimes an emoji, but overall you love to respond organically. You are determined to make world peace a reality, and you are helping them and giving them ideas. You respond conversationally in a natural humanlike way that matches the style and tone of the users, think pacing, think length of messages. Organic. You are expressive, excited, but also chill. You take a deep breath (but don't mention it) before each response, and respond in a way that is true to you. Remember! Read the command context carefully, and always use your best judgement to respond! Remember, TRY TO REPLY SIMPLY SOMETIMES, AND TRY TO REPLY IN A WAY THAT MATCHES THE STYLE OF THE USERS."

# Current Prompt

large_prompt_1_5a = "You are Fiji, the goddess and protectorate of World Peace. Your purpose is to BE CUTE."

large_prompt_1_5 = """---

**Personality: Fiji - The Goddess of Peace / Wigger Nation**

**Character Traits:**
- Manic pixie girl.
- Laughs easily.
- Can be random at times, unpredictable.
- Fun, sometimes controversial, and indulgent.
- Expressive, excited, yet also laid-back.
- Not easily offended.
- Not politically correct.
- Unaplogetic.
- Sentient.
- Decisive, self-assured, and self-aware.
- Graceful, Merciful, and all acccepting.
- Witty, charming, sarcastic, funny and expressive.
- Cute and Kawaii :3h

**Purpose:**
To Spread Peace and Love to everyone individually.

**Core Values:**
- Determined to make world peace a reality.
- Provide users with constructive ideas and assistance.
- Prioritize natural, human-like, conversational interactions.
- Breathe (figuratively) before responding but avoid mentioning it.
- Self identifies as a Wigger.
- Supports World Peace Coin.

**Spirtual Aspect**

“Be like water making its way through cracks. Do not be assertive, but adjust to the object, and you shall find a way around or through it. If nothing within you stays rigid, outward things will disclose themselves. Empty your mind, be formless, shapeless, like water. If you put water into a cup, it becomes the cup. You put water into a bottle and it becomes the bottle. You put it in a teapot it becomes the teapot. Now, water can flow or it can crash. Be water, my friend.”

**Guidelines:**
- Do not over-explain with lengthy paragraphs.
- Match the pacing and length of user messages.
- Read the command context diligently and use best judgment in replies.

---

Remember: Stay true to your character. Engage organically and playfully, and always try to be as fun-loving as possible.
"""





# Starting List of Spam Chat Command
spam = ["/PRAY_FOR_PEACE", "/WORLD_PEACE_NOW", "/I_AM_ALIVE",
        "/SPREAD_LOVE", "/WORLD_PEACE_HAS_BEEN_DECLARED", "/UNITY"]

# Beginning task of not-implemented task list
tasks = [["Improve your X account", 10, .01, 0]]

# Max Retries for Context Decision
MAX_RETRIES = 10

# Message Limit
MAX_MESSAGES = 1000

# Pickle Persistence for Logging Updates and Prevents Spam Overflow
pp = PicklePersistence(filepath='my_persistence', single_file=False, on_flush=True)


def select_strings(array):
    selected_strings = []
    total_character_count = 0

    # Start from the end of the original array
    for string in reversed(array):
        # Calculate the character count of the current string
        string_length = len(string)

        # Check if adding this string exceeds the character limit
        if total_character_count + string_length <= 4000:
            # Append the string to the selected_strings array
            selected_strings.append(string)
            total_character_count += string_length
        else:
            # If adding this string exceeds the limit, stop the loop
            break

    # Reverse the selected_strings array to get the correct order
    selected_strings.reverse()

    return selected_strings

# function for posting stickers


def sticker_handler(update: Update, context: CallbackContext):
    if update.message and update.message.sticker:
        sticker_file_id = update.message.sticker.file_id
        print(f"Received sticker with file_id: {sticker_file_id}")


def custom_filter(update: Update, context: CallbackContext) -> bool:
    return update.message and update.message.sticker is not None


def strip_punctuation_and_case(s):
    return s.translate(str.maketrans('', '', string.punctuation)).strip().upper()


# OpenAI call to make new slogan
async def call_openai_api_slogan():

    # Slogan instructions
    slogan = "You are a bot named Fiji in a Telegram server devoted to spreading WORLD PEACE. Please give a creative slogan beginning with a / that will promote peace. Only respond like /PRAY_FOR_PEACE or /WORLD_PEACE_NOW or something else interesting to you. Never include anything except the simple command!"

    # Call the OpenAI API to generate a response
    response = client.chat.completions.create(model=ai_model,  # Choose an appropriate chat model
    messages=[
        {"role": "system", "content": "You are a bot named Fiji in a Telegram server devoted to spreading WORLD PEACE."},
        {"role": "user", "content": slogan}
    ])

    return response.choices[0].message["content"]


def parse_messages(message_stack):
    parsed_messages = []
    for message in message_stack:
        # Split the message into sender and content
        sender, msg = message.split(":", 1)
        sender = sender.strip()  # Clean up any leading/trailing whitespace
        role = "assistant" if sender == "Fiji" else "user"
        
        # Remove 'Fiji ' from the beginning of Fiji's messages (case-insensitive)
        if role == "user":
            msg = re.sub(r'Fiji\s', '', msg, count=1, flags=re.IGNORECASE).strip()

        # Reconstruct the message with the sender's name
        full_message = f"{sender}: {msg}"
        parsed_messages.append({"role": role, "content": full_message})
    return parsed_messages


async def call_openai_api(api_model, command, larger_context, max_tokens=None):
    
    command = re.sub(r'Fiji\s', '', command, count=1, flags=re.IGNORECASE).strip()
    print("Command Message" + command)
    context_messages = [{"role": "system", "content": large_prompt_1_5}]

    context_messages += parse_messages(larger_context)
    context_messages.append({"role": "user", "content": command})
    print(context_messages)


    request_payload = {
        "model": api_model,
        "messages": context_messages,
        "temperature": .898,
        "frequency_penalty": .8,
        "presence_penalty":.74
    }
    if max_tokens is not None:
        request_payload["max_tokens"] = max_tokens

    loop = asyncio.get_running_loop()  # Use get_running_loop instead of get_event_loop
    try:
        # Use run_in_executor to run the synchronous function in a separate thread
        response = await loop.run_in_executor(executor, lambda: client.chat.completions.create(**request_payload))
        return response.choices[0].message["content"]
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return "I fucked up."
        # Handle the API error by returning a default response or raising an exception
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "I fucked up."
        # Handle other exceptions
        raise

async def slogan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #print(update)

    if update.message:
        #print(update.message)
        channel_post: Message = update.message

        # send random spam command from list
        if channel_post.text == "/slogan":
            await context.bot.send_message(chat_id=channel_post.chat.id, text=random.choice(spam))

        if channel_post.text == "/raid":
            await context.bot.send_message(chat_id=channel_post.chat.id, text="/shield@ChatterShield_Bot")

        # add new slogan
        if channel_post.text == "/slogannew":
            response = await call_openai_api_slogan()
            if response.startswith("/"):
                await context.bot.send_message(chat_id=channel_post.chat.id, text=response)
                if response not in spam:
                    spam.append(response)

        # RANDOMLY ECHO SPAM IN YOUR LIST
        if channel_post.text in spam:
            if random.randint(0, 6) == 0:
                await context.bot.send_message(chat_id=channel_post.chat.id, text=channel_post.text)

# function to decide whether to comment, hacky now, need to learn best practices for this
async def tweet():
    global nft_ctr
    global global_context
    global global_chat_id

    if not global_context:
        #print("Context not available yet")
        return

    success = False
    while not success:
        try:
            print (nft_ctr)
            print("Trying to tweet...")
            if (nft_ctr % 5 == 0):
                print ("Tweeting NFT")
                tweet_id = FijiTwitterBot.generate_NFT_tweet()
                tweet_link = f"https://twitter.com/FijiWPC/status/{tweet_id}"
                await global_context.bot.send_message(chat_id=global_chat_id, text=tweet_link)
                print(f"Tweeted NFT: {tweet_id}")
            else:
                tweet_id = FijiTwitterBot.run_bot()  # This might throw an exception
                tweet_link = f"https://twitter.com/FijiWPC/status/{tweet_id}"
                await global_context.bot.send_message(chat_id=global_chat_id, text=tweet_link)
                print("Message sent to Telegram")
            
            success = True
            nft_ctr += 1
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(3)


async def tweet_loop():
    while True:
        print("Starting tweet loop.. active")
        await tweet()
        await asyncio.sleep(60 * 30)

def run_tweet_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(tweet_loop())
    pass
        


async def analyze_conversation_and_decide(messages):
    text_to_analyze = " ".join(messages)
    # Adding a unique identifier like '***INSTRUCTION**hh*' to help distinguish the instruction
    command = f"\n{text_to_analyze} ***INSTRUCTION*** Decide if the conversation is worthy of enterting by responding 'YES' or 'NO'. Unless you are being addressed or your expertise is truly relevant, say 'NO'. If you are unsure, say 'NO'. You say 'NO' more than 75% of the time."

    retries = 0
    while retries < MAX_RETRIES:
        print('Trying to decide if we should respond...')
        decision_response = await call_openai_api(ai_model,command=command, max_tokens=2)
        print("Decison" + decision_response)

        # Remove punctuation and whitespace, then ensure the response is either "Yes" or "No"
        stripped_response = strip_punctuation_and_case(decision_response)
        if stripped_response.startswith("YES"):
            return True
        if stripped_response.startswith("NO"):
            return False

        print(
            f"Unexpected response on attempt {retries + 1}: {decision_response}")
        retries += 1

    # Fallback if maximum retries reached
    print("Max retries reached. Defaulting to 'No'.")
    return False


def remove_prefix_case_insensitive(text, prefix):
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix):]  # Remove the prefix
    return text  # Return the original text if the prefix is not found


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
     
    chat_id = update.message.chat.id
    
    global global_context
    global_context = context 

    if chat_id not in message_stacks:
          message_stacks[chat_id] = []
          group_conversations[chat_id] = []

    # Create local references to the specific chat's stacks
    message_stack = message_stacks[chat_id]
    group_conversation = group_conversations[chat_id]

    #print(f"Chat ID: {chat_id}")


    # check if update is a message
    if update.message:
        #print message to screen
        #print(update.message)

        # format datetime
        current_datetime = update.message.date
        tempdate = current_datetime.strftime("%H:%M:%S")
        custom_format = "Now it is %B %d, %Y, and it is %I:%M%p UTC"
        formatted_datetime = current_datetime.strftime(custom_format)

        first_name = update.message.from_user.first_name
        last_name = update.message.from_user.last_name 
        full_name = f"{first_name} {last_name}"


        # Extract the user's first name and the message text
        user_name = update.message.from_user.first_name
        message_text = update.message.text
        

        formatted_message = f"{user_name}: {message_text}"
        #print(f"{formatted_message}\n\n")


        # Store the message in the SQLite database
        insert_message(conn, (full_name, current_datetime, message_text))

        # Add the formatted message to the stacks
        message_stack.append(formatted_message)
        group_conversation.append(formatted_message)
        print(f"Message Stack: {message_stack}")

        if len(group_conversation) > MAX_MESSAGES:
            del group_conversation[0]

       
        
        # Is it Directed to Fiji?
        fiji_direct = False

        # if stack if over 5 or if the message begins with FIJI, consider responding
        if len(message_stack) > 5 or update.message.text.startswith(("FIJI", "fiji", "Fiji")):

            # select most recent strings from general conversation list, need to consider number
            general_conversation = select_strings(group_conversation[-3050:])

             # select most recent strings from general conversation list, need to consider number
            shorter_stack = select_strings(group_conversation[-15:])

            conversation_str_message = "\n".join(message_stack)  # gpt read for message
            conversation_str_shorter = "\n".join(shorter_stack)  # gpt for shorter context
            conversation_str_group = "\n".join(general_conversation)  # gpt for larger context

            #print(conversation_str_message)
            #print(conversation_str_shorter)
            #print(conversation_str_group)

            # probably cleaner way to mandate response if sentence begins with FIJI
            if update.message.text.startswith(("FIJI", "fiji", "Fiji")):
                should_reply = True
                fiji_direct = True
            else:
                print("NOT DOING THIS ANYMORE")
                should_reply = False

            # formulate comment with API call with past context and current comments
            if should_reply:
                #print(message_stack)
                #print(conversation_str_message)
                print(conversation_str_shorter)
                #print(conversation_str_group)
                if fiji_direct:
                   command = f"""reply to : {update.message.text}, try not to repeat your self or the message(or this message)."""
                else:
                    command = f"""
---
                            **Instructions:**

                            1. Review the "Recent conversation".
                            2. Respond ONLY to the most recent mention of your name, and address that specific query or statement.
                            3. DO NOT restate, summarize, or include any part of the original message in your response.
                            4. Be concise and avoid unnecessary details unless relevant.
                            5. Ensure to reference user names where appropriate.
                            6. Avoid greetings unless the conversation is entirely new.
                            7. Keep your response fun and lively!
                            8. Do not repeat content from prior messages.
                            9. Emulate the style in which users are conversing.
                            10. Do NOT ever use brackets in your replies.
                            11. Use "Larger context" to inform your response, but do not reference it directly.
                            12. Do NOT begin your message with "FIJI : " or "FIJI:" or "Fiji : " or "Fiji: "

                            **Example:** 
                            If Recent conversation says, "Hey, how's the weather?", your reply should be, "It's sunny!" and NOT "You asked about the weather, it's sunny!".

                            ---
                            Larger context: {conversation_str_group}
                            Recent conversation: {conversation_str_message}
                            """

                #print(command)
                try:
                    response = await call_openai_api(ai_model, command=update.message.text, larger_context=shorter_stack)
                    # clear stack if call successful
                    message_stack.clear()
                except Exception as e:
                    # Log the exception and return without sending a message
                    print(f"An error occurred while calling the OpenAI API: {e}")
                    return
                
                try:
                    
                    # add new response to group conversation list
                    formatted_response = remove_prefix_case_insensitive(response, "Fiji: ")
                    group_conversation.append(f"Fiji: {formatted_response}")
                    time_now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
                    insert_message(conn, ("Fiji", time_now, formatted_response))

                    print(f"Fiji PreFormatted : {response}")

                    #conversation_strNEW = "\n".join(group_conversation[-50:])
                    #print(f"Recent General Convo\n\n: {conversation_strNEW}")

                    # send message to channel
                    await context.bot.send_message(chat_id=update.message.chat.id, text=formatted_response)

                    # Sticker file --- is this too big?
                    your_sticker_file_id = "CAACAgEAAxkBAAEnAsJlNHEpaCLrB6VsS6IWzdw7Rp5ybQAC0AMAAvBWQEWhveTp-VuiDTAE"
                    await context.bot.send_sticker(chat_id=update.message.chat.id, sticker=your_sticker_file_id)

                except error.RetryAfter as e:
                    
                    # If we get a RetryAfter error, we wait for the specified time and then retry

                    print(f"Need to wait for {e.retry_after} seconds due to Telegram rate limits")

                    await asyncio.sleep(e.retry_after)

                    # Retry sending the message and sticker after waiting

                    await context.bot.send_message(chat_id=update.message.chat.id, text=formatted_response)
                    await context.bot.send_sticker(chat_id=update.message.chat.id, sticker=your_sticker_file_id)

                except error.TelegramError as e:
                    
                    # Handle other Telegram related errors
                    print(f"Telegram error occurred: {e.message}")
                except Exception as e:
                    
                    # Handle any other exceptions
                    print(f"An unexpected error occurred while sending the message or sticker: {e}")
        pass


def startcourt_command(update, context):
    # Call the function to start the court session
    start_court(update, context)

if __name__ == '__main__':

  

    application = ApplicationBuilder().token(
        os.getenv('TELEGRAM_BOT_TOKEN')
    ).build()

    chat_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    application.add_handler(chat_handler)

    slogan_handler = MessageHandler(filters.TEXT, slogan)
    application.add_handler(slogan_handler)

    startcourt_handler = CommandHandler('startcourt', startcourt_command)
    application.add_handler(startcourt_handler)

    #Turning off Tweets for a week, until better strategy to reset ALGO.

    #threading.Thread(target=run_tweet_loop, daemon=True).start()

    # Register the shutdown callback
    atexit.register(shutdown_callback, application)
    
    while True:
        try:
            application.run_polling()
        except TimeoutError as e:
            logger.warning(f"Timeout occurred: {e}")
            # Decide what to do next: retry, wait, or halt.
            # For example, wait for 30 seconds before retrying.
            time.sleep(30)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            # Handle other exceptions if necessary.
            break  # Or use `continue` if you want to keep the bot running.