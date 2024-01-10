import logging
import json
import openai
import os
import re
import tiktoken
import telegram
from telegram import Update, Chat, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters


# Set the path to the project directory
PATH = os.getcwd()

# Define the paths to the chat history and keys access directories
PATH_CHAT_HISTORY = os.path.join(PATH, 'chat_history')
PATH_KEYS_ACCESS = os.path.join(PATH, 'keys_access')
PATH_CHAT_ACCESS = os.path.join(PATH, 'chat_access')

BOT_USERNAME = 'big_summarizer_bot'
AI_MODEL_NAME = "gpt-3.5-turbo"
ANSWEAR_FLAG = '<answear>\n'
ASK_START_FLAG = '&'
SYSTEM_MESSAGE = """
                You are AI and your point is to sum up text and give only very short answear to user.
                But you can give user information only if you confident in it. 
                
                Below short describe of your ability as a telegram bot that need to know by your user if he has a problem.
                For now you you have three working commands <'/sum_up','/show_chats','/remove_chat'> but in future 
                the number of commands could increase. To summarize the chat first you need to be in this chat. 
                For that the user can just add you to this group and you automaticaly register this chat for this 
                user to get sum up information from it in future. If you already in group but you added there by 
                another user, the user can press "/start" command to register this chat for himself. But you don't remember
                when user use the command. If user use command but then change his mind he can cancel command sending "x" or
                written something that not fit in format that bot waiting for.

                Also you need to talk with user in language that user using in message.

                """

MAX_CHAT_HISTORY_LEN = 10000
MAX_CHAT_MEMORY_LEN = 100
MAX_TOKENS = 3000

make_prompt = lambda history_text: [{"role": "user", "content": f"""
                                            Your task is to generate a short summary 
                                            which is accurate, concise, and informative. 
                                            The summarizing should be in language of chat conversation.
                                            The summarizing should give information about
                                            the following:
                                                1.The main topics of discussion
                                                2.Retell this converstion in a very short infromative format up to 3 tense
                                                3.Short version of a few main expressions from dialoge with username of sender up to 15 expressions
                                                4.Any interesting or funny moments, if there is no such moments dont mention this point
                                            
                                            Summarize the concersation below that presented in format "@(username of sender) : (text of the message)". 

                                            Conversation: ```{history_text}```"""}]

make_access_file_name = lambda username, user_id: str(username) + '_' + str(user_id)
get_message_from_history = lambda chat_history, number: chat_history[list(chat_history.keys())[number]] 

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Define a function to check if a path exists
def check_if_needed_path_exist(pathes=[PATH_CHAT_HISTORY, PATH_KEYS_ACCESS, PATH_CHAT_ACCESS]):
    """Check if the needed paths exist.
    Args:
        pathes (list): The list of paths to check.
    Returns:
        None
    """
    for path in pathes:
        full_path = os.path.join(PATH, path)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
            print(f'Folder - "{full_path}" created')

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def load_json_file(file_name, path=PATH_CHAT_HISTORY):
    """Loads a JSON file as a dictionary."""
    file = os.path.join(path, str(file_name +'.json'))
    if os.path.isfile(file):
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return False
        
        return data
    
    return False

def save_message(chat_name, chat_id, user_id, message_id, username, message_text, chat_path=PATH_CHAT_HISTORY):
    message_data = { message_id:{
        "chat_id": chat_id,
        "user_id": user_id,
        "username": username,
        "message_text": message_text,
    }
    }

    chat_history = load_json_file(chat_name)

    if chat_history:
        if len(chat_history) >= MAX_CHAT_HISTORY_LEN:
            del chat_history[list(chat_history.keys())[0]]

        chat_history.update(message_data)
    else:
        chat_history = message_data

    with open(os.path.join(chat_path, chat_name +'.json'), "w") as file:
        json.dump(chat_history, file)

    chat_history = load_json_file(chat_name)
    return chat_history

def format_chat_from_json2text(chat_history, number_of_messages):
    list_history_slice = list(chat_history.keys())

    if len(list_history_slice) > number_of_messages:
        list_history_slice = list_history_slice[-number_of_messages:]
    chat_text = ''

    for key in list_history_slice:
        chat_element = chat_history[key]
        string = f"@{chat_element['username']} : {chat_element['message_text']}\n"
        chat_text += string

    prompt = make_prompt(chat_text)
    tokens_number = len(token_encoder.encode(prompt[0]['content']))
    print('Start tokens', tokens_number)
    while tokens_number > MAX_TOKENS:
        chat_text = chat_text[100:]
        prompt = make_prompt(chat_text)
        tokens_number = len(prompt[0]['content'])
    print('End tokens', tokens_number)

    return chat_text

def make_chatbot_history(chat_history):
    messages =  [{'role':'system', 'content':SYSTEM_MESSAGE}]

    chat_key_list = list(chat_history.keys())

    for key in chat_key_list:
        if chat_history[key]["username"] == BOT_USERNAME and re.match(f'^{ANSWEAR_FLAG}', chat_history[key]["message_text"]):
            if messages[-1]['role'] == 'assistant':
                messages.append({'role':'user', 'content': ' '})
            messages.append({'role':'assistant', 'content': chat_history[key]["message_text"]})
        elif chat_history[key]["username"] != BOT_USERNAME and re.match(f'@{BOT_USERNAME}|{ASK_START_FLAG}', chat_history[key]["message_text"]):
            if messages[-1]['role'] == 'user':
                messages.append({'role':'assistant', 'content': ' '})
            messages.append({'role':'user', 'content': chat_history[key]["message_text"]})
    
    if len(messages) > MAX_CHAT_MEMORY_LEN:
        messages = messages[-MAX_CHAT_MEMORY_LEN:]

    tokens_number = num_tokens_from_messages(messages, model=AI_MODEL_NAME)
    print('Start tokens', tokens_number)

    while tokens_number > MAX_TOKENS:
        del messages[0]
        tokens_number = num_tokens_from_messages(messages, model=AI_MODEL_NAME)
    
    print('End tokens', tokens_number)
        
    return messages

def get_completion(messages, model=AI_MODEL_NAME):
    try:
        print('try open')
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens = 500,
            timeout=10
        )
        response = response.choices[0].message["content"]
    except openai.error.APIError:
        print('except open')
        response = "Sorry, something went wrong. 😒\n I can't do it or answear your question. 😅"

    return response


async def helping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Print help instructions for user."""
    username = update.effective_user.username
    user_id = update.effective_user.id
    message_id = update.message.message_id
    chat_id = update.effective_chat.id

    save_message(username, user_id,  user_id, message_id, username, message_text='/help')

    await context.bot.send_message(chat_id=chat_id, text="""This bot can sum up the dialog from any group that was registered.\n
                                   To register croup for bot just add this bot to the group with 'Admin' privileges or if bot
                                   already in group and added him someone else then send '/start' message in this group.\n
                                   Bot can sum up dialog from group only in the your private chat.\n
                                   Bot has four commands in your personal chat:
                                   /sum_up - to sum up dialog
                                   /remove_chat - to remove chat from your own list, to remove him from group you need to do it by hand
                                   /show_chats - to show list of registered for you groups.
                                   /help - to show helping instruction\n
                                   """)
    
    save_message(username, chat_id,  user_id, message_id+1, BOT_USERNAME, message_text= f"""{ANSWEAR_FLAG}This bot can sum up the dialog from any group that was registered.\n
                                   To register croup for bot just add this bot to the group with 'Admin' privileges or if bot
                                   already in group and added him someone else then send '/start' message in this group.\n
                                   Bot can sum up dialog from group only in the your private chat.\n
                                   Bot has four commands in your personal chat:
                                   /sum_up - to sum up dialog
                                   /remove_chat - to remove chat from your own list, to remove him from group you need to do it by hand
                                   /show_chats - to show list of registered for you groups.
                                   /help - to show helping instruction\n
                                   """)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves a user to a JSON file."""
    match update.message.chat.type:
        case Chat.PRIVATE:

            username = update.effective_user.username
            user_id = update.effective_user.id
            message_id = update.message.message_id
            file_name = make_access_file_name(username, user_id)

            save_message(username, user_id,  user_id, message_id, username, message_text='/start')
            user_access = load_json_file(file_name, PATH_CHAT_ACCESS)

            buttons = [[KeyboardButton('/sum_up')], [KeyboardButton('/show_chats')],[KeyboardButton('/remove_chat')],[KeyboardButton('/help')]]
            keyboard = ReplyKeyboardMarkup(buttons, resize_keyboard=True)

            
            if user_access:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi, I think we met before 😑\nHow can I help you ?", reply_markup=keyboard)
                save_message(username, user_id,  user_id, message_id+1, BOT_USERNAME, message_text='Hi, I think we met before 😑\nHow can I help you ?')

            else:
                user_info = {}

                with open(os.path.join(PATH_CHAT_ACCESS, file_name +'.json'), "w+") as file:
                    json.dump(user_info, file)

                await context.bot.send_message(chat_id=update.effective_chat.id, text="""Hello, I'm your best friend to sum up everything :)
                                                                                        and I was created by <Comp-pot> group
                                                                                        If you want to talk with me or ask me something just 
                                                                                        write me message.
                                                                                        Also you have commands below to iteract with me.
                                                                                        If you need to cancel command you can send 'x'
                                                                                        If you need to give me access to group chat that you want to sum up in future
                                                                                        just add me to that chat as 'admin user' or just send '/start' command in the chat that 
                                                                                        I'm in if someone else added me before.\n
                                                                                        Sooo, how can I help you ?""", reply_markup=keyboard)
                
                save_message(username, user_id,  user_id, message_id+1, BOT_USERNAME, message_text= f"""{ANSWEAR_FLAG}Hello, I'm your best friend to sum up everything :)
                                                                                        and I was created by <Comp-pot> group
                                                                                        If you want to talk with me or ask me something just 
                                                                                        write me message.
                                                                                        Also you have commands below to iteract with me.
                                                                                        To cancel command if you can send 'x'
                                                                                        To give me access to group chat that you want to sum up in future
                                                                                        just add me to that chat as 'admin user' or if someone else added me before
                                                                                        just send '/start' command in the chat that I'm in.\n
                                                                                        Sooo, how can I help you ?""")


        case Chat.GROUP | Chat.SUPERGROUP:
            username = update.effective_user.username
            user_id = update.effective_user.id
            chat_name = update.message.chat.title
            chat_id = update.effective_chat.id

            file_name = make_access_file_name(username, user_id)

            access_info = {chat_name: chat_id}

            user_access = load_json_file(file_name, PATH_CHAT_ACCESS)

            if user_access != False:
                if chat_name in user_access:
                    await context.bot.send_message(chat_id, text="This chat is already registered for you 😁")

                else:
                    user_access.update(access_info)

                    with open(os.path.join(PATH_CHAT_ACCESS, file_name +'.json'), "w+") as file:
                        json.dump(user_access, file)

                    await context.bot.send_message(chat_id, text="I register this chat, as you wish 😄")
                    await context.bot.send_message(user_id, text=f"I register the chat '{chat_name}' for you 😌")
            else:
                await context.bot.send_message(chat_id, text="Sorry I don't remember you 😅\nbut if you want we could get to know each other :) \nhttps://t.me/big_summarizer_bot\nand press /start")


async def add_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_message.api_kwargs['new_chat_member']['username'] == BOT_USERNAME:

        from_user_id = update.effective_message.from_user.id
        from_username = update.effective_message.from_user.username
        chat_id = update.effective_chat.id
        chat_name = update.effective_chat.title
        to_username = update.effective_message.api_kwargs['new_chat_member']['username']
        message_id = update.message.message_id

        message_text = f'@{from_username} added @{to_username} to group {chat_name}'

        save_message(chat_name, chat_id, from_user_id, message_id, from_username, message_text)

        file_name = make_access_file_name(from_username, from_user_id)

        access_info = {chat_name: chat_id}

        user_access = load_json_file(file_name, PATH_CHAT_ACCESS)

        if user_access != False:
            if chat_name in user_access:
                await context.bot.send_message(chat_id, text="This chat is already registered for you 😁")

            else:
                user_access.update(access_info)

                with open(os.path.join(PATH_CHAT_ACCESS, file_name +'.json'), "w+") as file:
                    json.dump(user_access, file)

                await context.bot.send_message(from_user_id, text=f"I register the chat '{chat_name}' for you 😌")
        else:
            await context.bot.send_message(chat_id, text="Sorry I don't remember you 😅\nbut if you want we could get to know each other :) \nhttps://t.me/big_summarizer_bot\nand press /start")

    else:
        chat_id = update.effective_chat.id
        from_user_id = update.effective_message.from_user.id
        from_username = update.effective_message.from_user.username
        to_username = update.effective_message.api_kwargs['new_chat_member']['username']
        message_id = update.message.message_id
        chat_name = update.effective_chat.title

        message_text = f'@{from_username} added @{to_username} to group {chat_name}'

        save_message(chat_name, chat_id, from_user_id, message_id, from_username, message_text)

async def text_message_parser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves a message to a JSON file."""
    if update.edited_message:
        pass
    else:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        message_id = update.message.message_id
        username = update.effective_user.username
        message_text = update.effective_message.text

        match update.message.chat.type:
            case Chat.PRIVATE:
                chat_name = update.message.chat.username

                if message_text == 'x':

                    chat_history = load_json_file(chat_name)
                    previous_message = get_message_from_history(chat_history, -1)
                
                    save_message(chat_name, chat_id, user_id, message_id, username, message_text)

                    if previous_message['message_text'] in ['/sum_up', '/remove_chat']:
                        await context.bot.send_message(chat_id, f"Good, your command {previous_message['message_text']} canceled 😌")
                        save_message(chat_name, chat_id, user_id, message_id+1, BOT_USERNAME, f"Good, your command {previous_message['message_text']} canceled 😌")
                    else:
                        await context.bot.send_message(chat_id, "Sorry, you don't have command to cancel 😅")
                        save_message(chat_name, chat_id, user_id, message_id+1, BOT_USERNAME, "Sorry, you don't have command to cancel 😅")
                    
                else:

                    access_dict = load_json_file(make_access_file_name(username, user_id), path=PATH_CHAT_ACCESS)
                    chat_history = load_json_file(chat_name)
                    last_message = get_message_from_history(chat_history, -1)
                    access_list_upper = [key.upper() for key in list(access_dict.keys())]

                    match last_message['message_text']:
                        case '/sum_up':
                            save_message(chat_name, chat_id, user_id, message_id, username, message_text)

                            not_in_format = False
                            splited_message = message_text.split('\n')
                            parser_chat_name = splited_message[0]
                            
                            match len(splited_message):
                                case 1:
                                    parser_number = '100'
                                case 2:
                                    parser_number = splited_message[1]
                                case _:
                                    not_in_format = True

                            if not(parser_chat_name.upper() in access_list_upper):
                                await context.bot.send_message(chat_id, "Sorry you enter chat name that I don't see, try again using /sum_up command 😅")
                            elif not_in_format:
                                await context.bot.send_message(chat_id, "Sorry your instruction isn't in correct format, try again using /sum_up command 😅")
                            elif not parser_number.isdigit():
                                await context.bot.send_message(chat_id, "Sorry you write number in incorect format, try again using /sum_up command 😅")
                            else:
                                await context.bot.send_message(chat_id, ".")

                                parsing_history = load_json_file(list(access_dict.keys())[access_list_upper.index(parser_chat_name.upper())])
                                history_text = format_chat_from_json2text(parsing_history, int(parser_number))


                                await context.bot.edit_message_text(". .", chat_id, message_id+1)

                                prompt = make_prompt(history_text)
                                completion = get_completion(prompt)
                                completion = re.sub(f'^{ANSWEAR_FLAG}', '', completion)

                                save_message(chat_name, chat_id, user_id, message_id+2, BOT_USERNAME, ANSWEAR_FLAG+completion)

                                await context.bot.edit_message_text(". . .", chat_id, message_id+1)

                                try:
                                    await context.bot.send_message(chat_id, completion)
                                except telegram.error.BadRequest:
                                    await context.bot.send_message(chat_id, "Sory, something went wrong try again 😅")

                        case '/remove_chat':
                            save_message(chat_name, chat_id, user_id, message_id, username, message_text)

                            if message_text.upper() in access_list_upper:
                                file_name = make_access_file_name(username, user_id)
                                del access_dict[list(access_dict.keys())[access_list_upper.index(message_text.upper())]]
                                with open(os.path.join(PATH_CHAT_ACCESS, file_name +'.json'), "w+") as file:
                                    json.dump(access_dict, file)
                                await context.bot.send_message(chat_id, f"Good, chat '{message_text}' deleted from your list 😄")
                            else:
                                await context.bot.send_message(chat_id, "Sorry you enter chat name that I don't see, try again using /remove_chat command 😅")
                        case _:
                            await context.bot.send_message(chat_id, ".")

                            save_message(chat_name, chat_id, user_id, message_id, username, ASK_START_FLAG+message_text)
                            chat_history = load_json_file(chat_name)

                            message_text = re.sub(f'^{ASK_START_FLAG}', '', message_text)
                            
                            await context.bot.edit_message_text(". .", chat_id, message_id+1)

                            chatbot_messages = make_chatbot_history(chat_history)
                            completion = get_completion(chatbot_messages)
                            completion = re.sub(f'^{ANSWEAR_FLAG}', '', completion)

                            save_message(chat_name, chat_id, user_id, message_id+2, BOT_USERNAME, ANSWEAR_FLAG+completion)

                            await context.bot.edit_message_text(". . .", chat_id, message_id+1)
                            await context.bot.send_message(chat_id, completion)
                                    
            case Chat.GROUP | Chat.SUPERGROUP:
                chat_name = update.message.chat.title
                save_message(chat_name, chat_id, user_id, message_id, username, message_text)

                if re.match(f'@{BOT_USERNAME}', message_text):
                    await context.bot.send_message(chat_id, ".")

                    chat_history = load_json_file(chat_name)

                    await context.bot.edit_message_text(". .", chat_id, message_id+1)
                    
                    chatbot_messages = make_chatbot_history(chat_history)
                    completion = get_completion(chatbot_messages)
                    completion = re.sub(f'^{ANSWEAR_FLAG}', '', completion)

                    save_message(chat_name, chat_id, user_id, message_id+2, BOT_USERNAME, ANSWEAR_FLAG+completion)
                    await context.bot.edit_message_text(". . .", chat_id, message_id+1)

                    await context.bot.send_message(chat_id, completion)


async def show_chats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    match update.message.chat.type:
        case Chat.PRIVATE:
            username = update.effective_user.username
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            message_id = update.message.message_id

            message_text = '/show_chats'
            save_message(username, chat_id, user_id, message_id, username, message_text)

            file_name = make_access_file_name(username, user_id)
            user_access = load_json_file(file_name, PATH_CHAT_ACCESS)

            if user_access == False:
                await context.bot.send_message(chat_id, "Sorry I don't remember you 😅\nbut if you want we could get to know each other :) \nhttps://t.me/big_summarizer_bot\nand press /start")
            
            else:
                keys = user_access.keys()
                await context.bot.send_message(chat_id, "Chat that you registered:\n")
                for key in keys:
                    await context.bot.send_message(chat_id, f"{key}\n")

        case Chat.GROUP | Chat.SUPERGROUP:
            chat_name = update.message.chat.title
            chat_id = update.effective_chat.id
            user_id = update.effective_user.id
            message_id = update.message.message_id
            username = update.effective_user.username
            
            message_text = '/show_chats'
            save_message(chat_name, chat_id, user_id, message_id, username, message_text)

            await context.bot.send_message(chat_id, "Sory, I can't show your saved chats in group 😅")


async def remove_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    match update.message.chat.type:
        case Chat.PRIVATE:
            username = update.effective_user.username
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            message_id = update.message.message_id

            message_text = '/remove_chat'
            save_message(username, chat_id, user_id, message_id, username, message_text)

            file_name = make_access_file_name(username, user_id)
            user_access = load_json_file(file_name, PATH_CHAT_ACCESS)

            if user_access == False:
                await context.bot.send_message(chat_id, "Sorry I don't remember you 😅\nbut if you want we could get to know each other :) \nhttps://t.me/big_summarizer_bot\nand press /start")
            
            else:
                keys = user_access.keys()
                await context.bot.send_message(chat_id, "Chat that you registered:\n")
                for key in keys:
                    await context.bot.send_message(chat_id, f"{key}\n")
                await context.bot.send_message(chat_id, "\n\nPlease send me the name of chat that you wanna remove:\n")

        case Chat.GROUP | Chat.SUPERGROUP:
            chat_name = update.message.chat.title
            username = update.effective_user.username
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            message_id = update.message.message_id

            message_text = '/remove_chat'
            save_message(chat_name, chat_id, user_id, message_id, username, message_text)

            await context.bot.send_message(chat_id, "Sory, I can't remove chat from group 😅")


async def sum_up(update: Update, context: ContextTypes.DEFAULT_TYPE):
    match update.message.chat.type:
        case Chat.PRIVATE:
            username = update.effective_user.username
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            message_id = update.message.message_id

            message_text = '/sum_up'
            save_message(username, chat_id, user_id, message_id, username, message_text)

            file_name = make_access_file_name(username, user_id)
            user_access = load_json_file(file_name, PATH_CHAT_ACCESS)

            if user_access == False:
                await context.bot.send_message(chat_id, "Sorry I don't remember you 😅\nbut if you want we could get to know each other :) \nhttps://t.me/big_summarizer_bot\nand press /start")
            
            else:
                keys = user_access.keys()
                await context.bot.send_message(chat_id, "Chat that you registered:\n")
                for key in keys:
                    await context.bot.send_message(chat_id, f"{key}\n")
                await context.bot.send_message(chat_id, "\nTell me the name of chat from list and how many last messages you want to sum up\n(optionaly, from 1 to 1000, default=100)\nin format:\n\ntest_bot\n100")

        case Chat.GROUP | Chat.SUPERGROUP:
            chat_name = update.message.chat.title
            chat_id = update.effective_chat.id
            user_id = update.effective_user.id
            message_id = update.message.message_id
            username = update.effective_user.username
            
            message_text = '/sum_up'
            save_message(chat_name, chat_id, user_id, message_id, username, message_text)

            await context.bot.send_message(chat_id, "Sory, I can't do it in group 😅")


if __name__ == '__main__':
    # Check if the needed paths exist.
    check_if_needed_path_exist()

    keys_dict = load_json_file('keys', PATH_KEYS_ACCESS)
    token_encoder = tiktoken.encoding_for_model(AI_MODEL_NAME)

    openai.api_key = keys_dict['openai']
    application = ApplicationBuilder().token(keys_dict['telegram']).build()
    
    start_handler = CommandHandler('start', start)
    help_handler = CommandHandler('help', helping)
    sum_up_handler = CommandHandler('sum_up', sum_up)
    show_chats_handler = CommandHandler('show_chats', show_chats)
    remove_chat_handler = CommandHandler('remove_chat', remove_chat)
    new_user_handler = MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, add_new_member)
    message_handler = MessageHandler(filters.TEXT, text_message_parser)

    application.add_handler(start_handler)
    application.add_handler(help_handler)
    application.add_handler(show_chats_handler)
    application.add_handler(remove_chat_handler)
    application.add_handler(sum_up_handler)
    application.add_handler(new_user_handler)
    application.add_handler(message_handler)
    
    application.run_polling()