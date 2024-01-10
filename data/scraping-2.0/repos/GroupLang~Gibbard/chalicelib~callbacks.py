import pinecone
from chalicelib.credentials import BUCKET, MAIN_MOD_USERNAME, PINECONE_API_KEY, PINECONE_ENV, MAIN_BOT_NAME
from chalicelib.utils import (
    add_expert, add_global_mod, add_moderator, add_tool, delete_pending_queries, deregister_bot, get_all_tools, get_community_members, get_config, get_default_config,
    get_default_database, get_default_experts, get_default_methods, get_default_prompts, get_default_tools, get_experts, get_global_mods, get_main_chat_id, get_main_mod_id,
    get_moderators, get_name_to_id_dict, get_pending_queries, get_plan, get_tools, get_user_prompt, register_bot, register_user, remove_global_mod, send_typing_action,
    set_main_chat_id, store_object, update_action_with_feedback, update_chat_state)
from git import Blob, Repo
import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain.schema import AgentAction, AgentFinish
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from telegram.ext import CallbackContext
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from chalicelib.custom_objects import Query
from telegram import ParseMode, Update, ChatAction
from io import StringIO
import traceback
import io
import json
import uuid
import boto3
AWS_PROFILE = 'localstack'
boto3.setup_default_session(profile_name=AWS_PROFILE)

import logging
logging.basicConfig(level=logging.INFO)

os.environ["GIT_PYTHON_REFRESH"] = "quiet"


# s3 = boto3.client("s3")
AWS_REGION = "us-east-1"
ENDPOINT_URL = "http://localhost:4566"
s3 = boto3.client("s3", region_name=AWS_REGION, endpoint_url=ENDPOINT_URL)

# user commands
user_commands_list = [
    "*User commands:*",
    "/start - to get this message again and see the current settings",
    "/reset - to reset all of your stored messages",
    "/query - to ask a question",
    "/answer - to answer a question",
    "/help - to see all available commands",
]

# moderator commands
moderator_commands_list = [
    "*Moderator commands:*",
    "/set\_user\_prompt <user prompt> - to set the first message shown to users.",
    "/add\_moderator <username> - to add a moderator",
    "/add\_expert <username> <description> - to add an expert",
    "/add\_tool <tool name> <description> - to add a tool",
    "/show\_tools - to see all tools available to the bot",
    "/show\_experts - to see all experts available to the bot",
    "/reset\_defaults - to reset the default settings of this community",
    "/enable\_feedback - to ask moderator for feedback before sending final answer",
    "/disable\_feedback - to disable asking moderator for feedback before sending final answer",
    "/enable\_debug - to enable debug mode",
    "/disable\_debug - to disable debug mode",
    "/deregister\_bot - to DELETE ALL DATA and deregister this bot",
    "\n",
    '\n'.join(user_commands_list)
]


# # Adds moderator to the list of moderators
def cmd_add_moderator(update, context, moderator=None):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    bot_username = context.bot.get_me().username
    moderators = get_moderators(bot_username)
    # check if moderator
    if username in moderators:
        # check if user provided a moderator
        if moderator == None:
            if len(context.args) == 0:
                context.bot.send_message(
                    chat_id=chat_id, text="Please provide a moderator.", parse_mode=ParseMode.MARKDOWN,)
                return

            # add moderator
            moderator = context.args[0]

        # strip @ if present
        if moderator[0] == "@":
            moderator = moderator[1:]

        # check if user is a member of the community
        name_to_id = get_name_to_id_dict(bot_username)
        if moderator not in name_to_id.keys():
            context.bot.send_message(
                chat_id=chat_id, text=f"{moderator} is not a member of the community.")
            return

        add_moderator(bot_username=bot_username, moderator=moderator)
        context.bot.send_message(
            chat_id=chat_id, text=f"Added {moderator} as a moderator.")
    else:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry, you are not a moderator.")


def cmd_add_global_moderator(update, context, moderator=None):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    bot_username = context.bot.get_me().username
    if username != MAIN_MOD_USERNAME:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry, only the main moderator {MAIN_MOD_USERNAME} can add global moderators.")
        return

    if moderator == None:
        if len(context.args) == 0:
            context.bot.send_message(
                chat_id=chat_id, text="Please provide a moderator.", parse_mode=ParseMode.MARKDOWN,)
            return
        # add moderator
        moderator = context.args[0]

    add_global_mod(moderator)
    context.bot.send_message(
        chat_id=chat_id, text=f"Added {moderator} as a global moderator.")


def cmd_remove_global_moderator(update, context, moderator=None):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    bot_username = context.bot.get_me().username
    if username != MAIN_MOD_USERNAME:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry, only the main moderator {MAIN_MOD_USERNAME} can remove global moderators.")
        return

    if moderator == None:
        if len(context.args) == 0:
            context.bot.send_message(
                chat_id=chat_id, text="Please provide a moderator.", parse_mode=ParseMode.MARKDOWN,)
            return
        # add moderator
        moderator = context.args[0]

    remove_global_mod(moderator)
    context.bot.send_message(
        chat_id=chat_id, text=f"Removed {moderator} as a global moderator.")


def cmd_show_global_moderators(update, context):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    if username != MAIN_MOD_USERNAME:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry, only the main moderator {MAIN_MOD_USERNAME} can see global moderators.")
        return

    global_mods = get_global_mods()
    context.bot.send_message(
        chat_id=chat_id, text=f"Global moderators: {', '.join(global_mods)}")

# # add expert


def cmd_add_expert(update, context, **kwargs):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    bot_username = context.bot.get_me().username
    # check if moderator
    if username in get_moderators(bot_username):
        if kwargs.get('username') != None:
            expert_name = kwargs.get('username')
            expert_description = kwargs.get('description')
        else:
            # check if format is correct /add_expert  expert_description
            if len(context.args) < 2:
                context.bot.send_message(
                    chat_id=chat_id, text=f"Sorry, the format is incorrect. Please use /add_expert <username> <description>")
                return
            # get relevant variables
            expert_name = context.args[0]
            expert_description = ' '.join(context.args[1:])

        experts_names = [expert['name']
                         for expert in get_experts(bot_username)]
        community_members = get_community_members(bot_username)

        # strip @ if present
        if expert_name[0] == "@":
            expert_name = expert_name[1:]

        # check if user is a member of the community
        if expert_name not in community_members:
            context.bot.send_message(
                chat_id=chat_id, text=f"Sorry {expert_name} is not a member of the community. You need to write a message directly to the community bot first.")
            return

        add_expert(name=expert_name, description=expert_description,
                   community=bot_username)
        context.bot.send_message(
            chat_id=chat_id, text=f"Ok, expert added!. \nName: {expert_name} \nDescription: {expert_description}")
    else:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry, you are not a moderator.", parse_mode=ParseMode.MARKDOWN,)

# add tool


def cmd_add_tool(update, context, **kwargs):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    bot_username = context.bot.get_me().username
    # check if moderator
    if username in get_moderators(bot_username):
        if kwargs.get('name') != None:
            tool_name = kwargs.get('name')
            tool_description = kwargs.get('description')
        else:
            # check if format is correct /add_tool  tool_description
            if len(context.args) < 2:
                context.bot.send_message(
                    chat_id=chat_id, text=f"Sorry, the format is incorrect. Please use /add_tool <name> <description>")
                return
            # get relevant variables
            tool_name = context.args[0]
            tool_description = ' '.join(context.args[1:])

        tools_names = [tool['name'] for tool in get_tools(bot_username)]

        # strip @ if present
        if tool_name[0] == "@":
            tool_name = tool_name[1:]

        # check if tool is already avaiable
        elif tool_name in tools_names:
            context.bot.send_message(
                chat_id=chat_id, text=f"{tool_name} is already a tool.")
            return

        add_tool(name=tool_name, description=tool_description,
                 community=bot_username)
        context.bot.send_message(
            chat_id=chat_id, text=f"Sure!, I added {tool_name} as a tool.")
    else:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry, you are not a moderator.", parse_mode=ParseMode.MARKDOWN,)

# Shows  all available commands to user


def cmd_help(update, context):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    # check if moderator
    if username in get_moderators(bot_username):
        context.bot.send_message(chat_id=chat_id, text='\n'.join(
            moderator_commands_list) + "\n", parse_mode=ParseMode.MARKDOWN,)
        context.bot.send_message(
            chat_id=chat_id, text="You can also just tell me what you want and I'll try to figure out which command you want to use.")
    else:
        context.bot.send_message(chat_id=chat_id, text='\n'.join(
            user_commands_list) + "\n", parse_mode=ParseMode.MARKDOWN,)
        context.bot.send_message(
            chat_id=chat_id, text="You can also just tell me what you want and I'll try to figure out which command you want to use.")

# gets the username from the message and description and runs the add_expert command


def cmd_introduce(update, context, description=None):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    bot_username = context.bot.get_me().username
    # check if moderator
    if username in get_moderators(bot_username):
        if description is None:
            description = ' '.join(context.args)
        else:
            description = description
        add_expert(name=username, description=description,
                   community=bot_username)
        context.bot.send_message(
            chat_id=chat_id, text=f"Added expert. \nName: {username} \nDescription: {description}")

    else:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry, you are not a moderator.", parse_mode=ParseMode.MARKDOWN,)

# Resets the default settings of the community (prompts and methods)


def cmd_reset_defaults(update, context):
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    bot_username = context.bot.get_me().username
    moderators_key = f"{bot_username}/moderators"

    # get moderators
    response = s3.get_object(Bucket=BUCKET, Key=moderators_key)
    moderators = json.loads(response['Body'].read().decode('utf-8'))

    if (username == MAIN_MOD_USERNAME or username in moderators):
        context.bot.send_message(
            chat_id=chat_id, text=f"Hey {username}. Ok, I will reset the default settings of this community", parse_mode=ParseMode.MARKDOWN,)
        # get default prompts from s3 PRIVATE_BUCKET
        prompts = get_default_prompts()

        # get default methods from s3 PRIVATE_BUCKET
        methods = get_default_methods()

        # get default database.json from s3 PRIVATE_BUCKET
        database = get_default_database()
        csv_buffer = StringIO(database)

        # get default experts
        experts = get_default_experts()

        # reset pending queries
        pending_queries = {}

        # reset tools
        tools = get_default_tools()

        # config
        config = get_default_config()

        # set default settings
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/methods.json", Body=json.dumps(methods))
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/prompts.json", Body=json.dumps(prompts))
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/database.csv", Body=csv_buffer.getvalue())
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/experts.json", Body=json.dumps(experts))
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/tools.json", Body=json.dumps(tools))
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/config.json", Body=json.dumps(config))
        store_object(obj=pending_queries, name="pending_queries",
                     community=bot_username)

        context.bot.send_message(
            chat_id=chat_id, text=f"Done!", parse_mode=ParseMode.MARKDOWN,)
    else:
        context.bot.send_message(
            chat_id=chat_id, text=f"Hey {username}. You are not allowed to reset the default settings of this community", parse_mode=ParseMode.MARKDOWN,)


# Shows defaults and initial instructions (depends on the role of the user)
def cmd_start(update, context):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    user_id = update.message.from_user.id

    if not MAIN_BOT_NAME:
        raise Exception(
            "looks like you haven't set the MAIN_BOT_NAME variable")

    # check if main bot
    if bot_username == MAIN_BOT_NAME:
        text_1 = """Hey there. Send the /newbot command to https://t.me/botfather to create your bot. Once you have the bot's token, send it to me in the following format /token <bot's token>."""
        text_2 = " If you want your bot to be able to read the messages once added to a group then you need to disable 'Group Privacy' by sending the /mybots command to https://t.me/botfather then selecting your bot > 'Bot Settings' > 'Group Privacy' > 'Turn off'."
        context.bot.send_message(chat_id=chat_id, text=text_1.replace(
            "\n", "").replace("\t", ""), parse_mode=ParseMode.MARKDOWN,)
        context.bot.send_message(chat_id=chat_id, text=text_2.replace(
            "\n", "").replace("\t", ""), parse_mode=ParseMode.MARKDOWN,)
        return

    # register user if not already registered
    community_members = get_community_members(bot_username)
    if username not in community_members:
        register_user(bot_username, user_id=user_id, username=username)
        context.bot.send_message(
            chat_id=chat_id, text=f"You have been registered {username}. Welcome to the community!")
    try:
        user_prompt = get_config(bot_username)['prompts']['user']
    except Exception as e:
        print(f"Error getting user prompt: {e}")
        logging.info(f"Error getting user prompt: {e}")
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry there was an error, please report it to the moderators.")
    # user_prompt = get_user_prompt(bot_username)
    context.bot.send_message(
        chat_id=chat_id, text=user_prompt, parse_mode=ParseMode.MARKDOWN,)


# # Deregisters a bot
def cmd_deregister_bot(update, context):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    username = update.message.from_user.username
    # check if main bot
    if bot_username == MAIN_BOT_NAME:
        text_1 = "You can't deregister the main bot."
        context.bot.send_message(
            chat_id=chat_id, text=text_1, parse_mode=ParseMode.MARKDOWN,)
        return
    # check if moderator
    moderators = get_moderators(bot_username)
    if (username == MAIN_MOD_USERNAME or username in moderators):
        context.bot.send_message(
            chat_id=chat_id, text="Bot deregistered successfully! ALL data has been deleted and you can no longer talk to me :/.", parse_mode=ParseMode.MARKDOWN,)
        deregister_bot(bot_username)
    else:
        text_1 = "You don't have permission to deregister this bot."
        context.bot.send_message(
            chat_id=chat_id, text=text_1, parse_mode=ParseMode.MARKDOWN,)

# Registers a new bot


@send_typing_action
def cmd_token(update, context, token=None):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    username = update.message.from_user.username

    if token is None:
        token = ' '.join(context.args)
    else:
        token = token

    global_mods = get_global_mods()

    # check if main bot
    if bot_username == MAIN_BOT_NAME:
        if username != MAIN_MOD_USERNAME and username not in global_mods:
            context.bot.send_message(
                chat_id=chat_id, text="Sorry, you don't have permission to register a bot.")
            return
        if token:
            try:
                bot_username = register_bot(
                    bot_token=token, moderator=username, main_chat_id=chat_id)
            except Exception as e:
                print("Error registering bot: ", e)
                logging.info("Error registering bot: ", e)
                print(traceback.format_exc())
                context.bot.send_message(
                    chat_id=chat_id, text="Bot could not be registered. Please check your token and command format and try again.")
            else:
                # get bot username
                # prefix any underscores in bots name with a backslash
                bot_username_ = bot_username.replace("_", "\_")
                text_2 = f"Bot registered successfully!. As the first member you are a moderator and can now go setup your community by sending the `/start` command to t.me/{bot_username_} and have a one-to-one chat with the bot."
                #
                create_group_instructions_list = [
                    "iOS: Start a new message (tap the icon in the top right corner in Chats) > 'New Group'.",
                    "Android: Tap the circular pencil icon in the chat list > 'New Group'.",
                    "Telegram Desktop: Click the menu button in the top left corner > 'New Group'.\n"]
                add_bot_instructions_list = [
                    f"go to t.me/{bot_username}, click the bot's icon in the top bar and then click 'Add to Group or Channel, then select your group and click 'Add bot as a Member'.\n",
                    "You can also add the bot by going in the group, clicking the group's name in the top bar and then clicking 'Add Members', then search for the bot and click 'Add'.\n",
                    "After you have added the bot promote it to admin by going into the group, clicking the group's name in the top bar, pressing the bot's name and then clicking 'Promote to Admin'.\n"
                ]
                text_3 = [
                    "You can also create a group by following these instructions:",
                    '\n'.join(create_group_instructions_list),
                    "Then add the bot to your group by following these instructions:",
                    '\n'.join(add_bot_instructions_list),
                    "(group mode still in early test phase so it will be buggy)."
                ]

                # main_mod_id = int(get_main_mod_id())
                # context.bot.send_message(chat_id=main_mod_id, text=f"New bot {bot_username_} registered by {username}.", parse_mode=ParseMode.MARKDOWN,)
                context.bot.send_message(
                    chat_id=chat_id, text=text_2, parse_mode=ParseMode.MARKDOWN,)
                context.bot.send_message(
                    chat_id=chat_id, text='\n'.join(text_3))
        else:
            context.bot.send_message(
                chat_id=chat_id, text="Did not find your token. Please use the format: /token <bot's token>", parse_mode=ParseMode.MARKDOWN,)
    else:
        context.bot.send_message(
            chat_id=chat_id, text="New bots can only be registered by the main bot.")

# show current tools to user


def cmd_show_tools(update, context):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    # get tools
    # tools = get_tools(bot_username)
    tools = get_all_tools(bot_username,)
    # if no tools, send message
    if not tools:
        context.bot.send_message(
            chat_id=chat_id, text="No tools found. Please ask a moderator to add some first.")
        return
    # send experts
    text = "The following tools are available: \n"
    for tool in tools:
        text += f"name :{tool.name}. description: {tool.description}\n"
        # text += f"name :{tool['name']}. description: {tool['description']}\n"

    context.bot.send_message(chat_id=chat_id, text=text)

# show current experts to user


def cmd_show_experts(update, context):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    # get experts
    experts = get_experts(bot_username)
    # if no experts, send message
    if not experts:
        context.bot.send_message(
            chat_id=chat_id, text="No experts found. Please ask a moderator to add some first.")
        return
    # send experts
    text = "The following experts are available: \n"
    for expert in experts:
        text += f"name :{expert['name']}. description: {expert['description']}\n"
    context.bot.send_message(chat_id=chat_id, text=text)

# show pending queries


def cmd_show_pending_queries(update, context):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    # get pending queries
    queries = get_pending_queries(bot_username)
    # if no queries, send message
    if not queries:
        context.bot.send_message(
            chat_id=chat_id, text="No pending queries found.")
        return
    # send queries
    text = "The following queries are pending: \n"
    for query in queries.values():
        text += f"query client :{query.client_username}. expert asked: {query.expert_username}. query: {query.query_text}\n"
    context.bot.send_message(chat_id=chat_id, text=text)


def cmd_delete_pending_queries(update, context):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    # get pending queries
    queries = get_pending_queries(bot_username)
    # if no queries, send message
    if not queries:
        context.bot.send_message(
            chat_id=chat_id, text="No pending queries found.")
        return
    # delete queries
    delete_pending_queries(bot_username)
    context.bot.send_message(chat_id=chat_id, text="Pending queries deleted.")


def ask_expert(query, context, update, get_feedback=False):
    bot_username = context.bot.username
    pending_queries = get_pending_queries(bot_username)
    chat_id = update.message.chat_id

    if get_feedback:
        # ask moderator for feedback on the action intended (check if it makes sense)
        message = f"I received the question \n'''{query.query_text}'''\nand my answer is:\n'''{query.final_answer}'''\n Is my reasoning ok or do you have any feedback?"
        chat_state = "waiting_for_feedback"
    else:
        # check if expert is already helping someone else
        if query.expert_id in pending_queries:
            return False
        # the client is the expert of the meaning of the query
        if query.expert_username == query.client_username:
            # ask clarification question
            message = f"{query.last_action.tool_input}"
            chat_state = "waiting_for_clarification"
        else:
            message = f"Hello {query.expert_username}, I need your help with this question: \n *{query.last_action.tool_input}*"
            chat_state = "waiting_for_answer"

    # ask expert for help
    try:  # TODO: remove this try catch as it seems unecessary
        chat_id = int(query.moderator_id) if get_feedback else int(
            query.expert_id)
        context.bot.send_message(chat_id=chat_id, text=message)
        update_chat_state(bot_username, chat_id, chat_state)

    except Exception as e:
        chat_id = update.message.chat_id
        context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
        return False

    # add query to pending queries and save
    pending_queries[chat_id] = query
    store_object(obj=pending_queries, name="pending_queries",
                 community=bot_username)

    return True


@send_typing_action
def cmd_query(update, context, query=None):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    username = update.message.from_user.username  # change for group name

    # check if callback handler or bot command
    if query is None:
        query_text = ' '.join(context.args)
    else:
        query_text = query

    moderator_id = get_main_chat_id(bot_username)

    # get plan
    try:
        action = get_plan(bot_username, query_text, username=username)
    except Exception as e:
        context.bot.send_message(chat_id=moderator_id, text=f"Error: {e}")
        return
    context.bot.send_message(
        chat_id=chat_id, text=f"Ok, I'll try to answer your question. It might take a while since I have to do some research and maybe verify with experts. I'll let you know when I'm done :).")

    # create query object
    query = Query(client_username=username, client_id=chat_id,
                  query_text=query_text, last_action=action, moderator_id=moderator_id)
    debug = True if get_config(bot_username)['debug'] == 'yes' else False
    if debug:
        context.bot.send_message(
            chat_id=query.moderator_id, text=f"I've just received the query \n'''{query.query_text}'''\nfrom @{query.client_username} and I'm going to try to answer it.\nSince debug mode is enabled I'll log all of my chain of thought here.")
        context.bot.send_message(chat_id=query.moderator_id, text=action.log)

    query_loop(update, context, query)


def query_loop(update, context, query):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    name_to_id = get_name_to_id_dict(bot_username)
    # TODO: intergate update below better.
    # update the id of the asking user to be the id of the chat in which the question was made (to avoid bot asking for clarification in private chat when user asked question in a group chat)
    name_to_id.update({query.client_username: query.client_id})
    # TODO: add original client as an expert
    experts = get_experts(bot_username)
    tools = get_all_tools(bot_username, client_username=query.client_username)
    name_to_tool_map = {tool.name: tool for tool in tools}
    iterations = 0
    max_iterations = 6
    action = query.last_action
    debug = True if get_config(bot_username)['debug'] == 'yes' else False

    if len(tools) == 0:
        context.bot.send_message(
            chat_id=chat_id, text="There are no tools or experts available. Please ask a moderator to add some experts or tools.")
        return
    while isinstance(action, AgentAction) and (action.tool in [tool.name for tool in tools]) and iterations < max_iterations:
        iterations += 1
        # check if tool is a human expert
        if action.tool in [expert["name"] for expert in experts] + [query.client_username]:
            # ask human
            try:
                expert_id = name_to_id[action.tool]
            except KeyError:
                context.bot.send_message(
                    chat_id=chat_id, text=f"Could not find expert {action.tool}")
                return
            query.expert_id = expert_id
            query.expert_username = action.tool
            question_sent = ask_expert(query, context, update=update)
            if question_sent:
                if query.expert_id != chat_id:
                    context.bot.send_message(
                        chat_id=query.client_id, text=f"Ok, I've asked an expert for help. It can take some time for them to respond. I'll let you know when they do.")
                return
            else:
                observation = f'Expert {query.expert_username} is already helping someone else.'
                if debug:
                    context.bot.send_message(
                        chat_id=query.moderator_id, text=f'Observation: {observation}')
                query.intermediate_steps.append(
                    (query.last_action, observation))
                try:
                    action = get_plan(bot_username=bot_username, query_text=query.query_text,
                                      username=query.client_username, intermediate_steps=query.intermediate_steps)
                except Exception as e:
                    context.bot.send_message(
                        chat_id=chat_id, text=f"Error: {e}")
                    return
                query.last_action = action
                if debug:
                    # send observaion to user
                    context.bot.send_message(
                        chat_id=query.moderator_id, text=f'Observation: {observation} \n Thought: {action.log}')
        else:
            # run tool
            tool = name_to_tool_map[action.tool]
            tool_input = action.tool_input
            observation = tool.run(tool_input)
            if debug:
                context.bot.send_message(
                    chat_id=query.moderator_id, text=f"Observation: {observation}")
            query.intermediate_steps.append((query.last_action, observation))
            try:
                action = get_plan(bot_username=bot_username, query_text=query.query_text,
                                  username=query.client_username, intermediate_steps=query.intermediate_steps)
            except Exception as e:
                context.bot.send_message(
                    chat_id=query.client_id, text="Sorry, I couldn't find a solution to your problem. Please try again later or ask the moderators for help :(.")
                context.bot.send_message(
                    chat_id=query.moderator_id, text=f"Error: {e}")
                return
            query.last_action = action
            if debug:
                context.bot.send_message(
                    chat_id=query.moderator_id, text=f'Thought: {action.log}')

    if isinstance(action, AgentFinish):
        query.final_answer = action.return_values['output']
        config = get_config(bot_username)
        feedback_mode = config['feedback_mode']
        if feedback_mode == 'yes':
            query.needs_feedback = True
        if query.needs_feedback:
            # verification if final makes sense
            asked_for_feedback = ask_expert(
                query, context, update=update, get_feedback=True)
            if not asked_for_feedback:
                context.bot.send_message(
                    chat_id=query.client_id, text="Sorry, there was an error. Please try again later or ask the moderators for help :(.")
                context.bot.send_message(
                    chat_id=query.moderator_id, text="There was an error asking for feedback on the final answer.")
        else:
            context.bot.send_message(
                chat_id=query.client_id, text=query.final_answer)
            debug = True if get_config(bot_username)[
                'debug'] == 'yes' else False
            if debug:
                context.bot.send_message(
                    chat_id=query.moderator_id, text="Ok, I've sent the final answer to the user.")

        return
    elif iterations >= max_iterations:
        context.bot.send_message(
            chat_id=chat_id, text=f"Max iterations reached.")
        return
    else:
        context.bot.send_message(
            chat_id=chat_id, text=f"Unexpected tool: {action.tool}")
        return


@send_typing_action
def cmd_answer(update, context, answer=None):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    pending_queries = get_pending_queries(bot_username=bot_username)
    expert_id = chat_id

    update_chat_state(bot_username=bot_username,
                      chat_id=chat_id, chat_state="normal")
    # check if callback handler or if the command was routed by the bot
    if answer is None:
        observation = ' '.join(context.args)
    else:
        observation = answer

    # check if answer is for a pending query
    if expert_id not in pending_queries:
        context.bot.send_message(
            chat_id=chat_id, text=f"Sorry but there are no pending queries for you.")
        return

    # retrieve query and remove it from pending queries
    query = pending_queries[expert_id]
    del pending_queries[expert_id]
    store_object(obj=pending_queries, name="pending_queries",
                 community=bot_username)

    # check if answer is feedback provided by a moderator
    # TODO: generalize to any action. ( for now only final answer is checked. )
    if query.needs_feedback:
        feedback = observation
        if feedback.strip().lower() == "ok":
            context.bot.send_message(
                chat_id=query.client_id, text=query.final_answer)
            context.bot.send_message(
                chat_id=query.moderator_id, text="Great, I've sent the final answer to the user.")
        else:
            try:
                final_answer_updated = update_action_with_feedback(
                    query=query.query_text, answer=query.final_answer, feedback=feedback, bot_username=bot_username)
            except Exception as e:
                context.bot.send_message(
                    chat_id=query.client_id, text="Sorry, there was an error. Please try again later or ask the moderators for help :(.")
                context.bot.send_message(
                    chat_id=query.moderator_id, text=f"Error getting updated final answer: {e}")
                return
            context.bot.send_message(
                chat_id=query.client_id, text=final_answer_updated)
            context.bot.send_message(
                chat_id=query.moderator_id, text=f"Ok, I've updated my answer based on your feedback and sent it to the user. \nFinal answer: {final_answer_updated}")
        return

    # add observation to query
    query.intermediate_steps.append((query.last_action, observation))
    context.bot.send_message(chat_id=query.moderator_id,
                             text=f"Observation: {observation}")

    try:
        action = get_plan(bot_username=bot_username, query_text=query.query_text,
                          username=query.client_username, intermediate_steps=query.intermediate_steps)
    except Exception as e:
        context.bot.send_message(
            chat_id=query.moderator_id, text=f"Error: {e}")
        return

    query.last_action = action
    context.bot.send_message(chat_id=query.moderator_id, text=action.log)

    # enter query loop
    query_loop(context=context, update=update, query=query)


def new_member(update, context):
    bot_username = context.bot.get_me().username
    for member in update.message.new_chat_members:
        if member.username == bot_username:
            set_main_chat_id(bot_username=bot_username,
                             chat_id=update.message.chat_id)
            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=f"Hello! I am {bot_username}. I have set this group as the main moderators group, I'll route my logs to this channel :).\n You can type /start to see what I can do. \n Also I can be much more helpful if you promote me to admin!.")
            return
        else:
            add_moderator(bot_username=bot_username, moderator=member.username)
            welcome_text = f"Hello @{member.username}! I am {bot_username}. I have added you as a moderator of this community :). Please send me a message directly (on our private chat) so that I can send you messages asking you for help if I need it."
            # I can help you with your queries. You can type /start to see what I can do."
            instruction_text = f"Also @{member.username}, it would help me if you introduce yourself and tell me more about what your expertise is. \n e.g. I am an expert in python programming.\n"
            instruction_text += "This will help me know when to ask you for help when answering user queries."
            context.bot.send_message(
                chat_id=update.message.chat_id, text=welcome_text)
            context.bot.send_message(
                chat_id=update.message.chat_id, text=instruction_text)
            return


# command that gets the tools that are available, removes the specified tool from the list, and stores the list back
def cmd_remove_tool(update, context, tool_name=None):
    chat_id = update.message.chat_id
    bot_username = context.bot.get_me().username

    if tool_name is None:
        tool_name = ' '.join(context.args)
    try:
        tools = get_tools(bot_username)
    except Exception as e:
        context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
        return
    if len(tools) == 0:
        context.bot.send_message(chat_id=chat_id, text=f"No tools found.")
        return
    if tool_name not in [tool['name'] for tool in tools]:
        context.bot.send_message(
            chat_id=chat_id, text=f"Tool {tool_name} not found.")
        return
    tools = [tool for tool in tools if tool['name'] != tool_name]
    s3.put_object(
        Bucket=BUCKET, Key=f"{bot_username}/tools.json", Body=json.dumps(tools))
    context.bot.send_message(
        chat_id=chat_id, text=f"Tool {tool_name} removed.")


@send_typing_action
def cmd_ask(update, context):
    query = ' '.join(context.args)
    bot_username = context.bot.get_me().username
    # context.bot.send_message(chat_id=update.effective_chat.id, text="Ok. your query is: " + query)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )
    embeddings = OpenAIEmbeddings()

    index_name = "grouplang"
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    retriever = docsearch.as_retriever()

    llm = OpenAI(temperature=0.9, model_name="text-davinci-003")
    # send typing action
    context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Created QA Chain ...")

    try:
        llm_opinion = qa.run(query)
    except Exception as e:
        print(f'Could not run LLM Chain: {e}')
        logging.info(f'Could not run LLM Chain: {e}')
        context.bot.send_message(
            chat_id=update.effective_chat.id, text="Could not run LLM Chain: \n" + str(e))
        return
    context.bot.send_message(
        chat_id=update.effective_chat.id, text=llm_opinion)


def handle_document(update: Update, context: CallbackContext):
    document = update.message.document
    bot_username = context.bot.get_me().username
    embeddings = OpenAIEmbeddings()
    filename = document.file_name.split(".")[0]
    # store document title in s3 as simple string
    s3_key = f"{bot_username}/current_document_name"
    s3.put_object(Bucket=BUCKET, Key=s3_key, Body=filename)

    # Check if it's a text document based on MIME type
    if document.mime_type == 'text/plain':
        file_id = document.file_id
        file = context.bot.getFile(file_id)
        doc_name = "newest"
        # Download the file into s3
        s3_key = f"{bot_username}/documents/{doc_name}.txt"
        with io.BytesIO() as f:
            file.download(out=f)
            f.seek(0)
            s3.upload_fileobj(f, BUCKET, s3_key)

        # Read the file from s3
        s3_response = s3.get_object(Bucket=BUCKET, Key=s3_key)
        text = s3_response['Body'].read().decode('utf-8')

    elif document.mime_type == 'application/pdf':
        import PyPDF2
        # get the text from the f
        file_id = document.file_id
        file = context.bot.getFile(file_id)
        doc_name = "newest"
        # Download the file into s3
        s3_key = f"{bot_username}/documents/{doc_name}.pdf"
        with io.BytesIO() as f:
            file.download(out=f)
            f.seek(0)
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num].extract_text()
                # Split the text into lines
                lines = page.splitlines()
                # Print each line
                for line in lines:
                    text += line + '\n'
    else:
        update.message.reply_text(
            f'Sorry I still cannot handle {document.mime_type} documents.')
        return

    # Your processing logic here
    update.message.reply_text(
        f"I've received your text document. It has {len(text)} characters.")
    update.message.reply_text(
        f"Here are the first lines of it:\n\n {text[:200]}...")

    # delete all info in index
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )
    index_name = 'grouplang'
    index = pinecone.Index(index_name=index_name)

    # text = s3_response['Body'].read()
    metadata = {"source": s3_key}
    documents = [Document(page_content=text, metadata=metadata)]
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator="\n")
    docs = text_splitter.split_documents(documents)

    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Indexing document...")

    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # docsearch = Pinecone.from_texts(texts, embeddings, metadatas, index_name=index_name)
    batch_size = 32
    text_key = "text"
    ids = None
    namespace = None
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=f"Number of chunks = {len(texts)}. Batch size = {batch_size} \n Entering indexing loop...")
    for i in range(0, len(texts), batch_size):
        # set end position of batch
        i_end = min(i + batch_size, len(texts))
        # get batch of texts and ids
        lines_batch = texts[i:i_end]
        # create ids if not provided
        if ids:
            ids_batch = ids[i:i_end]
        else:
            ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
        # create embeddings
        embeds = embeddings.embed_documents(lines_batch)
        context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"Batch {i} to {i_end} embedded.")
        print(f"Batch {i} to {i_end} embedded.")
        logging.info(f"Batch {i} to {i_end} embedded.")
        # prep metadata and upsert batch
        if metadatas:
            metadata = metadatas[i:i_end]
        else:
            metadata = [{} for _ in range(i, i_end)]
        for j, line in enumerate(lines_batch):
            metadata[j][text_key] = line
        to_upsert = zip(ids_batch, embeds, metadata)

        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert), namespace=namespace)
        print(f"Batch {i} to {i_end} upserted.")
        logging.info(f"Batch {i} to {i_end} upserted.")
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Ok, the document is indexed :).")


def cmd_reset_index(update: Update, context: CallbackContext):
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )
    index_name = 'grouplang'
    index = pinecone.Index(index_name=index_name)
    stats = index.describe_index_stats()
    namespaces = stats["namespaces"]
    for ns in namespaces:
        index.delete(delete_all=True, namespace=ns)
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Ok, I have reset the index.")


def cmd_enable_feedback(update: Update, context: CallbackContext):
    bot_username = context.bot.get_me().username
    username = update.message.from_user.username
    moderators = get_moderators(bot_username)
    # check if moderator
    if username in moderators:
        config = get_config(bot_username)
        config['feedback_mode'] = 'yes'
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/config.json", Body=json.dumps(config))
        context.bot.send_message(
            chat_id=update.effective_chat.id, text="Ok, feedback mode is enabled.")
    else:
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="Sorry, that command is only available to moderators.")


def cmd_disable_feedback(update: Update, context: CallbackContext):
    bot_username = context.bot.get_me().username
    username = update.message.from_user.username
    moderators = get_moderators(bot_username)
    # check if moderator
    if username in moderators:
        config = get_config(bot_username)
        config['feedback_mode'] = 'no'
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/config.json", Body=json.dumps(config))
        context.bot.send_message(
            chat_id=update.effective_chat.id, text="Ok, feedback mode is disabled.")

# enable debug


def cmd_enable_debug(update: Update, context: CallbackContext):
    bot_username = context.bot.get_me().username
    username = update.message.from_user.username
    moderators = get_moderators(bot_username)
    # check if moderator
    if username in moderators:
        config = get_config(bot_username)
        config['debug'] = 'yes'
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/config.json", Body=json.dumps(config))
        context.bot.send_message(
            chat_id=update.effective_chat.id, text="Ok, debug mode is enabled.")
    else:
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="Sorry, that command is only available to moderators.")

# disable debug


def cmd_disable_debug(update: Update, context: CallbackContext):
    bot_username = context.bot.get_me().username
    username = update.message.from_user.username
    moderators = get_moderators(bot_username)
    # check if moderator
    if username in moderators:
        config = get_config(bot_username)
        config['debug'] = 'no'
        s3.put_object(
            Bucket=BUCKET, Key=f"{bot_username}/config.json", Body=json.dumps(config))
        context.bot.send_message(
            chat_id=update.effective_chat.id, text="Ok, debug mode is disabled.")
    else:
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="Sorry, that command is only available to moderators.")


def handle_url(update: Update, context: CallbackContext, url=None):
    bot_username = context.bot.get_me().username
    chat_id = update.message.chat_id
    url = update.message.text
    tmp_repo_path = '/tmp/repo'
    context.bot.send_message(chat_id=chat_id, text=f"Got url {url}.")
    repo = Repo.clone_from(url, tmp_repo_path)
    try:
        repo.git.checkout("main")
    except:
        repo.git.checkout("master")
    # inform user
    context.bot.send_message(
        chat_id=chat_id, text=f"Cloned repo {url} to {tmp_repo_path}.")

    docs = []
    current_file = 0
    print("loading files from repository...")
    logging.info("loading files from repository...")
    for item in repo.tree().traverse():
        if not isinstance(item, Blob):
            continue
        file_path = os.path.join(tmp_repo_path, item.path)

        ignored_files = repo.ignored([file_path])
        if len(ignored_files):
            continue

        rel_file_path = os.path.relpath(file_path, tmp_repo_path)
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                file_type = os.path.splitext(item.name)[1]

                # loads only text files
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                metadata = {
                    "source": rel_file_path,
                    "file_path": rel_file_path,
                    "file_name": item.name,
                    "file_type": file_type,
                }
                doc = Document(page_content=text_content, metadata=metadata)
                docs.append(doc)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            logging.info(f"Error reading file {file_path}: {e}")
        current_file += 1
        if current_file % 10 == 0:
            print(f'loaded {current_file} files', end='\r')
            logging.info(f'loaded {current_file} files')
            context.bot.send_message(
                chat_id=chat_id, text=f'loaded {current_file} files')

    ext_to_lang = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".java": Language.JAVA,
        ".go": Language.GO,
        ".cpp": Language.CPP,
        ".php": Language.PHP,
        ".proto": Language.PROTO,
        ".rst": Language.RST,
        ".rb": Language.RUBY,
        ".rs": Language.RUST,
        ".scala": Language.SCALA,
        ".swift": Language.SWIFT,
        ".md": Language.MARKDOWN,
        ".tex": Language.LATEX,
        ".html": Language.HTML,
    }

    all_docs = []

    context.bot.send_message(
        chat_id=chat_id, text=f'loaded {current_file} files. Splitting files into chunks...')
    for doc in docs:
        print(doc.metadata['file_type'])
        logging.info(doc.metadata['file_type'])
        if doc.metadata['file_type'] in ext_to_lang:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=ext_to_lang[doc.metadata['file_type']])
        else:
            splitter = CharacterTextSplitter()
        split_docs = splitter.split_documents([doc])
        all_docs.extend(split_docs)
    context.bot.send_message(
        chat_id=chat_id, text=f'Split into {len(all_docs)} chunks.')

    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )

    index_name = 'grouplang'
    index = pinecone.Index(index_name=index_name)

    texts = [d.page_content for d in all_docs]
    metadatas = [d.metadata for d in all_docs]

    batch_size = 32
    text_key = "text"
    ids = None
    namespace = None
    for i in range(0, len(texts), batch_size):
        # set end position of batch
        i_end = min(i + batch_size, len(texts))
        # get batch of texts and ids
        lines_batch = texts[i:i_end]
        # create ids if not provided
        if ids:
            ids_batch = ids[i:i_end]
        else:
            ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
        # create embeddings
        embeds = embeddings.embed_documents(lines_batch)
        context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"Batch {i} to {i_end} embedded.")

        # prep metadata and upsert batch
        if metadatas:
            metadata = metadatas[i:i_end]
        else:
            metadata = [{} for _ in range(i, i_end)]
        for j, line in enumerate(lines_batch):
            metadata[j][text_key] = line
        to_upsert = zip(ids_batch, embeds, metadata)

        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert), namespace=namespace)
        print(f"Batch {i} to {i_end} embedded and indexed.")
        logging.info(f"Batch {i} to {i_end} embedded and indexed.")

    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Ok, the repository is indexed :).")

# def cmd_relevant_docs(update, context, query=None):
#     bot_username = context.bot.get_me().username
#     index_name = 'grouplang'
#     chat_id = update.message.chat_id
#     if query is None:
#         query_text = ' '.join(context.args)
#     else:
#         query_text = query
#     try:
#         docs = get_relevant_docs(index_name=index_name, query=query_text)
#     except Exception as e:
#         context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
#         return
#     if len(docs) == 0:
#         context.bot.send_message(chat_id=chat_id, text=f"No relevant docs found.")
#         return
#     context.bot.send_message(chat_id=chat_id, text=f"{docs}")

# def cmd_relevant_texts(update, context, query=None):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     index_name = 'grouplang'

#     if query is None:
#         query_text = ' '.join(context.args)
#     else:
#         query_text = query
#     try:
#         texts = get_relevant_texts(index_name=index_name, query=query_text)
#     except Exception as e:
#         context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
#         return
#     if len(texts) == 0:
#         context.bot.send_message(chat_id=chat_id, text=f"No relevant texts found.")
#         return
#     context.bot.send_message(chat_id=chat_id, text=f"{texts}")

# def cmd_all_docs(update, context):
#     chat_id = update.message.chat_id
#     try:
#         docs = get_all_docs(index_name='grouplang')
#     except Exception as e:
#         context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
#         return
#     if len(docs) == 0:
#         context.bot.send_message(chat_id=chat_id, text=f"No docs found.")
#         return
#     context.bot.send_message(chat_id=chat_id, text=f"{docs}")

# def cmd_all_texts(update, context):
#     chat_id = update.message.chat_id
#     try:
#         texts = get_all_texts(index_name='grouplang')
#     except Exception as e:
#         context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
#         return
#     if len(texts) == 0:
#         context.bot.send_message(chat_id=chat_id, text=f"No texts found.")
#         return
#     context.bot.send_message(chat_id=chat_id, text=f"{texts}")

# def cmd_delete_all_entries(update, context):
#     chat_id = update.message.chat_id
#     index_name = 'grouplang'
#     namespace = 'test'
#     try:
#         delete_all_entries(index_name=index_name, namespace=namespace)
#     except Exception as e:
#         context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
#         return
#     context.bot.send_message(chat_id=chat_id, text=f"All entries deleted.")

# @send_typing_action
# def cmd_task(update, context):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     query = ' '.join(context.args)

#     context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
#     context.bot.send_message(chat_id=chat_id, text="Ok let me think of a plan....")

#     plan = get_steps(bot_username=bot_username, query=query)
#     task = Task(client_id=chat_id, plan=plan, objective=query)

#     text = "Plan:\n"
#     for i, step in enumerate(plan.steps, start=1):
#         text += f'{i}. {step.value}\n'

#     context.bot.send_message(chat_id=update.message.chat_id, text=text)

#     store_object(obj=task, name='task', community=bot_username)
#     confirm(chat_id, context)

# def confirm(chat_id, context):
#     keyboard = [[InlineKeyboardButton("Yes", callback_data='confirm_yes'),
#                  InlineKeyboardButton("No", callback_data='confirm_no')]]
#     reply_markup = InlineKeyboardMarkup(keyboard)
#     context.bot.send_message(chat_id=chat_id, text=f"Should I continue?", reply_markup=reply_markup)

# def handle_confirmation(update: Update, context):
#     query = update.callback_query
#     chat_id = query.message.chat_id
#     bot_username = context.bot.get_me().username
#     try:
#         query.answer(cache_time=10)
#     except Exception as e:
#         context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
#         return
#     if query.data == 'confirm_yes':
#         context.bot.send_message(chat_id=chat_id, text=f"Continuing plan...")
#         task = get_object(community=bot_username, obj_name='task')
#         if task is None:
#             context.bot.send_message(chat_id=chat_id, text=f"Task not found.")
#             return
#         if task.last_action is None:
#             continue_plan(chat_id, context)
#         else:
#             action = task.last_action
#             name_to_tool_map = {tool.name: tool for tool in get_all_tools(bot_username, include_experts=False)}
#             try:
#                 tool = name_to_tool_map[action.tool]
#             except KeyError:
#                 context.bot.send_message(chat_id=chat_id, text=f"Tool {action.tool} not found.")
#                 return
#             try:
#                 observation = tool.run(action.tool_input)
#             except Exception as e:
#                 context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
#                 return
#             context.bot.send_message(chat_id=chat_id, text=f"Observation: {observation}")
#             task.intermediate_steps.append((action, observation))
#             store_object(obj=task, name='task', community=bot_username)
#             continue_plan(chat_id, context)
#     else:
#         # if no, stop
#         context.bot.send_message(chat_id=chat_id, text=f"Stopping plan...")


# def continue_plan(chat_id, context):
#     bot_username = context.bot.get_me().username
#     task = get_object(community=bot_username, obj_name='task')
#     if task is None:
#         context.bot.send_message(chat_id=chat_id, text=f"Something went wrong continuing the plan, task not found.")
#         return

#     # check if current step out of bounds
#     if task.current_step_n >= len(task.plan.steps):
#         context.bot.send_message(chat_id=chat_id, text=f"Plan complete!")
#         return
#     context.bot.send_message(chat_id=chat_id, text=f'Step: {task.current_step_n + 1}')

#     # get next action
#     context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

#     task_str = str(task)
#     context.bot.send_message(chat_id=chat_id, text=f"This is the task input to the executer agent:\n{task_str}")
#     action = get_next_action(bot_username=bot_username,  task=task)

#     context.bot.send_message(chat_id=chat_id, text=f"Agent output:\n{action.log}")
#     if isinstance(action, AgentAction):
#         task.last_action = action
#         store_object(obj=task, name='task', community=bot_username)
#         confirm(chat_id, context)
#     elif isinstance(action, AgentFinish):
#         response = action.return_values['output']
#         task.step_container.add_step(task.plan.steps[task.current_step_n], StepResponse(response=response))
#         task.current_step_n += 1
#         task.intermediate_steps = []
#         context.bot.send_message(chat_id=chat_id, text=f"Final response for step {task.current_step_n}: {response}")
#         store_object(obj=task, name='task', community=bot_username)
#         continue_plan(chat_id, context)

# # Shows the user's current state
# def cmd_my_info(update, context):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     user_id = update.message.from_user.id

#     # get user info
#     user_dict = get_user(bot_username, user_id)
#     # if user dict is empty, send message
#     if not user_dict["strings"]:
#         context.bot.send_message(chat_id=chat_id, text="You haven't provided any info yet. Please tell me something about you first.", parse_mode=ParseMode.MARKDOWN,)
#         return
#     state_update_method = get_state_update_method(bot_username)
#     if state_update_method == "concatenate":
#         user = list(user_dict["strings"].values())
#     elif state_update_method == "summarize":
#         user = get_user_summary(bot_username, user_id)
#     else:
#         context.bot.send_message(chat_id=chat_id, text=f"State update method = {state_update_method} not recognized. Please contact the community's moderator.", parse_mode=ParseMode.MARKDOWN,)
#         return
#     # send user info
#     context.bot.send_message(chat_id=chat_id, text=f"Your info: \n {user}", parse_mode=ParseMode.MARKDOWN,)

# Stores user review of a match after the fact with the format '/review_match <username> <review text>' where username is a word that starts with '@' and review text is a string of text
# @send_typing_action
# def cmd_review_match(update, context, **kwargs):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     user_id = update.message.from_user.id
#     username = update.message.from_user.username

#     if kwargs.get('username'):
#         reviewed_username = kwargs.get('username')
#         review_text = kwargs.get('review')
#     else:
#         # get arguments
#         args = context.args
#         # check if there are enough arguments
#         if len(args) < 2:
#             context.bot.send_message(chat_id=chat_id, text="Not enough arguments. Please use the format: /review_match <username> <review text>", parse_mode=ParseMode.MARKDOWN,)
#             return
#         reviewed_username = args[0]
#         review_text = ' '.join(args[1:])
#     # strip '@' from reviewed user name
#     if reviewed_username[0] == '@':
#         reviewed_username = reviewed_username[1:]

#     # get user_id from username using name_to_id.json get_name_to_id_dict
#     name_to_id = get_name_to_id_dict(bot_username)
#     try:
#         reviewed_userid = name_to_id[reviewed_username]
#     except KeyError:
#         context.bot.send_message(chat_id=chat_id, text=f"Could not find user {reviewed_username}. Please check the username and try again.")
#         return
#     reviewed_userid = name_to_id[reviewed_username]
#     # check if reviewed user was in a match with the user
#     print(f"checking match between {username} and {reviewed_username} with user_id {user_id} and reviewed_userid {reviewed_userid}")
#     matched = check_match(bot_username, user_id, reviewed_userid)
#     if not matched:
#         context.bot.send_message(chat_id=chat_id, text="You can only review matches you have had.", parse_mode=ParseMode.MARKDOWN,)
#         return

#     # store review
#     store_review(bot_username, user_id, reviewed_userid, review_text)
#     # send message to user
#     context.bot.send_message(chat_id=chat_id, text=f"Your review of {reviewed_username} has been stored.", parse_mode=ParseMode.MARKDOWN,)


# def handle_accept_request(update, context):
#     query = update.callback_query
#     data = query.data
#     chat_id = query.message.chat_id
#     offerer_username = query.message.chat.username

#     action, requester_id_str, requester_name = data.split(":")
#     if action == "accept_request_yes":
#         # Inform the requester and ask if they want to accept the offer
#         requester_id = int(requester_id_str)
#         send_accept_offer_question(chat_id=requester_id, user_name=requester_name, offerer_id=query.from_user.id, offerer_name=offerer_username, context=context)
#         # tell user that he can review the interaction after the fact with the command '/review_match <username> <review text>'
#         message = f"You have accepted the request. You can review the interaction after the fact with the command '/review_match @{requester_name} <review text>'."
#         context.bot.send_message(chat_id=chat_id, text=message)
#     elif action == "accept_request_no":
#         # The offerer declined the request
#         context.bot.send_message(chat_id=chat_id, text="You have declined the request.")

#     # Answer the callback query
#     query.answer()

# def handle_accept_offer(update, context):
#     query = update.callback_query
#     data = query.data
#     chat_id = query.message.chat_id
#     requester_username = query.message.chat.username

#     print("requester_user_name: ", requester_username)
#     action, offerer_id_str, offerer_username = data.split(":")
#     if action == "accept_offer_yes":
#         # Both the offerer and requester have accepted
#         context.bot.send_message(chat_id=chat_id, text="You have accepted the offer.")
#         # Inform both the offerer and requester to create a group chat and add the bot
#         bot_username = context.bot.get_me().username
#         message = f"Please create a group chat with @{offerer_username} and add the bot @{bot_username} to the chat."
#         context.bot.send_message(chat_id=chat_id, text=message)
#         # store match
#         store_match(bot_username, user1=offerer_id_str, user2=str(chat_id))
#         # tell user that he can review the interaction after the fact with the command '/review_match <username> <review text>'
#         message = f"You can review the interaction after the fact with the command '/review_match @{offerer_username} <review text>'"
#         context.bot.send_message(chat_id=chat_id, text=message)

#     elif action == "accept_offer_no":
#         # The requester declined the offer
#         context.bot.send_message(chat_id=chat_id, text="You have declined the offer.")

#     # Answer the callback query
#     query.answer()


# def send_accept_match_question(chat_id, user_name, requester_id, requester_name, context):
#     # get requester info
#     bot_username = context.bot.username
#     state_update_method = get_state_update_method(bot_username)
#     print(f"bot_username = {bot_username}, requester_id = {requester_id}, requester_name = {requester_name}")
#     user_dict = get_user(bot_username, requester_id)
#     if state_update_method == "concatenate":
#         requester = list(user_dict["strings"].values())
#     elif state_update_method == "summarize":
#         requester = user_dict["summary"]
#     message = f"Hi {user_name}, there is someone you would probably enjoy interacting with. This is their summary: {requester}. Would you like to match?"
#     keyboard = [
#         [
#             InlineKeyboardButton("Yes", callback_data=f"accept_request_yes:{requester_id}:{requester_name}"),
#             InlineKeyboardButton("No", callback_data=f"accept_request_no:_:_"),
#         ]
#     ]
#     reply_markup = InlineKeyboardMarkup(keyboard)
#     context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)

# def send_accept_offer_question(chat_id, user_name, offerer_id, offerer_name, context):
#     bot_username = context.bot.username
#     state_update_method = get_state_update_method(bot_username)
#     user_dict = get_user(bot_username, offerer_id)
#     if state_update_method == "concatenate":
#         user = list(user_dict["strings"].values())
#     elif state_update_method == "summarize":
#         user = user_dict["summary"]
#     message = f"Hi {user_name}, there is someone you would probably enjoy interacting with. These is their summary: {user}. Would you like to match?"
#     keyboard = [
#         [
#             InlineKeyboardButton("Yes", callback_data=f"accept_offer_yes:{offerer_id}:{offerer_name}"),
#             InlineKeyboardButton("No", callback_data=f"accept_offer_no"),
#         ]
#     ]
#     reply_markup = InlineKeyboardMarkup(keyboard)
#     context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)

# TODO: Remove this function and use only send_accept_match_question
# def inform_user(matched_user_id, requester_id, requester_name, context):
#     bot_username = context.bot.get_me().username
#     matched_key = f"{bot_username}/{matched_user_id}.json"

#     # Fetch the matched object from S3
#     response = s3.get_object(Bucket=BUCKET, Key=matched_key)

#     # Extract the metadata
#     metadata = response.get('Metadata', {})
#     user_id = metadata.get('user_id')
#     user_name = metadata.get('user_name')

#     if user_id:
#         # Convert user_id back to integer
#         user_id = int(user_id)
#         # Send the message
#         send_accept_match_question(chat_id=user_id, user_name=user_name, requester_id=requester_id, requester_name=requester_name, context=context)
#     else:
#         print("User ID not found in metadata")

# Creates a csv table from all the available user information
# 1. Gets all the strings of all users
# 2. passes them to the extract_data function to extract the data
# 3. shows the data to the user
# @send_typing_action
# def cmd_show_database(update, context):
#     bot_username = context.bot.get_me().username
#     database_key = f"{bot_username}/database.csv"

#     database_obj = get_database_object(bot_username)
#     # get database metadata
#     metadata = database_obj.get("Metadata", {})
#     # check if database is updated
#     updated = metadata.get("updated", "false")

#     # if database is updated, send it to user
#     if updated == "true":
#         # write csv string to file
#         csv_string = database_obj["Body"].read().decode("utf-8")
#         csv_buffer = StringIO(csv_string)
#         csv_buffer.seek(0)
#         update.message.reply_text("The database has not changed:")
#         update.message.reply_document(document=csv_buffer, filename="database.csv")
#         return

#     # if database is not updated then create it again with the new information
#     csv_buffer = create_database(bot_username)

#     # add updated attribute to metadata of csv file
#     metadata = {"updated": "true"}

#     # upload csv file to s3
#     s3.put_object(Bucket=BUCKET, Key=database_key, Body=csv_buffer.getvalue(), Metadata=metadata)

#     # send csv file to user
#     update.message.reply_text("The database has been updated with new information:")
#     update.message.reply_document(document=csv_buffer, filename="database.csv")


# # Search for potential matches given a specific user
# @send_typing_action
# def cmd_match_me(update, context):
#     chat_id = update.message.chat_id
#     username = update.message.from_user.username
#     bot_username = context.bot.get_me().username
#     user_id = update.message.from_user.id
#     user_key = f"{bot_username}/{user_id}.json"

#     # send message to user
#     update.message.reply_text("Searching for potential matches...")

#     # get users
#     users_dict = get_users(bot_username)
#     current_user_dict = get_user(bot_username, user_id)

#     # if user dict is empty, send message
#     if not current_user_dict["strings"]:
#         context.bot.send_message(chat_id=chat_id, text="You haven't provided any info yet. Please tell me something about you first.", parse_mode=ParseMode.MARKDOWN,)
#         return

#     # depending on state update method, get users info
#     state_update_method = get_state_update_method(bot_username)
#     if state_update_method == "concatenate":
#         current_user = {'self report': list(current_user_dict["strings"].values()), 'reviews': list(current_user_dict["reviews"].values())}
#         users_dict = {user_id: {'self report' :list(user_dict["strings"].values()), 'reviews': list(user_dict['reviews'].values())} for user_id, user_dict in users_dict.items()}
#     elif state_update_method == "summarize":
#         current_user = get_user_summary(bot_username, user_id)
#         users_dict = {user_id: get_user_summary(bot_username, user_id) for user_id in users_dict.keys()}

#     match_prompt = get_match_prompt(bot_username)
#     input_variables_list = extract_variables_from_template(match_prompt)

#     if len(users_dict) == 1:
#         update.message.reply_text("No other users have registered yet :(")
#         return

#     # evaluate each potential match individually
#     for candidate_user_id, user in users_dict.items():
#         print(f"key: {candidate_user_id}, user: {user}, current user_id: {user_id}")
#         # skip current user to avoid self-match
#         if candidate_user_id == str(user_id):
#             continue

#         # evaluate potential match
#         input_variables_values = [current_user, user]
#         input_variables = dict(zip(input_variables_list, input_variables_values))
#         is_match, llm_opinion = evaluate_potential_match(match_prompt=match_prompt, input_variables=input_variables)
#         context.bot.send_message(chat_id=chat_id,
#             text=f'This is a potential match:\n user1: {current_user} \n user2: {user} \n llm opinion: {llm_opinion} \n match decision: {is_match}',
#             parse_mode=ParseMode.MARKDOWN,)

#         # inform other user incase of match
#         if is_match == True:
#             context.bot.send_message(chat_id=chat_id, text=f"there is one match", parse_mode=ParseMode.MARKDOWN,)
#             inform_user(matched_user_id=candidate_user_id, requester_id=user_id, requester_name=username, context=context)
#             return
#         elif is_match == None:
#             context.bot.send_message(chat_id=chat_id, text="There was an exception handling your message :(", parse_mode=ParseMode.MARKDOWN, )
#             break

# # Search for global matches
# @send_typing_action
# def cmd_match_all(update, context):
#     # extract relevant info
#     chat_id = update.message.chat_id
#     username = update.message.from_user.username
#     bot_username = context.bot.get_me().username

#     moderators = get_moderators(bot_username)
#     if username in moderators:

#         # send message to user
#         update.message.reply_text("Searching for global matches...")

#         # get users
#         users_dict = get_users(bot_username)

#         # if len(users) <= 1 inform and return
#         if len(users_dict) <= 1:
#             update.message.reply_text("At least two users are required to perform global matching")
#             return

#         # depending on state update method, get users info
#         state_update_method = get_state_update_method(bot_username)
#         if state_update_method == "concatenate":
#             # users_dict = {key: list(user_dict["strings"].values()) for key, user_dict in users_dict.items()}
#             users_dict = {user_id: {'self report' :list(user_dict["strings"].values()), 'reviews': list(user_dict['reviews'].values())} for user_id, user_dict in users_dict.items()}
#         elif state_update_method == "summarize":
#             # users_dict = {key: get_user_summary(bot_username, key) for key in users_dict.keys()}
#             users_dict = {user_id: get_user_summary(bot_username, user_id) for user_id in users_dict.keys()}

#         # get global match prompt
#         match_prompt = get_global_match_prompt(bot_username)

#         input_variables_list = extract_variables_from_template(match_prompt)
#         input_variables_values = list(users_dict.values())
#         input_variables = dict(zip(input_variables_list, input_variables_values))
#         matched_pairs, llm_opinion = get_matched_pairs(global_match_prompt=match_prompt, input_variables=input_variables)
#         if matched_pairs == -1:
#             update.message.reply_text("There was an exception parsing the output of the llm.")
#             update.message.reply_text(f"This is the output of the llm: {llm_opinion}")
#             return

#         # show all matches
#         context.bot.send_message(chat_id=chat_id, text=f"llm opinion: {llm_opinion}", parse_mode=ParseMode.MARKDOWN, )

#         users_keys = list(users_dict.keys())
#         # inform users of matches
#         for pair in matched_pairs:
#             try:
#                 key1 = users_keys[pair[0]]
#                 key2 = users_keys[pair[1]]
#             except IndexError:
#                 update.message.reply_text("the llm gave a wrong index for the matchings", parse_mode=ParseMode.MARKDOWN, )
#                 return

#             # get user names from keys and metadata
#             response = s3.get_object(Bucket=BUCKET, Key=key1)
#             metadata = response.get('Metadata', {})
#             user1_name = metadata.get('user_name')
#             user1_id = metadata.get('user_id')

#             response = s3.get_object(Bucket=BUCKET, Key=key2)
#             metadata = response.get('Metadata', {})
#             user2_name = metadata.get('user_name')
#             user2_id = metadata.get('user_id')

#             context.bot.send_message(chat_id=chat_id, text=f"Matched {user1_name} with {user2_name}", parse_mode=ParseMode.MARKDOWN, )
#             #TODO: modify inform user to user-level matching logic (instead of old entry level)
#             inform_user(matched_key=key1, requester_id=user2_id, requester_name=user2_name, context=context)
#     else:
#         context.bot.send_message(chat_id=chat_id, text=f"Sorry, only moderators can request global matches.", parse_mode=ParseMode.MARKDOWN,)

# # Stores users feedback
# def cmd_feedback(update, context, feedback=None):
#     bot_username = context.bot.get_me().username
#     username = update.message.from_user.username
#     chat_id = update.message.chat_id
#     feedback_key = f"{bot_username}/feedback.json"

#     if feedback is None:
#         feedback = ' '.join(context.args)
#     else:
#         feedback = feedback

#     if feedback:
#         try:
#             response = s3.get_object(Bucket=BUCKET, Key=feedback_key)
#             existing_content = json.loads(response['Body'].read().decode('utf-8'))
#         except s3.exceptions.NoSuchKey:
#             existing_content = {username: []}
#         existing_content[username].append(feedback)
#         response = s3.put_object(Bucket=BUCKET, Key=feedback_key, Body=json.dumps(existing_content))
#         context.bot.send_message(chat_id=chat_id, text="Your feedback was added successfully, Thank you!.", parse_mode=ParseMode.MARKDOWN,)
#     else:
#         context.bot.send_message(chat_id=chat_id, text="please add your feedback after the /feedback command. e.g. /feedback 'this is my feedback'", parse_mode=ParseMode.MARKDOWN,)

# # Resets the information of the user
# def cmd_reset(update, context):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     user_id = update.message.from_user.id
#     context.bot.send_message(chat_id=chat_id, text=f"Ok, I will reset all of your sentences in this community", parse_mode=ParseMode.MARKDOWN,)
#     s3.delete_object(Bucket=BUCKET, Key=f"{bot_username}/{user_id}.json")
#     context.bot.send_message(chat_id=chat_id, text=f"Done!", parse_mode=ParseMode.MARKDOWN,)

# # Resets all user information in the community
# def cmd_reset_all(update, context):
#     chat_id = update.message.chat_id
#     username = update.message.from_user.username
#     bot_username = context.bot.get_me().username
#     moderators_key = f"{bot_username}/moderators"

#     # get moderators
#     response = s3.get_object(Bucket=BUCKET, Key=moderators_key)
#     moderators = json.loads(response['Body'].read().decode('utf-8'))

#     if (username == 'MAIN_MOD_USERNAME' or username in moderators):
#         context.bot.send_message(chat_id=chat_id, text=f"Hey {username}. Ok, I will reset the messages of this community", parse_mode=ParseMode.MARKDOWN,)
#         objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{bot_username}/")
#         if 'Contents' in objects:
#             for obj in objects['Contents']:
#                 key = obj['Key']
#                 if key.split('/', 1)[1][0].isdigit():
#                     s3.delete_object(Bucket=BUCKET, Key=key)

#         # pending queries
#         pending_queries = {}
#         store_object(pending_queries, community=bot_username, name='pending_queries')

#         context.bot.send_message(chat_id=chat_id, text=f"Done!. All user information and pending queries have been reset", parse_mode=ParseMode.MARKDOWN,)


# # command to set the user prompt
# def cmd_set_user_prompt(update, context, prompt=None):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     prompts_key = f"{bot_username}/prompts.json"
#     username = update.message.from_user.username

#     if prompt is None:
#         user_prompt = ' '.join(context.args)
#     else:
#         user_prompt = prompt
#     moderators_key = f"{bot_username}/moderators"

#     # get moderators
#     response = s3.get_object(Bucket=BUCKET, Key=moderators_key)
#     moderators = json.loads(response['Body'].read().decode('utf-8'))

#     if (username == 'MAIN_MOD_USERNAME' or username in moderators):
#         if user_prompt:
#             try:
#                 response = s3.get_object(Bucket=BUCKET, Key=prompts_key)
#                 existing_content = json.loads(response['Body'].read().decode('utf-8'))
#             except s3.exceptions.NoSuchKey:
#                 existing_content = {}
#             existing_content["user"] = user_prompt
#             s3.put_object(Bucket=BUCKET, Key=prompts_key, Body=json.dumps(existing_content))
#             context.bot.send_message(chat_id=chat_id, text="user prompt set successfully", parse_mode=ParseMode.MARKDOWN,)
#         else:
#             context.bot.send_message(chat_id=chat_id, text="please give a valid prompt.", parse_mode=ParseMode.MARKDOWN,)
#     else:
#         context.bot.send_message(chat_id=chat_id, text="sorry, only moderators can use this command.", parse_mode=ParseMode.MARKDOWN,)

# # Sets the pairwise match prompt
# def cmd_set_match_prompt(update, context, prompt=None):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     prompts_key = f"{bot_username}/prompts.json"
#     username = update.message.from_user.username

#     if prompt is None:
#         match_prompt = ' '.join(context.args)
#     else:
#         match_prompt = prompt

#     moderators_key = f"{bot_username}/moderators"

#     # get moderators
#     response = s3.get_object(Bucket=BUCKET, Key=moderators_key)
#     moderators = json.loads(response['Body'].read().decode('utf-8'))

#     if (username == 'MAIN_MOD_USERNAME' or username in moderators):
#         if match_prompt:
#             try:
#                 response = s3.get_object(Bucket=BUCKET, Key=prompts_key)
#                 existing_content = json.loads(response['Body'].read().decode('utf-8'))
#             except s3.exceptions.NoSuchKey:
#                 existing_content = {}
#             existing_content["match"] = match_prompt
#             s3.put_object(Bucket=BUCKET, Key=prompts_key, Body=json.dumps(existing_content))
#             context.bot.send_message(chat_id=chat_id, text="match prompt set successfully", parse_mode=ParseMode.MARKDOWN,)
#         else:
#             context.bot.send_message(chat_id=chat_id, text="please give a valid prompt.", parse_mode=ParseMode.MARKDOWN,)
#     else:
#         context.bot.send_message(chat_id=chat_id, text="sorry, only moderators can use this command.", parse_mode=ParseMode.MARKDOWN,)

# # Sets global match prompt
# def cmd_set_global_match_prompt(update, context, prompt=None):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     prompts_key = f"{bot_username}/prompts.json"
#     username = update.message.from_user.username

#     if prompt is None:
#         global_match_prompt = ' '.join(context.args)
#     else:
#         global_match_prompt = prompt

#     moderators = get_moderators(bot_username)
#     if (username == 'MAIN_MOD_USERNAME' or username in moderators):
#         if global_match_prompt:
#             try:
#                 response = s3.get_object(Bucket=BUCKET, Key=prompts_key)
#                 existing_content = json.loads(response['Body'].read().decode('utf-8'))
#             except s3.exceptions.NoSuchKey:
#                 existing_content = {}
#             existing_content["global_match"] = global_match_prompt
#             s3.put_object(Bucket=BUCKET, Key=prompts_key, Body=json.dumps(existing_content))
#             context.bot.send_message(chat_id=chat_id, text="global match prompt set successfully", parse_mode=ParseMode.MARKDOWN,)
#         else:
#             context.bot.send_message(chat_id=chat_id, text="please give a valid prompt.", parse_mode=ParseMode.MARKDOWN,)
#     else:
#         context.bot.send_message(chat_id=chat_id, text="sorry, only moderators can use this command.", parse_mode=ParseMode.MARKDOWN,)

# # Sets the summary prompt
# def cmd_set_summary_prompt(update, context, prompt=None):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     prompts_key = f"{bot_username}/prompts.json"
#     username = update.message.from_user.username

#     if prompt is None:
#         summary_prompt = ' '.join(context.args)
#     else:
#         summary_prompt = prompt

#     moderators_key = f"{bot_username}/moderators"

#     # get moderators
#     response = s3.get_object(Bucket=BUCKET, Key=moderators_key)
#     moderators = json.loads(response['Body'].read().decode('utf-8'))

#     if (username == 'MAIN_MOD_USERNAME' or username in moderators):
#         if summary_prompt:
#             try:
#                 response = s3.get_object(Bucket=BUCKET, Key=prompts_key)
#                 existing_content = json.loads(response['Body'].read().decode('utf-8'))
#             except s3.exceptions.NoSuchKey:
#                 existing_content = {}
#             existing_content["summary"] = summary_prompt
#             s3.put_object(Bucket=BUCKET, Key=prompts_key, Body=json.dumps(existing_content))
#             context.bot.send_message(chat_id=chat_id, text="summary prompt set successfully", parse_mode=ParseMode.MARKDOWN,)
#         else:
#             context.bot.send_message(chat_id=chat_id, text="please give a valid prompt.", parse_mode=ParseMode.MARKDOWN,)
#     else:
#         context.bot.send_message(chat_id=chat_id, text="sorry, only moderators can use this command.", parse_mode=ParseMode.MARKDOWN,)

# # set query agent prompt
# def cmd_set_query_prompt(update, context, prompt=None):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     prompts_key = f"{bot_username}/prompts.json"
#     username = update.message.from_user.username

#     if prompt is None:
#         query_agent_prompt = ' '.join(context.args)
#     else:
#         query_agent_prompt = prompt

#     moderators_key = f"{bot_username}/moderators"

#     # get moderators
#     response = s3.get_object(Bucket=BUCKET, Key=moderators_key)
#     moderators = json.loads(response['Body'].read().decode('utf-8'))

#     if (username == 'MAIN_MOD_USERNAME' or username in moderators):
#         if query_agent_prompt:
#             try:
#                 response = s3.get_object(Bucket=BUCKET, Key=prompts_key)
#                 existing_content = json.loads(response['Body'].read().decode('utf-8'))
#             except s3.exceptions.NoSuchKey:
#                 existing_content = {}
#             existing_content["query"] = query_agent_prompt
#             s3.put_object(Bucket=BUCKET, Key=prompts_key, Body=json.dumps(existing_content))
#             context.bot.send_message(chat_id=chat_id, text="query agent prompt set successfully", parse_mode=ParseMode.MARKDOWN,)
#         else:
#             context.bot.send_message(chat_id=chat_id, text="please give a valid prompt.", parse_mode=ParseMode.MARKDOWN,)
#     else:
#         context.bot.send_message(chat_id=chat_id, text="sorry, only moderators can use this command.", parse_mode=ParseMode.MARKDOWN,)

# # Sets the state update method
# def cmd_set_state_update_method(update, context):
#     bot_username = context.bot.get_me().username
#     chat_id = update.message.chat_id
#     username = update.message.from_user.username
#     methods_key = f"{bot_username}/methods.json"
#     state_update_method = ' '.join(context.args)
#     moderators_key = f"{bot_username}/moderators"

#     # get moderators
#     response = s3.get_object(Bucket=BUCKET, Key=moderators_key)
#     moderators = json.loads(response['Body'].read().decode('utf-8'))

#     if (username == 'MAIN_MOD_USERNAME' or username in moderators):
#         # first get the existing methods
#         try:
#             response = s3.get_object(Bucket=BUCKET, Key=methods_key)
#             methods = json.loads(response['Body'].read().decode('utf-8'))
#         except s3.exceptions.NoSuchKey:
#             methods = {}

#         if state_update_method in {"concatenate", "summarize"}:
#             methods["state_update"] = state_update_method
#             s3.put_object(Bucket=BUCKET, Key=methods_key, Body=json.dumps(methods))
#             context.bot.send_message(chat_id=chat_id, text=f"state update method set to {state_update_method}", parse_mode=ParseMode.MARKDOWN,)
#         else:
#             context.bot.send_message(chat_id=chat_id, text=f"State update method can only be 'concatenate' or 'summarize'.", parse_mode=ParseMode.MARKDOWN,)
#     else:
#         context.bot.send_message(chat_id=chat_id, text="sorry, only moderators can use this command.", parse_mode=ParseMode.MARKDOWN,)
