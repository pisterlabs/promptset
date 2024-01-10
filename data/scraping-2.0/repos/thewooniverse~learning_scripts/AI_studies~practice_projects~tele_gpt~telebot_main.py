

### importing modules ###
import os
import shutil
from dotenv import load_dotenv

## Telegram + formatting libraries
import telebot

## config support
import json
import requests
import re

## OpenAI libraries
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from msg_templates import system_template, start_template

# Vector Support
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain



### importing API Keys and setting constants ###
load_dotenv()
TG_API_KEY = os.getenv('TG_API_KEY')
OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY')
# print(OPEN_API_KEY) # prints 'test ABC' correctly
# print(TG_API_KEY)


bot = telebot.TeleBot(TG_API_KEY)
# default_embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=OPENAI_API_KEY)
# default_agent = ChatOpenAI(model='gpt-4-0613', openai_api_key=OPENAI_API_KEY, model_name="gpt-4-0613", temperature=0.7)



### Defining Telegram Bot functions and decorators ###

## Helper functions ##
def load_config(group_id):
    chats_path = os.path.join(os.path.dirname(__file__), f"chats") # tele_gpt/chats/
    group_path = f'{chats_path}{os.path.sep}{group_id}' # ex. tele_gpt/chats/2849193
    config_path = f"{group_path}{os.path.sep}config.json" # ex. tele_gpt/chats/2849193/config.json
    if not os.path.exists(config_path):
        # check if the group path itself exists, if it doesn't create it.
        if not os.path.exists(group_path):
            os.mkdir(group_path)
      
      # then open the default config, and write its content (default settings) into the config_path file
        with open(f"{os.path.dirname(__file__)}{os.path.sep}default_config.json", 'r') as rf:
            default_config = json.load(rf)
            # print(default_config)
            wf = open(config_path, 'w')
            json.dump(default_config, wf)
            wf.close()

   # load and return the configuration
    with open(config_path, 'r') as rf:
        config_dict = json.load(rf)
        return config_dict


def set_config(group_id, new_config):
    """
    Returns - None
    --
    Params
    group_id == chatid
    new_config - a dict config that has the same structure but with new settings; saves the newly passed config into the correct path for the chat.
    --
    """

    chats_path = os.path.join(os.path.dirname(__file__), f"chats")
    group_path = f'{chats_path}{os.path.sep}{group_id}'
    config_path = f"{group_path}{os.path.sep}config.json"

    with open(config_path, 'w') as wf:
        json.dump(new_config, wf)


def check_can_delete(group_id):
    """
    Returns - True or False based on whether the bot can delete messages in a given group
    """
    bot_token = TG_API_KEY
    chat_id = group_id # Replace with your group chat ID
    bot_user_id = bot.get_me().id # get the bot's own user ID
    url = f"https://api.telegram.org/bot{bot_token}/getChatMember?chat_id={chat_id}&user_id={bot_user_id}"
    response = requests.get(url).json()
    if response["ok"]:
        status = response["result"]["status"]
        can_delete_messages = response["result"].get("can_delete_messages", False)

        if status == "administrator" and can_delete_messages:
            # print("bot can delete messages")
            return True
        else:
            # print("bot cannot delete messages")
            return False
    else:
        print("Error in fetching chat member details.")


def construct_config_string(config):
    """
    Returns - A formatted Config string with markdown
    --
    config - python dict unpacked from loaded json configuration file
    """
    base_string = "Current configurations:"
    key_value_strings = "-----------------------"
    excluded_config_keys = ['group_id', 'OPENAI_API_KEY']
    for key,value in config.items():
        if key not in excluded_config_keys:
            key_value_string = f"*{key}*: {value}"
            key_value_strings += "\n"+key_value_string
        else:
            key_string = f"*{key}*: XXXXX"
            key_value_strings += "\n"+key_string
    
    complete_string = f"{base_string}\n{key_value_strings}"
    return complete_string


def is_valid_temp(temp):
    """
    Return - True / False
    --
    Returns true or false based on whether temp is between 0 and 1
    """
    return 0 <= temp <= 1

def is_decimal_num(num_str):

    pattern = r"^\d*\.?\d+$"
    return bool(re.match(pattern, num_str))











## settings and configuration comamands ##
# Defining basic /start and /help commands
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f"{start_template}", parse_mode='HTML')
    # all the configurations / guide tutorial strings here

@bot.message_handler(commands=['get_config'])
def get_config(message):
    config = load_config(message.chat.id) # message.chat.id == group_id
    response = construct_config_string(config)
    bot.reply_to(message, response, parse_mode="Markdown")

@bot.message_handler(commands=['set_apikey'])
def set_api_key(message):
    api_key_entered = message.text.split()[1]
    config = load_config(message.chat.id)
    config['OPENAI_API_KEY'] = api_key_entered
    set_config(message.chat.id, config)
    
    if check_can_delete(message.chat.id):
        # Attempt to delete the message
        bot.reply_to(message, "API Key has been saved, deleting message.")
        bot.delete_message(message.chat.id, message.message_id)
    else:
        bot.reply_to(message, "API Key was saved, however I do not have permission to delete this message. Please delete this message and set me as an admin.")

@bot.message_handler(commands=['set_model'])
def set_model(message):
    config = load_config(message.chat.id)
    accepted_models = ["gpt-3.5-turbo", "gpt-4-0613", "gpt-4", 'gpt-3.5-turbo-0613']
    model_selection = " ".join(message.text.split()[1:])

    # check if the selected model is in accepted models
    if model_selection.lower() in accepted_models:
        config['model_name'] = model_selection.lower()
        set_config(message.chat.id, config)
        bot.reply_to(message, f"Model is changed to {model_selection.lower()}")
    else:
        bot.reply_to(message, f"[{model_selection.lower()}] is not a valid model, please select from the following: {accepted_models}")

@bot.message_handler(commands=['set_temp'])
def set_temp(message):
    config = load_config(message.chat.id)
    passed_temp = " ".join(message.text.split()[1:])

    # check that the 
    if passed_temp.isnumeric() or is_decimal_num(passed_temp):
        passed_temp = float(passed_temp)
        if is_valid_temp(passed_temp):
            config['temperature'] = passed_temp
            set_config(message.chat.id, config)
            bot.reply_to(message, f"Model Temperature changed to: {passed_temp}")
        else:
            bot.reply_to(message, f"{passed_temp} is not a valid temperature, please pass a number between 0 and 1 inclusive. E.g. 0.56")
    else:
        bot.reply_to(message, f"{passed_temp} is not a valid temperature, please pass a number between 0 and 1 inclusive. E.g. 0.56")

@bot.message_handler(commands=['set_system_message'])
def set_system_message(message):
    config = load_config(message.chat.id)
    new_system_message = " ".join(message.text.split()[1:])
    config['additional_system_message'] = new_system_message
    set_config(message.chat.id, config)
    bot.reply_to(message, f"System Message has been set as below: \n{new_system_message}")




# define logging commands
@bot.message_handler(commands=['logging_on'])
def persistence_off(message):
    config = load_config(message.chat.id)
    config['logging'] = "True"
    set_config(message.chat.id, config)
    bot.reply_to(message, f"Chat logging has been turned on, your conversations are being stored persistently.")

@bot.message_handler(commands=['logging_off'])
def persistence_off(message):
    config = load_config(message.chat.id)
    config['logging'] = "False"
    set_config(message.chat.id, config)
    bot.reply_to(message, f"Chat logging has been turned off, your conversations are no longer being stored persistently.")

# define context awareness commands
@bot.message_handler(commands=['context_on'])
def persistence_off(message):
    config = load_config(message.chat.id)
    config['context'] = "True"
    set_config(message.chat.id, config)
    bot.reply_to(message, f"Context awareness has been turned on, Agent will now search chat history for relevant context.")

@bot.message_handler(commands=['context_off'])
def persistence_off(message):
    config = load_config(message.chat.id)
    config['context'] = "False"
    set_config(message.chat.id, config)
    bot.reply_to(message, f"Context awareness has been turned off, chat history is no longer being searched for relevant context.")


@bot.message_handler(commands=['clear_history'])
def clear_history(message):
    chats_path = os.path.join(os.path.dirname(__file__), f"chats")
    group_path = f'{chats_path}{os.path.sep}{message.chat.id}'
    chroma_path = os.path.join(group_path, "chroma_db")
    try:
        shutil.rmtree(chroma_path)
        # print(f"The directory {chroma_path} has been removed")
        bot.reply_to(message, "Your chat history has been cleared - all context has been removed!")
    except OSError as e:
        print(f"Error: {e.strerror}")
        bot.reply_to(message, "Your chat history is already.")
    





## chatting commands and helper functions ##
# Defining the query command handler
@bot.message_handler(commands=['chat'])
def respond_chat(message):
    # load the config and construct the agent and embeddings
    config = load_config(message.chat.id)
    agent = ChatOpenAI(model=config['model_name'], openai_api_key=config['OPENAI_API_KEY'], model_name=config['model_name'], temperature=config['temperature'])
    embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=config['OPENAI_API_KEY'])
    chats_path = os.path.join(os.path.dirname(__file__), f"chats")
    group_path = f'{chats_path}{os.path.sep}{message.chat.id}'
    chroma_path = os.path.join(group_path, "chroma_db")
    

    # split the query:
    query = " ".join(message.text.split()[1:])

    # get the context / retrieved summary of documents based on configuration
    if config['context'] == "True":
        retrieved_docs = get_relevant_documents(query, chroma_path, embeddings)
        context = summarize_docs(query, chroma_path, retrieved_docs, embeddings, agent)
    else:
        context = ""

    # construct the chat history based on the context and configurations
    context_aware_chat_history = construct_chat_history(query, config['additional_system_message'], context)

    # construct the chat agent and query it;
    response = chat_agent(context_aware_chat_history, agent)
    if not response:
        bot.reply_to(message, "API Key is invalid, please /set_apikey to set your OpenAI API Key again")
    else:
        # save the log if logging = True
        if config['logging'] == "True":
            new_log = create_log(query, response)
            save_log(new_log, chroma_path, embeddings)

        bot.reply_to(message, response.content, parse_mode='Markdown')



### Defining ChatGPT query related functionality ###
def chat_agent(chat_history, agent):
    """
    Creates a chat model based on the configurations passed to it by unpacking it;
    """
    
    try:
        response = agent(chat_history)
        return response
    except openai.AuthenticationError:
        return False
    

def construct_chat_history(query, addl_system_message, context=""):
    """
    Returns: chat_history object that contains SystemMessage, AIMessage and HumanMessage
    query - a plain string that was asked to the agent
    addl_system_message - Addtional System instructions for the agent, for example - "You are an expert travel guide..."
    context - a plain string containing the summarized retrieved context from the ChromaDB based on the query; default is an empty string.
    """
    system_message = system_template + context
    
    chat_history = [
        SystemMessage(content=system_message),
        SystemMessage(content=f"\n{addl_system_message}\n"),
        HumanMessage(content=query)
    ]
    return chat_history


## chat history and RAG commands and helper functions ##
# define creating chat logs and history
# create log script
def create_log(query, response):
    """
    Takes the query and response, and combine it into a log entry.
    """
    log_entry = f"""

HUMAN QUERY:
{query}
--------
AI RESPONSE:
{response}
"""
    return log_entry

# define persistently saving chat logs
def save_log(log_entry, chroma_path, embeddings):
    """
    Takes the log entry and saves it to the persistent ChromaDB designated;
    """
    # get the chroma path and open the persistent library to that destination when it is called;
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    embedded_document = embeddings.embed_documents(log_entry)
    document = Document(page_content=log_entry, embedding=embedded_document)
    vectorstore.add_documents([document])
    vectorstore.persist()

# define functions for context retrieval and summarizing
def get_relevant_documents(query, chroma_path, embeddings, k=5):
    """
    Returns: matched documents from the database to the query.
    query - plaintext str query
    chroma_path - PATH towards the database to search against.
    k - number of documents to retrieve, default = 3

    default search algorithm is MMR;
    
    """
    # search algorithms
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retrieved_docs = vectorstore.max_marginal_relevance_search(query=query, k=k)
    return retrieved_docs

def summarize_docs(query, chroma_path, retrieved_docs, embeddings, agent):
    chain = load_summarize_chain(agent, chain_type="stuff")
    retrieved_docs = get_relevant_documents(query, chroma_path, embeddings)
    context = chain.run(retrieved_docs) # context now contains the summarized string of context;
    return context















### Starting the bot ###
bot.polling()





