from flask import Flask, request, jsonify, Response
import os
import json
import openai
from datetime import datetime as dt
import logging
import matplotlib.pyplot as plt
import os
import csv
import ast
import datetime
import json
import glob
import tiktoken


# init logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)


def token_counter(text, model):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(str(text))
    return len(tokens)


def text_chat_gpt(prompt, model):
    openai.api_key = os.getenv("PHRASE_SEED")
    answer = openai.ChatCompletion.create(
        model = model,
        messages=prompt
    )
    return answer


def load_default_config(user_id):
    conf_path = 'user_conf/'
    with open(conf_path+'config.json', 'r') as f:
        config = json.load(f)
    
    return config


def read_config(user_id_original):
    user_id = str(user_id_original)
    conf_path = 'user_conf/'
    # if user.json conf not in user_conf folder, create it
    # default config file: config.json
    if not os.path.exists(conf_path+user_id+'.json'):
        config = load_default_config(user_id)
    else:
        with open(conf_path+user_id+'.json', 'r') as f:
            config = json.load(f)
    logger.info('read_config: '+str(config))
    return config


def save_config(config, user_id_original):
    user_id = str(user_id_original)
    # user configuration
    conf_path = 'user_conf/'
    # save config
    with open(conf_path+user_id+'.json', 'w') as f:
        json.dump(config, f)


def authenticate(user_id):
    message = ''
    # User exist if config file exist
    conf_path = 'user_conf/'
    if os.path.exists(conf_path+str(user_id)+'.json'):        
        return True, message
    else:
        message = "Для доступа к сервису, пожалуйста, перешлите это сообщение @format37\nВаш ID: `"+str(user_id)+"`"
        return False, escape_characters(message)


def escape_characters(text):
    """ https://core.telegram.org/bots/api#formatting-options
    Please note:

    Any character with code between 1 and 126 inclusively can be escaped anywhere with a preceding '\' character, in which case it is treated as an ordinary character and not a part of the markup. This implies that '\' character usually must be escaped with a preceding '\' character.
    Inside pre and code entities, all '`' and '\' characters must be escaped with a preceding '\' character.
    Inside (...) part of inline link definition, all ')' and '\' must be escaped with a preceding '\' character.
    In all other places characters '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!' must be escaped with the preceding character '\'.
    In case of ambiguity between italic and underline entities __ is always greadily treated from left to right as beginning or end of underline entity, so instead of ___italic underline___ use ___italic underline_\r__, where \r is a character with code 13, which will be ignored.
    """
    # characters = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    characters = ['_', '*', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for character in characters:
        text = text.replace(character, '\\'+character)
    return text


def delete_conversation_files(user_id, chat_id, chat_type):
    if chat_type == 'group' or chat_type == 'supergroup':
        logger.info("delete group chat")
        # Create group id folder in the data path if not exist
        path = os.path.join("data", "groups", str(chat_id))
        # Get all files in folder
        list_of_files = glob.glob(path + "/*.json")
    else:
        logger.info("delete private chat")
        # Create user id folder in the data path if not exist
        path = os.path.join("data", "users", str(user_id))
        # Get all files in folder
        list_of_files = glob.glob(path + "/*.json")

    # Sort files by creation time ascending
    list_of_files.sort(key=os.path.getctime)
    # Delete all files except the last one
    # for file_path in list_of_files[:-1]:
    for file_path in list_of_files:
        logger.info("delete file: "+file_path)
        os.remove(file_path)


def read_latest_messages(user_id, chat_id, chat_type, chat_gpt_prompt_original, model):
    token_limit = 3000
    chat_gpt_prompt = []
    if chat_type == 'group' or chat_type == 'supergroup':
        logger.info("read group chat")
        # Create group id folder in the data path if not exist
        path = os.path.join("data", "groups", str(chat_id))
        # Get all files in folder
        list_of_files = glob.glob(path + "/*.json")
    else:
        logger.info("read private chat")
        # Create user id folder in the data path if not exist
        path = os.path.join("data", "users", str(user_id))
        # Get all files in folder
        list_of_files = glob.glob(path + "/*.json")

    # Sort files by creation time ascending
    list_of_files.sort(key=os.path.getctime, reverse=True)

    # Iterate over sorted files and append message to messages list
    limit_reached = False
    for file_name in list_of_files:
        logger.info("reading file: "+file_name)
        if limit_reached == False and token_counter(chat_gpt_prompt, model)<token_limit:            
            with open(file_name, "r") as f:
                data = json.load(f)
                if data["user_name"] == "assistant":
                    role = "assistant"
                    chat_gpt_prompt.append({"role": role, "content": data["message"]})
                else:
                    role = "user"
                    user_name = data["user_name"]
                    user_name = user_name if user_name else "Unknown"
                    chat_gpt_prompt.append({"role": role, "content": user_name +': '+ data["message"]})
        else:
            limit_reached = True
            logger.info("token limit reached. removing file: "+file_name)
            os.remove(file_name)

    # Sort chat_gpt_prompt reversed
    chat_gpt_prompt.reverse()
    # Now add all values of chat_gpt_prompt to chat_gpt_prompt_original
    for item in chat_gpt_prompt:
        chat_gpt_prompt_original.append(item)

    # logger.info("chat_gpt_prompt_original: "+str(chat_gpt_prompt_original))

    return chat_gpt_prompt_original


def save_message(user_id, user_name, chat_id, chat_type, message):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
    data = {
        "user_id": user_id,
        "user_name": user_name,
        "message": message,
    }
    if chat_type == 'group' or chat_type == 'supergroup':
        logger.info("group chat: "+str(chat_id))
        # Create group id folder in the data path if not exist
        group_path = os.path.join("data", "groups", str(chat_id))
        os.makedirs(group_path, exist_ok=True)
        # Each message saved as a new file with date in a filename
        file_name = f"{date_time}.json"
        with open(os.path.join(group_path, file_name), "w") as f:
            json.dump(data, f)
    else:
        logger.info("private chat: "+str(user_id))
        # Create user id folder in the data path if not exist
        user_path = os.path.join("data", "users", str(user_id))
        os.makedirs(user_path, exist_ok=True)
        # Each message saved as a new file with date in a filename
        file_name = f"{date_time}.json"
        with open(os.path.join(user_path, file_name), "w") as f:
            json.dump(data, f)


####################
# Call functions:
####################
@app.route("/test")
def call_test():
    return "get ok"


@app.route("/reset", methods=["POST"])
def call_reset():
    # print('call_reset_p')
    # logger.info('call_reset')
    """request_str = json.loads(str(await request.text()))
    data = json.loads(request_str)"""
    r = request.get_json()
    data = json.loads(r)
    user_id = str(data['user_id'])
    chat_id = str(data['chat_id'])
    chat_type = str(data['chat_type'])
    # logger.info(str(dt.now())+' '+'User: '+str(user_id)+' call_reset')
    
    authentication, message = authenticate(user_id)
    if not authentication:
        logger.info(str(dt.now())+' '+'User: '+str(user_id)+' not authenticated. message: '+str(message))
        # return web.Response(text=message, content_type="text/html")
        # return jsonify({'text': message})
        return jsonify({"result": message})
    
    # reset_prompt(user_id)
    delete_conversation_files(user_id, chat_id, chat_type)
    # return web.Response(text='Память очищена', content_type="text/html")
    """result = jsonify({'result': 'Память очищена'})
    logger.info(str(dt.now())+' '+'User: '+str(user_id)+' reset: '+str(result))
    return result"""
    return jsonify({"result": 'Память очищена'})


@app.route("/message", methods=["POST"])
def call_message():
    """request_str = json.loads(str(await request.text()))
    data = json.loads(request_str)"""
    r = request.get_json()
    r_dict = json.loads(r)
    user_id = r_dict["user_id"]
    user_name = r_dict["user_name"]
    chat_id = r_dict["chat_id"]
    chat_type = r_dict["chat_type"]
    message = r_dict["text"]

    # logger.info('call_message: '+str(message))
    # Define the default answer
    result = ""
    reaction = False
    if chat_type == 'group' or chat_type == 'supergroup':
        # Read config
        config = read_config(chat_id)        
        if message.startswith("/*") and len(message.strip()) > 2:
            reaction = True
            message = message[2:].strip()
    else:
        authentication, auth_message = authenticate(user_id)
        if not authentication:
            logger.info(str(dt.now())+' '+'User: '+str(user_id)+' not authenticated. message: '+str(message))
            # message = 'no'
            # return web.Response(text=message, content_type="text/html")
            return jsonify({"result": auth_message})
        reaction = True
        config = read_config(user_id)
            
    # Define the prompt
    chat_gpt_prompt = config['chat_gpt_prompt']
    # Save the original message
    # logger.info("saving message: {}".format(message))
    save_message(user_id, user_name, chat_id, chat_type, message)

    if reaction:
        chat_gpt_prompt = read_latest_messages(
            user_id, 
            chat_id, 
            chat_type, 
            chat_gpt_prompt,
            config['model']
            )
        # logger.info("chat_gpt_prompt: {}".format(chat_gpt_prompt))
        prompt_tokents = token_counter(chat_gpt_prompt, config['model'])
        logger.info("prompt_tokents: {}".format(prompt_tokents))
        openai_response = text_chat_gpt(chat_gpt_prompt, config['model'])
        result = openai_response['choices'][0]['message']['content']
        # logger.info("result: {}".format(result))
        # Save the answer
        save_message('assistant', 'assistant', chat_id, chat_type, result)
        # Replace 'assistant: ' with ''
        result = result.replace('assistant: ', '')

    return jsonify({"result": result})


@app.route("/user_add", methods=["POST"])
def call_user_add():
    """request_str = json.loads(str(await request.text()))
    data = json.loads(request_str)"""
    r = request.get_json()
    data = json.loads(r)
    user_id = str(data['user_id'])
    new_user_id = str(data['new_user_id'])
    new_user_name = str(data['new_user_name'])
    logger.info(str(dt.now())+' '+'User: '+str(user_id)+' call_user_add: '+str(new_user_id)+' '+str(new_user_name))
    # Check is user_id in admins.txt
    with open('data/admins.txt', 'r') as f:
        admins = f.read().splitlines()
    if user_id in admins:
        # add new user
        config = read_config(new_user_id)
        config['name'] = new_user_name
        save_config(config, new_user_id)
        content = 'Пользователь '+str(new_user_id)+' '+str(new_user_name)+' добавлен'
    else:
        content = 'Нет доступа'
    # return web.Response(text=content, content_type="text/html")
    return jsonify({"result": content})


@app.route("/financial_report", methods=["POST"])
def call_financial_report():
    """request_str = json.loads(str(await request.text()))
    data = json.loads(request_str)"""
    r = request.get_json()
    data = json.loads(r)
    user_id = str(data['user_id'])
    count_of_days = int(data['count_of_days'])

    logger.info(str(dt.now())+' '+'User: '+str(user_id)+' call_financial_report: '+str(count_of_days))
    # Check is user_id in admins.txt
    with open('data/admins.txt', 'r') as f:
        admins = f.read().splitlines()
    if user_id in admins:
        # Report
        logs_path = 'logs/'
        usernames_path = 'user_conf'
        # Create a dictionary to map user IDs to real names
        user_names = {}
        for filename in os.listdir(usernames_path):
            if filename=='config.json':
                continue
            with open(os.path.join(usernames_path, filename), 'r') as f:
                # print(os.path.join(usernames_path, filename))
                data = json.load(f)
                user_id = filename.split('.')[0]
                user_id = ''.join(filter(str.isdigit, user_id))
                user_names[user_id] = data['name']

        # Initialize empty dictionary for each user
        users_funds = {}

        # Calculate date N days ago
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=count_of_days)

        # Loop through all CSV files
        for filename in os.listdir(logs_path):
            
            if not '.csv' in filename:
                continue
                
            # Extract user ID from filename
            user_id = filename.split('.')[0]
            user_id = ''.join(filter(str.isdigit, user_id))
            # Get the real name of the user
            if user_id in user_names:
                user_name = user_names[user_id]
            else:
                user_name = user_id
            
            with open(os.path.join(logs_path, filename), newline='') as csvfile:
                
                reader = csv.reader(csvfile, delimiter=';')
                for row in reader:
                    # Parse timestamp and extract date
                    timestamp = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
                    date = timestamp.strftime('%Y-%m-%d')
                    
                    # Extract date from filename
                    # date_str = filename.split('.')[0]
                    # date = datetime.datetime.strptime(date_str, '%Y-%m-%d')

                    # Skip files that are older than N days
                    # if date < cutoff_date:
                    #     continue
                    if count_of_days > 0 and datetime.datetime.strptime(date, '%Y-%m-%d') < cutoff_date:
                        continue
                    
                    logger.info('row[2] value: '+str(row[2]))
                    logger.info('row[2] type: '+str(type(row[2])))
                    # Evaluate token count and calculate funds
                    try:
                        token_count = len(ast.literal_eval(row[2]))
                    except Exception as e:
                        logger.info('token calculation error. Exception: '+str(e))
                        token_count = 1000
                    funds = token_count * 0.02 / 1000
                    
                    # Append funds to corresponding user's list for this date
                    if date not in users_funds:
                        users_funds[date] = {}
                    if user_name not in users_funds[date]:
                        users_funds[date][user_name] = 0
                    users_funds[date][user_name] += funds

        # Prepare data for plotting
        user_names_sorted = sorted(list(set([user_name for date in users_funds for user_name in users_funds[date]])))
        dates = sorted(list(users_funds.keys()))
        funds_by_user_and_date = [[users_funds[date].get(user_name, 0) for date in dates] for user_name in user_names_sorted]

        # Plot stacked bar chart
        fig, ax = plt.subplots()
        ax.bar(dates, funds_by_user_and_date[0], label=user_names_sorted[0])
        bottom = funds_by_user_and_date[0]
        for i in range(1, len(user_names_sorted)):
            ax.bar(dates, funds_by_user_and_date[i], bottom=bottom, label=user_names_sorted[i])
            bottom = [bottom[j] + funds_by_user_and_date[i][j] for j in range(len(dates))]

        # Add chart labels and legend
        plt.xlabel('Date')
        plt.ylabel('Funds Spent (USD)')
        plt.title('Funds Spent by User and Date')
        plt.legend()

        # Save chart to image file
        plt.savefig('funds_spent.png')
        # content = 'Отчет сформирован'
        
        # Return image file
        with open('funds_spent.png', 'rb') as f:
            content = f.read()
        # return web.Response(body=content, content_type='image/png')
        return jsonify({"result": content})
    
    else:
        content = 'Нет доступа'
    # return web.Response(text=content, content_type="text/html")
    return jsonify({"result": content})


if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        debug=False,
        port=int(os.environ.get("PORT", os.environ.get('PORT', '')))
        )