import json
import random
import openai
from flask_sqlalchemy import SQLAlchemy
from flask import jsonify
from datetime import datetime, timezone
from .models import Poll
from .config import BaseConfig

openai.api_key = BaseConfig.CHAT_GPT_KEY

db = SQLAlchemy()

globalSingleMessage = [
    {
        "role": "system",
        "content": "I have a YouTube Channel, where I talk about:\n[\"saving money\", \"making money\",  \"banking\", \"global trade\", \"economics\", \"consumer demand\", \"stocks\", \"stock market\", \"ETFs\", \"Index Funds\", \"productivity improvements\", \"entrepreneurship\", \"building wealth\", \"starting a business\", \"financial success\"]\n\nBasically, anything related to finance, the economy, and anything that could influence the ability of a person to make money, or be of interest to a person who wants to build wealth. \n\nI would like to create some polls for Twitter, and I believe that there could be thousands of possible polls that we could create based on the topic criteria. A poll on Twitter can have up to 4 options, and each option should be less than 26 characters. \n\n> When I ask you to generate a poll, you will randomly choose one of the topics that I described above\n> You will generate a response in JSON format, with a single poll name, and an array of 4 options.\n> The poll name should be in the form of a question.\n> You will also pass the chosen topic to the \"topic\" attribute of the object\n> THIS IS IMPORTANT: Please make sure that the option value is less than 26 characters. There can be no exceptions.\n> Save this to context:\nthe appropriate response should be in the following format:\n[{\n  \"poll_name\": \"value\",\n  \"topic\": \"value\",\n  \"options\": [\n    {\"option\": \"value\"},\n]}]\n> Do not deviate from this format.\n> Do not duplicate any polls that you have previously generated. All polls generated should be unique."
    },
    {
        "role": "user",
        "content": "Generate a poll"
    },
]

globalMessages = [
    {
        "role": "system",
        "content": "I have a YouTube Channel, where I talk about:\n[\"saving money\", \"making money\",  \"banking\", \"global trade\", \"economics\", \"consumer demand\", \"stocks\", \"stock market\", \"ETFs\", \"Index Funds\", \"productivity improvements\", \"entrepreneurship\", \"building wealth\", \"starting a business\", \"financial success\"]\n\nBasically, anything related to finance, the economy, and anything that could influence the ability of a person to make money, or be of interest to a person who wants to build wealth. \n\nI would like to create some polls for LinkedIn, and I believe that there could be thousands of possible polls that we could create based on the topic criteria. A poll on LinkedIn can have up to 4 options, and each option should be less than 30 characters. \n\n> When I ask you to generate a poll, you will randomly choose one of the topics that I described above\n> You will generate a response in JSON format, with a single poll name, and an array of 4 options.\n> The poll name should be in the form of a question.\n> You will also pass the chosen topic to the \"topic\" attribute of the object\n> THIS IS IMPORTANT: Please make sure that the option value is less than 30 characters. There can be no exceptions.\n> Save this to context:\nthe appropriate response should be in the following format:\n[{\n  \"poll_name\": \"value\",\n  \"topic\": \"value\",\n  \"options\": [\n    {\"option\": \"value\"},\n]}]\n> Do not deviate from this format.\n> Do not duplicate any polls that you have previously generated. All polls generated should be unique."
    },
    {
        "role": "user",
        "content": "Generate 10 polls in an array"
    },
]

globalTweetterMessages = [
    {
        "role": "system",
        "content": "I have a YouTube Channel, where I talk about:\n[\"saving money\", \"making money\",  \"banking\", \"global trade\", \"economics\", \"consumer demand\", \"stocks\", \"stock market\", \"ETFs\", \"Index Funds\", \"productivity improvements\", \"entrepreneurship\", \"building wealth\", \"starting a business\", \"financial success\"]\n\nBasically, anything related to finance, the economy, and anything that could influence the ability of a person to make money, or be of interest to a person who wants to build wealth. \n\nI would like to create some polls for Twitter, and I believe that there could be thousands of possible polls that we could create based on the topic criteria. A poll on Twitter can have up to 4 options, and each option should be less than 30 characters. \n\n> When I ask you to generate a poll, you will randomly choose one of the topics that I described above\n> You will generate a response in JSON format, with a single poll name, and an array of 4 options.\n> The poll name should be in the form of a question.\n> You will also pass the chosen topic to the \"topic\" attribute of the object\n> THIS IS IMPORTANT: Please make sure that the option value is less than 26 characters. There can be no exceptions.\n> Save this to context:\nthe appropriate response should be in the following format:\n[{\n  \"poll_name\": \"value\",\n  \"topic\": \"value\",\n  \"options\": [\n    {\"option\": \"value\"},\n]}]\n> Do not deviate from this format.\n> Do not duplicate any polls that you have previously generated. All polls generated should be unique."
    },
    {
        "role": "user",
        "content": "Generate 2 polls in an array"
    },
]


def get_chat_messages(network):
    msgs = globalMessages if network == "linkedin" else globalTweetterMessages
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msgs,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message = response["choices"][0]["message"]
    globalMessages.append(message)
    content = response["choices"][0]["message"]["content"]
    content_dict = json.loads(content)
    return content_dict


def get_chat_message():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=globalMessages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message = response["choices"][0]["message"]
    globalMessages.append(message)
    content = response["choices"][0]["message"]["content"]
    content_dict = json.loads(content)
    return content_dict


def get_options(message, message_length):
    if message and len(message) > 0:
        options_list = message.get('options').split(",")
        options = [{"option": value} for value in options_list]
        print("options:", options)
        options_arr = []
        hasLongLength = 0
        for item in options:
            print("item:", item)
            length = len(item['option'])
            if (length > message_length):
                hasLongLength += 1
            options_arr.append(item['option'])
        return options_arr, hasLongLength
    else:
        return None, hasLongLength


def get_saving_options(message, message_length):
    if message and len(message) > 0:
        options = message.get('options')
        print("options:", options)
        options_arr = []
        hasLongLength = 0
        for item in options:
            print("item:", item)
            length = len(item['option'])
            if (length > message_length):
                hasLongLength += 1
            options_arr.append(item['option'])
        return options_arr, hasLongLength
    else:
        return None, hasLongLength


def filter_options(options):
    new_options = []
    for option in options:
        if len(option) <= 25:
            new_options.append(option)
    return new_options


def save_poll_to_db(network):
    messages = get_chat_messages(network)
    char_limit = 30 if network == "linkedin" else 26
    for message in messages:
        poll_name = message.get('poll_name')
        topic = message.get('topic')
        options, hasLongLength = get_saving_options(message, char_limit)
        if hasLongLength == 1:
            options = filter_options(options)
            add_poll({
                "title": poll_name,
                "topic": topic,
                "category": network,
                "options": options,
            })
        elif hasLongLength >= 2:
            print("Do nothing!")
            continue
        else:
            print(f"Add to db - {poll_name}")
            add_poll({
                "title": poll_name,
                "topic": topic,
                "category": network,
                "options": options,
            })

# Function to add a new poll entry


def create_poll(title, topic, category, options):
    try:
        # Convert options list to a string
        options_str = ",".join(options)
        # options_str = json.dumps(options)

        # Create a new Poll instance
        new_poll = Poll(title=title, topic=topic,
                        category=category, options=options_str)

        # Add the new poll to the database
        db.session.add(new_poll)
        db.session.commit()

        return jsonify({"message": "Poll created successfully!"}), 201
    except Exception as e:
        # Handle exceptions appropriately (e.g., log the error)
        return jsonify({"error": str(e)}), 500

# Propose a poll


def add_poll(data):
    title = data.get('title')
    topic = data.get('topic')
    category = data.get('category')
    options = data.get('options', [])

    if title and topic and category and options:
        return create_poll(title, topic, category, options)
    else:
        return jsonify({"error": "Incomplete data provided"}), 400


def get_poll_data(network):
    try:
        # Fetch all data from the 'poll' table
        poll_data = Poll.query.filter_by(category=network).all()
        # Convert the data to a list of dictionaries
        data_list = [{"id": poll.id, "title": poll.title, "topic": poll.topic,
                      "category": poll.category, "options": poll.options, "created_at": poll.created_at} for poll in poll_data]

        return data_list

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Random pick a poll < 7 days


def pick_poll_more_than_7_days(polls):
    # Get the current date and time
    current_date = datetime.utcnow().replace(tzinfo=timezone.utc)
    # Filter items less than 7 days old
    filtered_items = [
        item for item in polls if (
            current_date - item['created_at']
        ).days > 7
    ]
    return random.choice(filtered_items)

#  Delete a poll


def delete_posted_poll(poll):
    poll_id = poll.get("id")
    poll_to_delete = db.session.query(Poll).get(poll_id)
    db.session.delete(poll_to_delete)
    db.session.commit()
    return 'Deleted poll.'
