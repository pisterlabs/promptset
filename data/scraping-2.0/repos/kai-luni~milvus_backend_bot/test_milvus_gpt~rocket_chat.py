import json
import os
import time
from datetime import datetime

import openai
from rocketchat_API.rocketchat import RocketChat

from chat_utils import ask

def load_config(filename='config.json'):
    """
    Loads configuration details from a JSON file.

    :param filename: The path to the configuration file
    :return: A dictionary containing the configuration details
    """
    with open(filename) as config_file:
        return json.load(config_file)

def initialize_openai(config_details):
    """
    Initializes OpenAI with the given configuration details.

    :param config_details: A dictionary containing the OpenAI configuration details
    """
    openai.api_type = "azure"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = config_details['OPENAI_API_BASE']
    openai.api_version = config_details['OPENAI_API_VERSION']

def get_last_responded_timestamp(timestamp_file):
    """
    Retrieves the timestamp of the last responded message from a file.

    :param timestamp_file: The path to the file containing the timestamp
    :return: A datetime object representing the timestamp or None if not found
    """
    try:
        with open(timestamp_file, 'r') as f:
            return datetime.fromisoformat(f.read().strip().replace('Z', '+00:00'))
    except (FileNotFoundError, ValueError):
        return None

def respond_to_mention(rocket, server_ip: str, channel='GENERAL', timestamp_file='timestamp.txt'):
    """
    Responds to the latest mention in the given channel that is newer than the last responded timestamp.

    :param rocket: The RocketChat object
    :param channel: The channel to monitor
    :param timestamp_file: The file to store the timestamp of the last responded message
    """
    last_responded_timestamp = get_last_responded_timestamp(timestamp_file)
    history = rocket.channels_history(channel, count=10).json()

    if history["success"] == False:
        print(f"There was a problem: {history['error']}")
        return

    for message in reversed(history['messages']):
        message_timestamp = datetime.fromisoformat(message['_updatedAt'].replace('Z', '+00:00'))
        if 'mentions' in message and any(mention['username'] == 'PhatGpt' for mention in message['mentions']):
            if last_responded_timestamp is None or message_timestamp > last_responded_timestamp:
                if message['u']['username'] == "PhatGpt":
                    continue
                answer = ask(message['msg'], os.environ['BEARER_TOKEN'], server_ip)
                response = f"@{message['u']['username']} {answer}"
                rocket.chat_post_message(response, channel=channel)

                with open(timestamp_file, 'w') as f:
                    f.write(message['_updatedAt'].replace('+00:00', 'Z'))

                break

def main():
    config_details = load_config()
    initialize_openai(config_details)
    rocket = RocketChat('PhatGpt', 'phatgpt', server_url=f'http://{config_details["SERVER_IP"]}:3000')

    while True:
        respond_to_mention(rocket, config_details["SERVER_IP"])
        time.sleep(10)

if __name__ == '__main__':
    main()
