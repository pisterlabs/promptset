import os
import time
import logging
import requests
import openai
from datetime import datetime
from dateutil.parser import parse

from chat_utils import ask

def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Environment variable {var_name} is not set or is empty.")
    return value

# Global Constants
TEAMS_TENANT_ID = get_env_variable("TEAMS_TENANT_ID")
TEAMS_CLIENT_ID = get_env_variable("TEAMS_CLIENT_ID")
TEAMS_TENANT_ACCESS_TOKEN = get_env_variable("TEAMS_TENANT_ACCESS_TOKEN")
BEARER_TOKEN = get_env_variable('BEARER_TOKEN')
SERVER_IP = get_env_variable("SERVER_IP")
SCOPES = 'https://graph.microsoft.com/.default'
TOKEN_FILE_PATH = "teams_access_token.txt"



def initialize_openai():
    """
    Initialize the OpenAI settings using environment variables.

    This function configures the OpenAI SDK to use the "azure" type and sets up other necessary configurations
    using the environment variables. Make sure the required environment variables are set before calling this function.
    """
    openai.api_type = "azure"
    openai.api_key = get_env_variable("OPENAI_API_KEY")
    openai.api_base = get_env_variable('OPENAI_API_BASE')
    openai.api_version = get_env_variable('OPENAI_API_VERSION')

def request_device_code(tenant_id, client_id, scopes):
    """
    Request a device code for Microsoft's OAuth2.0 device flow.

    Args:
        tenant_id (str): The ID of the Azure AD tenant.
        client_id (str): The application's client ID.
        scopes (str): The space-separated list of scopes for which the token is requested.

    Returns:
        dict: A dictionary containing details about the device code.
    """
    device_code_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/devicecode"
    payload = {'client_id': client_id, 'scope': scopes}
    response = requests.post(device_code_url, data=payload)
    device_code_data = response.json()

    # Check if necessary keys are present in the response
    if 'verification_uri' in device_code_data and 'user_code' in device_code_data:
        logging.info(f"Please visit {device_code_data['verification_uri']} and enter the code: {device_code_data['user_code']}")
    else:
        logging.info("Failed to retrieve device code details. Please check the provided tenant ID, client ID, and scopes.")
        logging.info(f"tid {tenant_id}, clId {client_id},  scopes {scopes}")
        logging.info(f"Full response: {device_code_data}")

    return device_code_data


def poll_for_token(tenant_id, client_id, device_code_data):
    """
    Poll Microsoft's OAuth endpoint for an access token.

    This function keeps polling the token endpoint until an access token is received or an error occurs.

    Args:
        tenant_id (str): The ID of the Azure AD tenant.
        client_id (str): The application's client ID.
        device_code_data (dict): The device code data received from the request_device_code function.

    Returns:
        str: The received access token.
    """
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    logging.info("start waiting for device login")
    while True:
        payload = {
            'client_id': client_id,
            'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
            'device_code': device_code_data['device_code']
        }
        response = requests.post(token_url, data=payload)
        token_data = response.json()
        if 'access_token' in token_data:
            logging.info("finish waiting for device login")
            return token_data['access_token']
        time.sleep(device_code_data['interval'])  # Wait before polling again
    

def get_chats(access_token):
    """
    Retrieve the chats for the authenticated user.

    If the access token is invalid (e.g., expired), this function will re-run the process to get a new access token.

    Args:
        access_token (str): The access token to authenticate the request.

    Returns:
        list: A list containing the chats for the authenticated user.
    """
    url = "https://graph.microsoft.com/v1.0/me/chats"
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    
    # If forbidden, refresh the token and save it to the file
    # if response.status_code == 401:
    #     access_token = refresh_teams_access_token(tenant_id, client_id, scopes)
    #     headers = {'Authorization': f'Bearer {access_token}'}
    #     response = requests.get(url, headers=headers)
    
    response.raise_for_status()
    return response.json()['value']


def get_last_timestamp():
    """
    Retrieve the last timestamp from a file.

    This function reads the timestamp from a file named 'last_timestamp.txt'. 
    If the file does not exist, it returns None.

    Returns:
        datetime.datetime: The parsed datetime object from the file or None if the file does not exist.
    """
    try:
        with open('last_timestamp.txt', 'r') as file:
            return parse(file.read().strip())
    except FileNotFoundError:
        return None

def get_messages(access_token, chat_id):
    """
    Fetch the first 3 messages from a specific chat.

    Args:
        access_token (str): The access token to authenticate the request.
        chat_id (str): The ID of the chat from which messages need to be fetched.

    Returns:
        list: A list containing the first 3 messages from the specified chat.
    """
    url = f"https://graph.microsoft.com/v1.0/chats/{chat_id}/messages"
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    messages = response.json()['value']
    return messages[:3]  # Return only the first 3 messages

def get_messages_since(access_token, chat_id, last_timestamp):
    """
    Fetch messages from a chat that were created after a specified timestamp.

    Args:
        access_token (str): The access token to authenticate the request.
        chat_id (str): The ID of the chat from which messages need to be fetched.
        last_timestamp (datetime.datetime): The timestamp to filter messages.

    Returns:
        list: A list containing messages that were created after the specified timestamp.
    """
    messages = get_messages(access_token, chat_id)
    if last_timestamp:
        messages = [message for message in messages if parse(message['createdDateTime']) > last_timestamp]
    return messages

def get_teams_access_token(tenant_id, client_id, scopes):
    """
    Retrieve the access token from a local file. If not present, obtain a new one.
    
    Args:
        tenant_id (str): The ID of the Azure AD tenant.
        client_id (str): The application's client ID.
        scopes (str): The space-separated list of scopes for which the token is requested.

    Returns:
        str: The access token.
    """
    # Try reading the token from the file
    try:
        with open(TOKEN_FILE_PATH, 'r') as token_file:
            access_token = token_file.read().strip()
            # test the token
            test = get_chats(access_token)
            return access_token
    except (FileNotFoundError, IOError):
        # If reading fails, get a new token
        return refresh_teams_access_token(tenant_id, client_id, scopes)

def refresh_teams_access_token(tenant_id, client_id, scopes):
    """
    Obtain a new access token and save it to a local file.
    
    Args:
        tenant_id (str): The ID of the Azure AD tenant.
        client_id (str): The application's client ID.
        scopes (str): The space-separated list of scopes for which the token is requested.

    Returns:
        str: The newly obtained access token.
    """
    device_code_data = request_device_code(tenant_id, client_id, scopes)
    access_token = poll_for_token(tenant_id, client_id, device_code_data)
    
    # Save the new token to the file
    with open(TOKEN_FILE_PATH, 'w') as token_file:
        token_file.write(access_token)
    
    return access_token

def send_message_to_chat(access_token, chat_id, message_content):
    """
    Send a message to a specific chat.

    Args:
        access_token (str): The access token to authenticate the request.
        chat_id (str): The ID of the chat to which the message needs to be sent.
        message_content (str): The content of the message to be sent.

    Returns:
        dict: A dictionary containing the response from the server after sending the message.
    """
    url = f"https://graph.microsoft.com/v1.0/chats/{chat_id}/messages"
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    payload = {
        'body': {
            'contentType': 'text',
            'content': message_content
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # raise exception if any error
    return response.json()

def set_last_timestamp(timestamp):
    """
    Save the given timestamp to a file.

    This function writes the provided timestamp to a file named 'last_timestamp.txt'.

    Args:
        timestamp (datetime.datetime): The timestamp to be saved.
    """
    with open('last_timestamp.txt', 'w') as file:
        file.write(str(timestamp))


def main():
    initialize_openai()
    
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('teams_chat.log'), logging.StreamHandler()])
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    last_timestamp = get_last_timestamp()
    
    access_token = get_teams_access_token(TEAMS_TENANT_ID, TEAMS_CLIENT_ID, SCOPES)
    chats = get_chats(access_token)
    chat_gpt = next((chat for chat in chats if chat['topic'] == 'PhatGPT'), None)
    if not chat_gpt:
        logging.error("ChatGPT chat not found")
        return

    chat_id = chat_gpt['id']

    # Consider adding an exit condition or loop limit for safety
    while True:
        messages = None
        try:
            messages = get_messages_since(access_token, chat_id, last_timestamp) if last_timestamp else get_messages(access_token, chat_id)
        except Exception as e:
            logging.error(f"There was an error, retry in 60 seconds: {e}")
            time.sleep(60)
            continue
        
        for message in messages:
            content = message['body']['content'].replace("<p>", "").replace("</p>", "")
            timestamp = parse(message['createdDateTime'])

            # Check if the message starts with '<p>phatgpt' and mirror it if it does
            if content.lower().startswith('phatgpt'):
                answer_vector = ask(content, BEARER_TOKEN, SERVER_IP, max_characters_extra_info=48000)
                answer_direct_question = ask(content, BEARER_TOKEN, SERVER_IP, source="direct_search", max_characters_extra_info=48000)

                send_message_to_chat(access_token, chat_id, f"answer vectorsearch: {answer_vector}")
                send_message_to_chat(access_token, chat_id, f"answer directsearch: {answer_direct_question}")

            # Update the last timestamp
            last_timestamp = timestamp
            
        set_last_timestamp(last_timestamp)
        time.sleep(3)  # Consider making this a constant or configurable value


if __name__ == "__main__":
    main()